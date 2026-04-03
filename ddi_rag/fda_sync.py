"""
fda_sync.py — Incremental sync from openFDA Inference API into ChromaDB.

How it works:
  1. Fetch drug labels updated since `last_sync_date` from openFDA API
  2. For each updated drug, delete its old ChromaDB chunks
  3. Re-embed and upsert new chunks
  4. Persist the new last_sync_date

Scheduler (APScheduler) calls `run_sync()` nightly at SYNC_HOUR.
Can also be triggered manually via POST /api/sync.
"""

import logging
import os
from datetime import date, datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_BATCH_SIZE,
    OPENFDA_BASE_URL, OPENFDA_PAGE_SIZE, SYNC_HOUR, TEXT_COLS,
)
from rag_pipeline import _chunk_text, _get_collection, safe_str

log = logging.getLogger("ddi.sync")

# File to persist the last successful sync date
_SYNC_STATE_FILE = os.path.join(os.path.dirname(__file__), ".last_sync_date")


# ── Sync state ────────────────────────────────────────────────────────────────

def _load_last_sync_date() -> Optional[str]:
    """Return the last sync date as 'YYYYMMDD' string, or None."""
    try:
        if os.path.exists(_SYNC_STATE_FILE):
            return open(_SYNC_STATE_FILE).read().strip() or None
    except Exception:
        pass
    return None


def _save_last_sync_date(date_str: str):
    try:
        with open(_SYNC_STATE_FILE, "w") as f:
            f.write(date_str)
    except Exception as exc:
        log.warning("Could not save sync date: %s", exc)


# ── openFDA fetching ──────────────────────────────────────────────────────────

def _fetch_page(since: Optional[str], skip: int) -> dict:
    """
    Fetch one page of drug labels from openFDA.

    Args:
        since: 'YYYYMMDD' string — fetch labels updated on or after this date.
               None = fetch all (full sync, use sparingly).
        skip:  pagination offset
    """
    params = {"limit": OPENFDA_PAGE_SIZE, "skip": skip}

    if since:
        today = date.today().strftime("%Y%m%d")
        params["search"] = f"effective_time:[{since}+TO+{today}]"

    try:
        resp = requests.get(OPENFDA_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return {}   # no results for this date range
        raise
    except Exception as exc:
        log.error("openFDA fetch failed (skip=%d): %s", skip, exc)
        return {}


def _parse_label(record: dict) -> Optional[dict]:
    """
    Extract relevant fields from one openFDA drug label record.
    Returns None if the record has no useful drug data.
    """
    openfda = record.get("openfda", {})

    generic_names = openfda.get("generic_name", [])
    brand_names   = openfda.get("brand_name",   [])
    product_types = openfda.get("product_type", [])
    routes        = openfda.get("route",        [])

    generic_name = generic_names[0].lower().strip() if generic_names else ""
    brand_name   = brand_names[0].lower().strip()   if brand_names   else generic_name
    product_type = product_types[0].lower().strip() if product_types else ""
    route        = routes[0].lower().strip()        if routes        else ""

    if not generic_name:
        return None

    row = {
        "final_generic_name":   generic_name,
        "openfda_brand_name":   brand_name,
        "openfda_product_type": product_type,
        "openfda_route":        route,
    }

    for col in TEXT_COLS:
        values = record.get(col, [])
        row[col] = " ".join(values).strip() if values else ""

    # Only keep records that have at least one useful text section
    if not any(row.get(c, "") for c in TEXT_COLS):
        return None

    return row


# ── ChromaDB update ───────────────────────────────────────────────────────────

def _delete_drug_chunks(collection, drug_name: str):
    """Delete all ChromaDB chunks for a given generic drug name."""
    try:
        existing = collection.get(
            where={"generic_name": {"$eq": drug_name}}, limit=9_999
        )
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            log.debug("Deleted %d old chunks for '%s'.", len(ids), drug_name)
    except Exception as exc:
        log.warning("Could not delete chunks for '%s': %s", drug_name, exc)


def _upsert_drug(collection, embedding_model, row: dict, row_index: int):
    """Build chunks for one drug record and upsert into ChromaDB."""
    from rag_pipeline import build_chunk_df
    import pandas as _pd

    drug_df  = _pd.DataFrame([row])
    chunk_df = build_chunk_df(drug_df)

    if chunk_df.empty:
        return

    # Re-index doc_ids using row_index to avoid collisions
    chunk_df["doc_id"] = [
        f"sync_{row_index}_{col}_{ci}"
        for ci, col in enumerate(chunk_df["doc_id"].tolist())
    ]

    texts      = chunk_df["text"].tolist()
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids        = chunk_df["doc_id"].tolist(),
        documents  = texts,
        embeddings = embeddings,
        metadatas  = chunk_df[
            ["generic_name", "brand_name", "product_type", "route", "section"]
        ].to_dict(orient="records"),
    )


# ── Main sync entry point ─────────────────────────────────────────────────────

def run_sync(full: bool = False) -> dict:
    """
    Fetch updated drug labels from openFDA and update ChromaDB.

    Args:
        full: if True, ignore last_sync_date and fetch everything
              (WARNING: very slow — use only for initial rebuild)

    Returns:
        {"synced": int, "skipped": int, "errors": int, "sync_date": str}
    """
    from rag_pipeline import _embedding_model

    if _embedding_model is None:
        return {"error": "Embedding model not loaded. Call load_models() first."}

    since       = None if full else _load_last_sync_date()
    collection  = _get_collection()
    today_str   = date.today().strftime("%Y%m%d")

    log.info(
        "Starting FDA sync — since: %s, full: %s",
        since or "beginning", full,
    )

    synced = skipped = errors = 0
    skip   = 0

    while True:
        data = _fetch_page(since=since, skip=skip)

        if not data or "results" not in data:
            break

        records = data["results"]
        if not records:
            break

        for i, record in enumerate(records):
            row = _parse_label(record)
            if row is None:
                skipped += 1
                continue
            try:
                _delete_drug_chunks(collection, row["final_generic_name"])
                _upsert_drug(collection, _embedding_model, row, skip + i)
                synced += 1
            except Exception as exc:
                log.warning(
                    "Error syncing '%s': %s",
                    row.get("final_generic_name", "?"), exc,
                )
                errors += 1

        total_available = data.get("meta", {}).get("results", {}).get("total", 0)
        skip += OPENFDA_PAGE_SIZE

        log.info(
            "Sync progress — fetched: %d / %d, synced: %d, skip: %d, errors: %d",
            min(skip, total_available), total_available, synced, skipped, errors,
        )

        if skip >= total_available:
            break

    _save_last_sync_date(today_str)
    log.info(
        "FDA sync complete — synced: %d, skipped: %d, errors: %d",
        synced, skipped, errors,
    )

    return {
        "synced":    synced,
        "skipped":   skipped,
        "errors":    errors,
        "sync_date": today_str,
    }


# ── APScheduler setup ─────────────────────────────────────────────────────────

def start_scheduler():
    """
    Start APScheduler to run run_sync() nightly at SYNC_HOUR.
    Call once at application startup (from run.py).
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(timezone="UTC")
        scheduler.add_job(
            run_sync,
            trigger="cron",
            hour=SYNC_HOUR,
            minute=0,
            id="fda_nightly_sync",
            replace_existing=True,
        )
        scheduler.start()
        log.info("FDA nightly sync scheduler started (runs at %02d:00 UTC).", SYNC_HOUR)
        return scheduler
    except Exception as exc:
        log.error("Could not start scheduler: %s", exc)
        return None
