"""
rag_pipeline.py — Qdrant Cloud vector store + Groq LLM generation.

Public API:
    load_models()                          — loads embedding model once
    build_chunk_df(df)                     — slice DataFrame into overlapping word-window chunks
    retrieve_chunks(query, top_k, ...)     — cosine-similarity retrieval via Qdrant Cloud
    answer_ddi(drug_name, section, top_k, history_context)  — full RAG pipeline
    answer_general(query, history_context) — general medical assistant (no retrieval)
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE,
    DEFAULT_TOP_K, EMBED_BATCH_SIZE, EMBEDDING_MODEL,
    GENERAL_SYSTEM_PROMPT, GENERATION_MAX_NEW, GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT,
    MAX_TOP_K, QDRANT_API_KEY, QDRANT_URL, SYSTEM_PROMPT, TEXT_COLS,
)

log = logging.getLogger("ddi.rag")

# ── Module-level singletons ───────────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None
_qdrant_client:   Optional[QdrantClient]        = None

# Control-character stripper
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def safe_str(val) -> str:
    return _CTRL.sub("", str(val) if val is not None else "")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    """Load the SentenceTransformer embedding model. Call once at startup."""
    global _embedding_model

    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, device.upper())
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        log.info("Embedding model ready.")

    if GROQ_API_KEY:
        log.info("Groq API configured — model: %s", GROQ_MODEL)
    else:
        log.warning("GROQ_API_KEY not set. Add it to your .env file.")


def _get_qdrant() -> QdrantClient:
    """Return (and cache) the Qdrant Cloud client."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_API_KEY is not set. Add it to your .env / Streamlit secrets.")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        try:
            info = _qdrant_client.get_collection(COLLECTION_NAME)
            log.info('Qdrant collection "%s" — %d vectors.', COLLECTION_NAME,
                     info.points_count)
        except Exception:
            log.warning('Qdrant collection "%s" not found or not yet uploaded.', COLLECTION_NAME)
    return _qdrant_client


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping word-level windows."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slice every FDA label in `df` into overlapping word-window chunks.
    Returns a flat DataFrame used for lookup table building.
    """
    valid_cols = [c for c in TEXT_COLS if c in df.columns]
    documents: List[Dict] = []

    for idx, row in df.iterrows():
        generic_name = safe_str(row.get("final_generic_name", "") or "").strip()
        brand_name   = safe_str(row.get("openfda_brand_name",  "") or "").strip()
        product_type = safe_str(row.get("openfda_product_type","") or "").strip()
        route        = safe_str(row.get("openfda_route",       "") or "").strip()

        for col in valid_cols:
            section_content = row.get(col, "")
            if pd.isna(section_content) or not str(section_content).strip():
                continue
            header = f"{col.replace('_', ' ').title()}: "
            text   = header + str(section_content)
            for ci, chunk in enumerate(_chunk_text(text)):
                documents.append({
                    "doc_id"      : f"{idx}_{col}_{ci}",
                    "generic_name": generic_name,
                    "brand_name"  : brand_name,
                    "product_type": product_type,
                    "route"       : route,
                    "section"     : col,
                    "text"        : chunk,
                })

    chunk_df = pd.DataFrame(documents)
    log.info("Built chunk_df: %d chunks from %d rows.", len(chunk_df), len(df))
    return chunk_df


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query     : str,
    top_k     : int           = DEFAULT_TOP_K,
    drug_name : Optional[str] = None,
    section   : Optional[str] = None,
) -> pd.DataFrame:
    """
    Query Qdrant Cloud for the top-k most relevant chunks.
    Returns a DataFrame with columns:
        generic_name, brand_name, product_type, route, section, text, score
    """
    if _embedding_model is None:
        raise RuntimeError("Call load_models() first.")

    client = _get_qdrant()

    try:
        # Build filter conditions
        conditions = []
        if drug_name:
            conditions.append(
                FieldCondition(key="generic_name",
                               match=MatchValue(value=drug_name.strip().lower()))
            )
        if section:
            conditions.append(
                FieldCondition(key="section",
                               match=MatchValue(value=section.strip().lower()))
            )

        query_filter = Filter(must=conditions) if conditions else None

        query_emb = _embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()[0]

        # Try with filter first; fall back to no filter if no results
        results = client.search(
            collection_name = COLLECTION_NAME,
            query_vector    = query_emb,
            query_filter    = query_filter,
            limit           = min(top_k, MAX_TOP_K),
            with_payload    = True,
        )

        # If filtered search returned nothing and drug_name was set, try globally
        if not results and drug_name:
            log.info('No filtered match for "%s", falling back to global search.', drug_name)
            results = client.search(
                collection_name = COLLECTION_NAME,
                query_vector    = query_emb,
                limit           = min(top_k, MAX_TOP_K),
                with_payload    = True,
            )

        if not results:
            return pd.DataFrame()

        records = [
            {
                "generic_name": safe_str(r.payload.get("generic_name", "")),
                "brand_name"  : safe_str(r.payload.get("brand_name",   "")),
                "product_type": safe_str(r.payload.get("product_type", "")),
                "route"       : safe_str(r.payload.get("route",        "")),
                "section"     : safe_str(r.payload.get("section",      "")),
                "text"        : safe_str(r.payload.get("text",         "")),
                "score"       : round(float(r.score), 4),
            }
            for r in results
        ]
        return pd.DataFrame(records)

    except Exception:
        log.exception("retrieve_chunks failed")
        return pd.DataFrame()


# ── Groq API generation ───────────────────────────────────────────────────────

def _call_groq_api(
    messages: list,
    temperature: float = 0.4,
    max_tokens: int = None,
) -> str:
    """Send a messages list to Groq API and return the reply string."""
    if not GROQ_API_KEY:
        return (
            "GROQ_API_KEY not configured. "
            "Add GROQ_API_KEY to your .env file or Streamlit secrets."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens or GENERATION_MAX_NEW,
        "temperature": temperature,
        "stream":      False,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers = headers,
            json    = payload,
            timeout = GROQ_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return safe_str(data["choices"][0]["message"]["content"]).strip()

    except requests.Timeout:
        log.error("Groq API timed out after %ds", GROQ_TIMEOUT)
        return "Generation timed out. Please try again."
    except Exception as exc:
        log.exception("Groq API call failed")
        return f"Generation error: {safe_str(exc)}"


@lru_cache(maxsize=256)
def answer_general(
    query          : str,
    history_context: str = "",
) -> Dict:
    """
    General medical assistant mode — no RAG retrieval.
    Returns {"answer": str, "sources": []}
    """
    med_block = (
        f"\nPatient's current medications:\n{history_context}\n"
        if history_context else ""
    )
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
        {"role": "user",   "content": f"{med_block}\nQuestion: {query}"},
    ]
    answer = _call_groq_api(messages, temperature=0.5, max_tokens=600)
    return {"answer": answer, "sources": []}


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _cached_answer(
    drug_name      : Optional[str],
    section        : Optional[str],
    top_k          : int,
    history_context: str,
) -> Dict:
    query     = f"What are the drug interactions and warnings for {drug_name}?"
    retrieved = retrieve_chunks(query=query, top_k=top_k,
                                drug_name=drug_name, section=section)

    if retrieved.empty:
        return {"answer": "No relevant FDA label evidence found for this drug.", "sources": []}

    # Top 3 chunks, 300 chars each — keeps prompt small = faster response
    context = "\n".join(
        f"[{i}] {row['section'].replace('_',' ').title()}: {row['text'][:300]}"
        for i, (_, row) in enumerate(retrieved.head(3).iterrows(), 1)
    )

    med_block = (
        f"Patient's current medications:\n{history_context}\n\n"
        if history_context else ""
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{med_block}FDA Evidence:\n{context}\n\nQ: {query}"},
    ]

    answer = _call_groq_api(messages)

    return {
        "answer": answer,
        "sources": [
            {
                "generic_name": safe_str(r.get("generic_name", "")),
                "brand_name"  : safe_str(r.get("brand_name",   "")),
                "product_type": safe_str(r.get("product_type", "")),
                "route"       : safe_str(r.get("route",        "")),
                "section"     : safe_str(r.get("section",      "")),
                "score"       : float(r.get("score", 0.0)),
                "text"        : safe_str(r.get("text",         "")),
            }
            for _, r in retrieved.iterrows()
        ],
    }


def answer_ddi(
    drug_name      : Optional[str] = None,
    section        : str           = "drug_interactions",
    top_k          : int           = DEFAULT_TOP_K,
    history_context: str           = "",
) -> Dict:
    """Full RAG pipeline: retrieve → prompt → generate via Groq."""
    if _embedding_model is None:
        load_models()

    return _cached_answer(
        drug_name       = drug_name,
        section         = section,
        top_k           = min(top_k, MAX_TOP_K),
        history_context = history_context,
    )
