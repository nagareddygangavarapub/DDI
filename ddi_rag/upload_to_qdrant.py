"""
upload_to_qdrant.py — One-time migration: local ChromaDB → Qdrant Cloud.

Run from your local machine (where chroma_ddi_db/ exists):
    cd ddi_rag
    python upload_to_qdrant.py

Supports RESUME — if interrupted, run again and it continues from where it stopped.
"""

import json
import logging
import sys
from pathlib import Path

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHROMA_DIR, COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("qdrant.upload")

BATCH_SIZE     = 500
VECTOR_DIM     = 384
DRUG_NAMES_OUT = Path(__file__).resolve().parent / "drug_names.json"


def main():
    # ── Connect to local ChromaDB ─────────────────────────────────────────────
    log.info("Opening local ChromaDB at: %s", CHROMA_DIR)
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    col    = chroma.get_collection(COLLECTION_NAME)
    total  = col.count()
    log.info("ChromaDB — %d vectors total", total)

    # ── Connect to Qdrant Cloud ───────────────────────────────────────────────
    if not QDRANT_API_KEY:
        log.error("QDRANT_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    log.info("Connecting to Qdrant Cloud: %s", QDRANT_URL)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # ── Create collection only if it doesn't exist ───────────────────────────
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        log.info("Creating Qdrant collection '%s'...", COLLECTION_NAME)
        qdrant.create_collection(
            collection_name     = COLLECTION_NAME,
            vectors_config      = VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type       = ScalarType.INT8,
                    quantile   = 0.99,
                    always_ram = True,
                )
            ),
        )
        already_uploaded = 0
    else:
        # ── RESUME: find how many points already uploaded ─────────────────────
        already_uploaded = qdrant.get_collection(COLLECTION_NAME).points_count
        log.info("Collection exists — %d / %d already uploaded. Resuming...",
                 already_uploaded, total)
        if already_uploaded >= total:
            log.info("All vectors already uploaded! Skipping to drug_names.json step.")
            already_uploaded = total

    # ── Upload remaining batches ──────────────────────────────────────────────
    b2g:    dict = {}
    gnames: set  = set()
    uploaded     = already_uploaded

    for offset in range(0, total, BATCH_SIZE):
        # Collect drug names from ALL batches (even already-uploaded ones)
        # but only upload batches not yet in Qdrant
        batch = col.get(
            limit   = BATCH_SIZE,
            offset  = offset,
            include = ["embeddings", "metadatas", "documents"],
        )

        # Always collect drug names
        for meta in batch["metadatas"]:
            g = str(meta.get("generic_name", "")).strip().lower()
            b = str(meta.get("brand_name",   "")).strip().lower()
            if g and g != "nan":
                gnames.add(g)
            if b and b not in ("nan", ""):
                b2g[b] = g

        # Skip batches already uploaded
        if offset < already_uploaded:
            continue

        points = []
        for i, (doc_id, emb, meta, doc) in enumerate(
            zip(batch["ids"], batch["embeddings"],
                batch["metadatas"], batch["documents"])
        ):
            payload = {**meta, "text": doc, "doc_id": doc_id}
            points.append(PointStruct(id=offset + i, vector=emb, payload=payload))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += len(points)
        log.info("  Uploaded %d / %d", uploaded, total)

    log.info("Upload complete — %d vectors in Qdrant.", uploaded)

    # ── Save drug names JSON ──────────────────────────────────────────────────
    lookup = {"brand_to_generic": b2g, "generic_names": sorted(gnames)}
    with open(DRUG_NAMES_OUT, "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)

    size_kb = DRUG_NAMES_OUT.stat().st_size / 1024
    log.info("Saved drug_names.json (%.1f KB, %d generics, %d brands)",
             size_kb, len(gnames), len(b2g))
    log.info("All done! Now commit drug_names.json to GitHub.")


if __name__ == "__main__":
    main()
