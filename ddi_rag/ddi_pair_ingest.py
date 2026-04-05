"""
ddi_pair_ingest.py — Clean and index the DDI pairs dataset into ChromaDB.

Reads fully_processed_dataset.csv and upserts each drug pair as a vector
entry in the same ChromaDB collection used by the main RAG pipeline.

Doc ID format : ddi_{Drug1_Label}_{Drug2_Label}
Text format   : "{drug1} and {drug2}: {cleaned_description}"
Metadata      : generic_name=drug1, section=drug_interactions, source=ddi_pairs

Run once after placing fully_processed_dataset.csv in the ddi_rag/ folder:
    cd ddi_rag
    python ddi_pair_ingest.py
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# Allow running directly from ddi_rag/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_BATCH_SIZE, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ddi.pair_ingest")

DDI_PAIRS_CSV = Path(__file__).resolve().parent.parent / "data" / "datasets" / "fully_processed_dataset.csv"

_JUNK_DRUG  = "unknown_drug"
_JUNK_DESC  = "nodescription"
_WHITESPACE = re.compile(r"\s+")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    return _WHITESPACE.sub(" ", str(name).lower().strip())


def load_and_clean_pairs(csv_path: Path) -> pd.DataFrame:
    """
    Load fully_processed_dataset.csv, normalise drug names, and drop junk rows.
    Uses the already-lowercased Cleaned_Description column as the text.
    """
    df = pd.read_csv(csv_path)
    original = len(df)
    log.info("Loaded %d rows from %s", original, csv_path.name)

    df["drug1"] = df["Drug 1"].apply(_clean_name)
    df["drug2"] = df["Drug 2"].apply(_clean_name)
    df["description"] = df["Cleaned_Description"].fillna("").str.strip()

    # Drop rows with placeholder / junk values
    mask = (
        (df["drug1"] != _JUNK_DRUG) &
        (df["drug2"] != _JUNK_DRUG) &
        (df["description"] != _JUNK_DESC) &
        (df["description"] != "")
    )
    df = df[mask].reset_index(drop=True)
    log.info("After filtering junk: %d rows remain (%d dropped)", len(df), original - len(df))
    return df


# ── Chunk building ────────────────────────────────────────────────────────────

def build_pair_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each DDI pair row into a ChromaDB-ready record.

    Pairs sharing the same Drug1_Label + Drug2_Label (multiple interaction
    descriptions for the same pair) are merged into one combined text entry
    so doc_ids remain unique.

    Text format:
        "amoxicillin and clarithromycin: <desc1>. <desc2>."

    generic_name is set to drug1 so the existing retrieve_chunks() metadata
    filter works when drug1 is queried directly. Drug2 is still found via
    embedding similarity because it appears in the text.
    """
    # Group duplicate pairs — merge all descriptions with ". " separator
    grouped = (
        df.groupby(["Drug1_Label", "Drug2_Label"], sort=False)
        .agg(
            drug1       = ("drug1",       "first"),
            drug2       = ("drug2",       "first"),
            description = ("description", lambda x: ". ".join(x.unique())),
        )
        .reset_index()
    )

    merged = len(df) - len(grouped)
    if merged:
        log.info("Merged %d duplicate pairs into combined descriptions", merged)

    records = []
    for _, row in grouped.iterrows():
        drug1  = row["drug1"]
        drug2  = row["drug2"]
        desc   = row["description"]
        d1_id  = int(row["Drug1_Label"])
        d2_id  = int(row["Drug2_Label"])

        records.append({
            "doc_id"      : f"ddi_{d1_id}_{d2_id}",
            "generic_name": drug1,
            "brand_name"  : "",
            "product_type": "",
            "route"       : "",
            "section"     : "drug_interactions",
            "text"        : f"{drug1} and {drug2}: {desc}",
        })

    result = pd.DataFrame(records)
    log.info("Built %d unique pair records", len(result))
    return result


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_pairs(csv_path: Path = DDI_PAIRS_CSV) -> None:
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    # ── Load embedding model ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, device.upper())
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device == "cuda":
        log.info("GPU: %s (%.1f GB VRAM)",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Load and clean data ───────────────────────────────────────────────────
    df       = load_and_clean_pairs(csv_path)
    pair_df  = build_pair_records(df)
    total    = len(pair_df)

    # ── Open ChromaDB collection ──────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )
    before = collection.count()
    log.info('Collection "%s" — %d vectors before ingest', COLLECTION_NAME, before)

    # ── Upsert in batches ─────────────────────────────────────────────────────
    log.info("Upserting %d DDI pair vectors (batch size %d) ...", total, EMBED_BATCH_SIZE)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = pair_df.iloc[start : start + EMBED_BATCH_SIZE]
        texts = batch["text"].tolist()

        embeddings = model.encode(
            texts,
            show_progress_bar = False,
            convert_to_numpy  = True,
            normalize_embeddings = True,
        ).tolist()

        collection.upsert(
            ids        = batch["doc_id"].tolist(),
            documents  = texts,
            embeddings = embeddings,
            metadatas  = batch[
                ["generic_name", "brand_name", "product_type", "route", "section"]
            ].to_dict(orient="records"),
        )

        done = min(start + EMBED_BATCH_SIZE, total)
        log.info("  %d / %d  (%.0f%%)", done, total, done * 100 / total)

    after = collection.count()
    log.info(
        "Ingest complete. Vectors: %d → %d  (+%d added/updated)",
        before, after, after - before,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest_pairs()
