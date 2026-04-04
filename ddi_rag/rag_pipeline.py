"""
rag_pipeline.py — ChromaDB vector store + BioMistral-7B generation via HF Inference API.

Public API:
    load_models()                          — loads embedding model once
    build_chunk_df(df)                     — slice DataFrame into overlapping word-window chunks
    build_chroma_index(chunk_df)           — encode chunks and upsert into ChromaDB
    retrieve_chunks(query, top_k, ...)     — cosine-similarity retrieval with safe k-capping
    answer_ddi(drug_name, section, top_k, history_context)  — full RAG pipeline
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
import requests
import torch
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR, COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE,
    DEFAULT_TOP_K, EMBED_BATCH_SIZE, EMBEDDING_MODEL,
    GENERATION_MAX_NEW, GROQ_API_KEY, GROQ_MODEL, GROQ_TIMEOUT,
    MAX_TOP_K, SYSTEM_PROMPT, TEXT_COLS,
)

log = logging.getLogger("ddi.rag")

# ── Module-level singletons ───────────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None
_chroma_collection = None

# Control-character stripper
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def safe_str(val) -> str:
    return _CTRL.sub("", str(val) if val is not None else "")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> None:
    """
    Load the SentenceTransformer embedding model into memory.
    BioMistral generation is handled via HF Inference API — no local model needed.
    Call once at startup; subsequent calls are no-ops.
    """
    global _embedding_model

    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading embedding model: %s on %s", EMBEDDING_MODEL, device.upper())
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        if device == "cuda":
            log.info("GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)
        log.info("Embedding model ready.")

    if GROQ_API_KEY:
        log.info("Groq API configured — model: %s", GROQ_MODEL)
    else:
        log.warning(
            "GROQ_API_KEY not set. Generation will return a placeholder. "
            "Sign up at console.groq.com and add GROQ_API_KEY to your .env file."
        )


def _get_collection():
    """Return (and cache) the persistent ChromaDB collection handle."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            'ChromaDB collection "%s" opened (%d vectors).',
            COLLECTION_NAME, _chroma_collection.count(),
        )
    return _chroma_collection


# ── Chunking ──────────────────────────────────────────────────────────────────

def _build_full_document(row: pd.Series, valid_cols: List[str]) -> str:
    parts = [
        f"Generic Name: {row.get('final_generic_name', '')}",
        f"Brand Name: {row.get('openfda_brand_name', '')}",
        f"Product Type: {row.get('openfda_product_type', '')}",
        f"Route: {row.get('openfda_route', '')}",
    ]
    for col in valid_cols:
        val = row.get(col, "")
        if val and isinstance(val, str) and val.strip():
            parts.append(f"{col.replace('_', ' ').title()}:\n{val}")
    return "\n\n".join(parts)


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
    Returns a flat DataFrame with columns:
        doc_id, generic_name, brand_name, product_type, route, section, text
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


# ── Index building ────────────────────────────────────────────────────────────

def build_chroma_index(chunk_df: pd.DataFrame, rebuild: bool = False) -> None:
    """
    Encode all chunks and upsert them into the ChromaDB collection.
    """
    if _embedding_model is None:
        raise RuntimeError("Call load_models() before build_chroma_index().")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Existing collection deleted — rebuilding.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    global _chroma_collection
    _chroma_collection = collection

    total = len(chunk_df)
    log.info("Indexing %d chunks into ChromaDB ...", total)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch      = chunk_df.iloc[start : start + EMBED_BATCH_SIZE]
        texts      = batch["text"].tolist()
        embeddings = _embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        collection.upsert(
            ids        = batch["doc_id"].tolist(),
            documents  = texts,
            embeddings = embeddings,
            metadatas  = batch[
                ["generic_name", "brand_name", "product_type", "route", "section"]
            ].to_dict(orient="records"),
        )
        if (start // EMBED_BATCH_SIZE) % 10 == 0:
            log.info("  Upserted %d / %d", min(start + EMBED_BATCH_SIZE, total), total)

    log.info("Index built. Total vectors: %d", collection.count())


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query     : str,
    top_k     : int           = DEFAULT_TOP_K,
    drug_name : Optional[str] = None,
    section   : Optional[str] = None,
) -> pd.DataFrame:
    """
    Query ChromaDB for the top-k most relevant chunks.
    Returns a DataFrame with columns:
        generic_name, brand_name, product_type, route, section, text, score
    """
    if _embedding_model is None:
        raise RuntimeError("Call load_models() first.")

    collection = _get_collection()

    try:
        clauses = []
        if drug_name:
            clauses.append({"generic_name": {"$eq": drug_name.strip().lower()}})
        if section:
            clauses.append({"section": {"$eq": section.strip().lower()}})

        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        total = collection.count()
        if total == 0:
            return pd.DataFrame()

        # Cap top_k to total available — avoids ChromaDB crash
        safe_k    = min(top_k, total)
        query_emb = _embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        # Try with filters first; if ChromaDB raises (fewer docs than k), retry
        for attempt_k in [safe_k, max(1, safe_k // 2), 1]:
            try:
                results = collection.query(
                    query_embeddings = query_emb,
                    n_results        = attempt_k,
                    where            = where,
                    include          = ["documents", "metadatas", "distances"],
                )
                break
            except Exception as e:
                if "Number of requested results" in str(e) and attempt_k > 1:
                    log.warning("Reducing n_results to %d due to: %s", attempt_k // 2, e)
                    continue
                raise

        records = [
            {
                "generic_name": m.get("generic_name", ""),
                "brand_name"  : m.get("brand_name",   ""),
                "product_type": m.get("product_type", ""),
                "route"       : m.get("route",        ""),
                "section"     : m.get("section",      ""),
                "text"        : doc,
                "score"       : round(float(1 - dist), 4),
            }
            for doc, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return pd.DataFrame(records)

    except Exception:
        log.exception("retrieve_chunks failed")
        return pd.DataFrame()


# ── Groq API generation ───────────────────────────────────────────────────────

def _call_groq_api(prompt: str) -> str:
    """
    Send the prompt to Groq API (llama-3.1-8b-instant by default).
    Fast, free tier, reliable. Returns the generated text string.
    """
    if not GROQ_API_KEY:
        return (
            "GROQ_API_KEY not configured. "
            "Sign up at console.groq.com and add GROQ_API_KEY to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens":   GENERATION_MAX_NEW,
        "temperature":  0.3,
        "stream":       False,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=GROQ_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return safe_str(
            data["choices"][0]["message"]["content"]
        ).strip()

    except requests.Timeout:
        log.error("Groq API timed out after %ds", GROQ_TIMEOUT)
        return "Generation timed out. Please try again."
    except Exception as exc:
        log.exception("Groq API call failed")
        return f"Generation error: {safe_str(exc)}"


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _cached_answer(
    drug_name      : Optional[str],
    section        : Optional[str],
    top_k          : int,
    history_context: str,          # empty string = no history
) -> Dict:
    query     = f"What are the drug interactions and warnings for {drug_name}?"
    retrieved = retrieve_chunks(query=query, top_k=top_k,
                                drug_name=drug_name, section=section)

    if retrieved.empty and drug_name:
        log.info('No exact match for "%s", falling back to global search.', drug_name)
        retrieved = retrieve_chunks(query=query, top_k=top_k, section=section)

    if retrieved.empty:
        return {"answer": "No relevant FDA label evidence found for this drug.", "sources": []}

    # Limit to top 3 chunks, 300 chars each — keeps prompt small = faster response
    context = "\n".join(
        f"[{i}] {row['section'].replace('_',' ').title()}: {row['text'][:300]}"
        for i, (_, row) in enumerate(retrieved.head(3).iterrows(), 1)
    )

    history_block = (
        f"Patient history: {history_context}\n"
        if history_context else ""
    )

    full_prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"{history_block}"
        f"FDA Evidence:\n{context}\n\n"
        f"Q: {query}\nA:"
    )

    answer = _call_groq_api(full_prompt)

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
    """
    Full RAG pipeline: retrieve → prompt (with optional patient history) → generate.

    Args:
        drug_name:       generic drug name to filter retrieval (optional)
        section:         FDA section to filter on (default: drug_interactions)
        top_k:           number of chunks to retrieve (capped at MAX_TOP_K)
        history_context: formatted string of patient's past medications

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    if _embedding_model is None:
        raise RuntimeError("Call load_models() before answer_ddi().")

    return _cached_answer(
        drug_name       = drug_name,
        section         = section,
        top_k           = min(top_k, MAX_TOP_K),
        history_context = history_context,
    )
