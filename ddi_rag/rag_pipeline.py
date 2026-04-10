"""
rag_pipeline.py — Qdrant Cloud vector store + CRAG pipeline + Groq LLM generation.

CRAG (Corrective RAG) enhances standard RAG with a relevance grading step:
    1. Retrieve   — fetch top-K chunks from Qdrant Cloud
    2. Grade      — score retrieval quality (correct / ambiguous / incorrect)
    3. Correct    — apply corrective action based on grade:
                    correct   → use chunks as-is
                    ambiguous → broaden search, merge and re-rank results
                    incorrect → rewrite query via Groq, retry retrieval
    4. Generate   — produce answer using verified, high-quality context

Public API:
    load_models()                                          — load embedding model once
    retrieve_chunks(query, top_k, drug_name, section)      — Qdrant retrieval
    answer_ddi(drug_name, section, top_k, history_context) — full CRAG pipeline
    answer_general(query, history_context)                 — general assistant (no retrieval)
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

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

log = logging.getLogger("ddi.crag")

# ── Module-level singletons ───────────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None
_qdrant_client:   Optional[QdrantClient]        = None

# ── CRAG relevance thresholds ─────────────────────────────────────────────────
RELEVANCE_HIGH = 0.65   # score ≥ this → "correct"  (use chunks as-is)
RELEVANCE_LOW  = 0.40   # score < this → "incorrect" (rewrite query + retry)
                         # between the two  → "ambiguous" (broaden search)

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
        log.warning("GROQ_API_KEY not set. Add it to your .env / Streamlit secrets.")


def _get_qdrant() -> QdrantClient:
    """Return (and cache) the Qdrant Cloud client."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_API_KEY is not set.")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        try:
            info = _qdrant_client.get_collection(COLLECTION_NAME)
            log.info('Qdrant collection "%s" — %d vectors.', COLLECTION_NAME,
                     info.points_count)
        except Exception:
            log.warning('Qdrant collection "%s" not found.', COLLECTION_NAME)
    return _qdrant_client


# ── Chunking (kept for local index building if needed) ────────────────────────

def _chunk_text(text: str) -> List[str]:
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
            text = f"{col.replace('_', ' ').title()}: {section_content}"
            for ci, chunk in enumerate(_chunk_text(text)):
                documents.append({
                    "doc_id": f"{idx}_{col}_{ci}", "generic_name": generic_name,
                    "brand_name": brand_name, "product_type": product_type,
                    "route": route, "section": col, "text": chunk,
                })

    chunk_df = pd.DataFrame(documents)
    log.info("Built chunk_df: %d chunks from %d rows.", len(chunk_df), len(df))
    return chunk_df


# ── Base retrieval ────────────────────────────────────────────────────────────

def retrieve_chunks(
    query     : str,
    top_k     : int           = DEFAULT_TOP_K,
    drug_name : Optional[str] = None,
    section   : Optional[str] = None,
) -> pd.DataFrame:
    """
    Query Qdrant Cloud for the top-k most relevant chunks.
    Returns a DataFrame with: generic_name, brand_name, section, text, score.
    """
    if _embedding_model is None:
        raise RuntimeError("Call load_models() first.")

    client = _get_qdrant()

    try:
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
        query_emb    = _embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()[0]

        results = client.search(
            collection_name = COLLECTION_NAME,
            query_vector    = query_emb,
            query_filter    = query_filter,
            limit           = min(top_k, MAX_TOP_K),
            with_payload    = True,
        )

        if not results:
            return pd.DataFrame()

        return pd.DataFrame([
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
        ])

    except Exception:
        log.exception("retrieve_chunks failed")
        return pd.DataFrame()


# ── CRAG components ───────────────────────────────────────────────────────────

def _grade_retrieval(chunks: pd.DataFrame) -> str:
    """
    Evaluate retrieval quality based on cosine similarity scores.

    Returns:
        'correct'   — top chunk score ≥ RELEVANCE_HIGH  → use as-is
        'ambiguous' — score between thresholds           → broaden search
        'incorrect' — top chunk score < RELEVANCE_LOW    → rewrite query
    """
    if chunks.empty:
        return "incorrect"

    best_score = float(chunks["score"].max())

    if best_score >= RELEVANCE_HIGH:
        grade = "correct"
    elif best_score < RELEVANCE_LOW:
        grade = "incorrect"
    else:
        grade = "ambiguous"

    log.info("CRAG grade: %s (best_score=%.4f)", grade, best_score)
    return grade


def _rewrite_query(original_query: str) -> str:
    """
    CRAG corrective action for 'incorrect' grade.
    Uses Groq to rephrase the query for better vector retrieval.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a query rewriter for a medical FDA drug database. "
                "Rewrite the given query to be more specific, using correct "
                "pharmacological terminology to improve search results. "
                "Return only the rewritten query — no explanation."
            ),
        },
        {"role": "user", "content": f"Original query: {original_query}"},
    ]
    rewritten = _call_groq_api(messages, temperature=0.3, max_tokens=80)
    log.info("CRAG query rewrite: '%s' → '%s'", original_query, rewritten)
    return rewritten


def _crag_retrieve(
    query     : str,
    drug_name : Optional[str],
    section   : Optional[str],
    top_k     : int,
) -> Tuple[pd.DataFrame, str]:
    """
    Full CRAG retrieval pipeline.

    Returns:
        (chunks DataFrame, retrieval_status string)

    Retrieval status values:
        'correct'                 — high confidence, used as-is
        'ambiguous:broadened'     — merged narrow + broad search results
        'incorrect:rewritten'     — query rewritten, retried globally
        'incorrect:no_evidence'   — no relevant chunks found after correction
    """
    # ── Step 1: Initial retrieval ─────────────────────────────────────────────
    chunks = retrieve_chunks(query, top_k, drug_name, section)
    grade  = _grade_retrieval(chunks)

    # ── Step 2a: Correct — high confidence ───────────────────────────────────
    if grade == "correct":
        return chunks, "correct"

    # ── Step 2b: Ambiguous — broaden search ───────────────────────────────────
    if grade == "ambiguous":
        log.info("CRAG ambiguous — broadening search (no metadata filter).")
        broad = retrieve_chunks(query, top_k * 2)   # no drug_name/section filter

        if not broad.empty:
            # Merge narrow + broad results, deduplicate, re-rank by score
            combined = (
                pd.concat([chunks, broad])
                .drop_duplicates(subset=["text"])
                .sort_values("score", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )
            return combined, "ambiguous:broadened"

        return chunks, "ambiguous:broadened"   # fallback to original if broad empty

    # ── Step 2c: Incorrect — rewrite query + global retry ────────────────────
    log.info("CRAG incorrect — rewriting query and retrying globally.")
    rewritten      = _rewrite_query(query)
    retry_chunks   = retrieve_chunks(rewritten, top_k)   # global search, no filters
    retry_grade    = _grade_retrieval(retry_chunks)

    if retry_grade != "incorrect":
        return retry_chunks, "incorrect:rewritten"

    # Last resort: return whatever we have
    best = retry_chunks if not retry_chunks.empty else chunks
    status = "incorrect:no_evidence" if best.empty else "incorrect:rewritten"
    return best, status


# ── Groq API generation ───────────────────────────────────────────────────────

def _call_groq_api(
    messages   : list,
    temperature: float = 0.4,
    max_tokens : int   = None,
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


# ── General assistant (no retrieval) ─────────────────────────────────────────

@lru_cache(maxsize=256)
def answer_general(
    query          : str,
    history_context: str = "",
) -> Dict:
    """General medical assistant mode — no CRAG retrieval."""
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


# ── Full CRAG pipeline ────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _cached_answer(
    drug_name      : Optional[str],
    section        : Optional[str],
    top_k          : int,
    history_context: str,
) -> Dict:
    query = f"What are the drug interactions and warnings for {drug_name}?"

    # ── CRAG: retrieve + grade + correct ─────────────────────────────────────
    retrieved, crag_status = _crag_retrieve(
        query     = query,
        drug_name = drug_name,
        section   = section,
        top_k     = top_k,
    )

    log.info("CRAG status: %s | chunks: %d", crag_status, len(retrieved))

    if retrieved.empty:
        return {
            "answer"     : "No relevant FDA label evidence found for this drug.",
            "sources"    : [],
            "crag_status": crag_status,
        }

    # ── Build prompt with top-3 verified chunks ───────────────────────────────
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
        {
            "role": "user",
            "content": (
                f"{med_block}"
                f"FDA Evidence (CRAG-verified, status={crag_status}):\n{context}\n\n"
                f"Q: {query}"
            ),
        },
    ]

    answer = _call_groq_api(messages)

    return {
        "answer"     : answer,
        "crag_status": crag_status,
        "sources"    : [
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
    Full CRAG pipeline: retrieve → grade → correct → generate via Groq.

    Returns:
        {
            "answer"     : str,
            "sources"    : list[dict],
            "crag_status": str   # 'correct' | 'ambiguous:broadened' |
                                 # 'incorrect:rewritten' | 'incorrect:no_evidence'
        }
    """
    if _embedding_model is None:
        load_models()

    return _cached_answer(
        drug_name       = drug_name,
        section         = section,
        top_k           = min(top_k, MAX_TOP_K),
        history_context = history_context,
    )
