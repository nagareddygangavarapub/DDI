"""
ablation_study.py — DrugSafe AI Component Ablation Study
=========================================================
Measures Hit@5 and MRR across four system configurations to isolate the
contribution of each component to overall retrieval quality.

Configurations tested
---------------------
C1 — Vanilla RAG          : raw query, no drug filter, no CRAG
C2 — + Drug Normalisation : brand→generic normalisation + drug filter, no CRAG
C3 — + CRAG (no rewrite)  : C2 + grading + broadening, but NO query rewriting
C4 — Full CRAG            : C3 + LLM-powered query rewriting on incorrect grade

Hit@5 definition
----------------
A query is a "hit" if at least one of the top-5 returned chunks has a
cosine similarity score ≥ RELEVANCE_LOW (0.40).  This threshold is the
system's own boundary for "minimally relevant" evidence.

MRR definition
--------------
Reciprocal of the rank of the first chunk with score ≥ RELEVANCE_LOW.

Run
---
    cd ddi_rag
    python ablation_study.py

Outputs
-------
    ../outputs/ablation_detail.csv    — per-query scores all 4 configs
    ../outputs/ablation_summary.csv   — Hit@5 / MRR per query type + overall
    Prints a formatted summary table to stdout.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("ablation")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_OUTPUTS = _HERE.parent / "outputs"
_OUTPUTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_HERE))

# ── Pipeline imports ───────────────────────────────────────────────────────────
print("Loading embedding model …", flush=True)
from rag_pipeline import (
    RELEVANCE_HIGH,
    RELEVANCE_LOW,
    _grade_retrieval,
    _get_qdrant,
    _rewrite_query,
    load_models,
    retrieve_chunks,
)
from config import COLLECTION_NAME

load_models()
print("Embedding model ready.", flush=True)

# ── Ensure payload indexes exist (required for drug_name / section filters) ───
print("Ensuring payload indexes on Qdrant collection …", flush=True)
_qc = _get_qdrant()
for _field in ("generic_name", "section"):
    try:
        _qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=_field,
            field_schema="keyword",
        )
        print(f"  index '{_field}': ready", flush=True)
    except Exception as _e:
        # Already exists or benign conflict — safe to continue
        print(f"  index '{_field}': {_e}", flush=True)
print()

# ── Drug-name normalisation ────────────────────────────────────────────────────
_JSON = _HERE / "drug_names.json"
with open(_JSON, encoding="utf-8") as _f:
    _lookup = json.load(_f)

B2G: Dict[str, str] = _lookup.get("brand_to_generic", {})
_SORTED   = sorted(B2G, key=len, reverse=True)
_PATTERNS = {
    n: re.compile(r"\b" + re.escape(n) + r"\b", re.IGNORECASE)
    for n in _SORTED
}


def normalise(text: str) -> str:
    """Replace brand names with INN generics (longest-match, case-insensitive)."""
    out = text
    for name in _SORTED:
        out = _PATTERNS[name].sub(B2G[name], out)
    return out


# ── Metric helpers ─────────────────────────────────────────────────────────────
HIT_THRESH = RELEVANCE_LOW   # 0.40


def hit_at_5(chunks: pd.DataFrame) -> int:
    """1 if any chunk scores ≥ HIT_THRESH, else 0."""
    if chunks.empty:
        return 0
    return int(float(chunks["score"].max()) >= HIT_THRESH)


def reciprocal_rank(chunks: pd.DataFrame) -> float:
    """1 / rank of first chunk with score ≥ HIT_THRESH.  0 if none."""
    if chunks.empty:
        return 0.0
    for rank, (_, row) in enumerate(chunks.iterrows(), 1):
        if float(row["score"]) >= HIT_THRESH:
            return 1.0 / rank
    return 0.0


def peak_score(chunks: pd.DataFrame) -> float:
    if chunks.empty:
        return 0.0
    return round(float(chunks["score"].max()), 4)


# ── Config runners ─────────────────────────────────────────────────────────────

def c1_vanilla_rag(query: str, _target: Optional[str], top_k: int = 5) -> pd.DataFrame:
    """
    C1 — Vanilla RAG.
    Raw query, global search (no drug filter), no CRAG.
    Simulates the baseline where no system intelligence is applied.
    """
    return retrieve_chunks(query, top_k=top_k)


def c2_normalisation(query: str, target: Optional[str], top_k: int = 5) -> pd.DataFrame:
    """
    C2 — + Drug Normalisation.
    Brand→generic normalisation applied, drug filter added, no CRAG.
    Isolates the contribution of name normalisation alone.
    """
    norm_q = normalise(query)
    return retrieve_chunks(norm_q, top_k=top_k, drug_name=target)


def c3_crag_no_rewrite(query: str, target: Optional[str], top_k: int = 5) -> pd.DataFrame:
    """
    C3 — + CRAG (no rewrite).
    Normalisation + CRAG grading + broadened search for ambiguous queries.
    Incorrect queries are NOT rewritten — we return original chunks.
    Isolates the contribution of CRAG grading/broadening without rewriting.
    """
    norm_q = normalise(query)
    chunks = retrieve_chunks(norm_q, top_k=top_k, drug_name=target)
    grade  = _grade_retrieval(chunks)

    if grade == "correct":
        return chunks

    if grade == "ambiguous":
        broad = retrieve_chunks(norm_q, top_k=top_k * 2)   # no filter
        if not broad.empty:
            return (
                pd.concat([chunks, broad])
                .drop_duplicates(subset=["text"])
                .sort_values("score", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )
        return chunks

    # incorrect — no rewrite, return whatever we retrieved
    return chunks


def c4_full_crag(query: str, target: Optional[str], top_k: int = 5) -> pd.DataFrame:
    """
    C4 — Full CRAG.
    Normalisation + grading + broadening + LLM query rewriting on incorrect.
    The complete production pipeline.
    """
    norm_q = normalise(query)
    chunks = retrieve_chunks(norm_q, top_k=top_k, drug_name=target)
    grade  = _grade_retrieval(chunks)

    if grade == "correct":
        return chunks

    if grade == "ambiguous":
        broad = retrieve_chunks(norm_q, top_k=top_k * 2)
        if not broad.empty:
            return (
                pd.concat([chunks, broad])
                .drop_duplicates(subset=["text"])
                .sort_values("score", ascending=False)
                .head(top_k)
                .reset_index(drop=True)
            )
        return chunks

    # incorrect — rewrite and retry
    rewritten     = _rewrite_query(norm_q)
    retry_chunks  = retrieve_chunks(rewritten, top_k=top_k)
    retry_grade   = _grade_retrieval(retry_chunks)

    if retry_grade != "incorrect":
        return retry_chunks

    # last resort
    return retry_chunks if not retry_chunks.empty else chunks


# ── Test query bank ────────────────────────────────────────────────────────────
# Each entry: (query_text, target_generic_drug, query_type)
# query_type ∈ {"brand_ddi", "generic_ddi", "colloquial", "mechanism"}

QUERIES: List[Tuple[str, Optional[str], str]] = [

    # ── Brand-name DDI queries (normalisation has highest impact here) ─────────
    ("Advil interaction with blood thinners",
     "ibuprofen",         "brand_ddi"),
    ("Tylenol overdose liver damage risk",
     "acetaminophen",     "brand_ddi"),
    ("Lipitor muscle pain rhabdomyolysis",
     "atorvastatin",      "brand_ddi"),
    ("Glucophage kidney problems lactic acidosis",
     "metformin",         "brand_ddi"),
    ("Zithromax heart rhythm QT prolongation",
     "azithromycin",      "brand_ddi"),
    ("Motrin and aspirin concurrent use",
     "ibuprofen",         "brand_ddi"),
    ("Zoloft and MAOIs serotonin syndrome",
     "sertraline",        "brand_ddi"),
    ("Prilosec drug interactions CYP2C19",
     "omeprazole",        "brand_ddi"),
    ("Norvasc amlodipine grapefruit interaction",
     "amlodipine",        "brand_ddi"),
    ("Xanax and alcohol CNS depression",
     "alprazolam",        "brand_ddi"),

    # ── Generic-name DDI queries (CRAG grading has highest impact here) ────────
    ("ibuprofen and warfarin bleeding risk increase",
     "ibuprofen",         "generic_ddi"),
    ("metformin lactic acidosis contraindications",
     "metformin",         "generic_ddi"),
    ("amoxicillin penicillin cross-reactivity allergy",
     "amoxicillin",       "generic_ddi"),
    ("lisinopril ACE inhibitor hyperkalemia risk",
     "lisinopril",        "generic_ddi"),
    ("sertraline SSRI drug interactions MAO inhibitor",
     "sertraline",        "generic_ddi"),
    ("atorvastatin CYP3A4 inhibitor statin interactions",
     "atorvastatin",      "generic_ddi"),
    ("warfarin INR monitoring drug food interactions",
     "warfarin",          "generic_ddi"),
    ("azithromycin QT prolongation cardiac arrhythmia",
     "azithromycin",      "generic_ddi"),
    ("omeprazole proton pump inhibitor clopidogrel interaction",
     "omeprazole",        "generic_ddi"),
    ("alprazolam benzodiazepine CNS depressant opioid combination",
     "alprazolam",        "generic_ddi"),

    # ── Colloquial queries (query rewriting has highest impact here) ───────────
    ("can I take my blood thinner with over the counter pain medicine",
     "warfarin",          "colloquial"),
    ("my cholesterol pill is causing muscle aches what should I do",
     "atorvastatin",      "colloquial"),
    ("is it okay to drink alcohol while on antibiotics",
     "amoxicillin",       "colloquial"),
    ("my stomach acid medication and my heart pill dont mix",
     "omeprazole",        "colloquial"),
    ("blood pressure medicine making me dizzy in the morning",
     "lisinopril",        "colloquial"),

    # ── Mechanism queries (tests depth of FDA coverage) ───────────────────────
    ("CYP3A4 enzyme inhibition drug metabolism interactions",
     None,                "mechanism"),
    ("QT interval prolongation cardiac drugs risk",
     None,                "mechanism"),
    ("serotonin syndrome SSRI SNRI risk factors treatment",
     None,                "mechanism"),
    ("ACE inhibitor angiotensin converting enzyme renal effects",
     None,                "mechanism"),
    ("benzodiazepine CNS depression respiratory side effects",
     None,                "mechanism"),
]

CONFIGS = [
    ("C1 — Vanilla RAG",          c1_vanilla_rag),
    ("C2 — +Normalisation",       c2_normalisation),
    ("C3 — +CRAG (no rewrite)",   c3_crag_no_rewrite),
    ("C4 — Full CRAG",            c4_full_crag),
]

# ── Run ablation ───────────────────────────────────────────────────────────────

def run_ablation() -> pd.DataFrame:
    rows = []
    total = len(QUERIES)

    for qi, (query, target, qtype) in enumerate(QUERIES, 1):
        print(f"  [{qi:02d}/{total}] {qtype:12s} | {query[:55]:<55}", flush=True)
        row: Dict = {"query": query, "target": target or "—", "type": qtype}

        for cfg_name, cfg_fn in CONFIGS:
            t0     = time.perf_counter()
            chunks = cfg_fn(query, target)
            elapsed = round(time.perf_counter() - t0, 2)

            h      = hit_at_5(chunks)
            rr     = round(reciprocal_rank(chunks), 3)
            ps     = peak_score(chunks)
            n_hits = int(chunks["score"].ge(HIT_THRESH).sum()) if not chunks.empty else 0

            # short config key for column names
            key = cfg_name.split("—")[0].strip()
            row[f"{key}_hit"]       = h
            row[f"{key}_rr"]        = rr
            row[f"{key}_peak"]      = ps
            row[f"{key}_n_hits"]    = n_hits
            row[f"{key}_latency_s"] = elapsed

            mark = "✓" if h else "✗"
            print(f"         {cfg_name}: {mark}  peak={ps:.3f}  RR={rr:.3f}  ({elapsed:.2f}s)",
                  flush=True)

        rows.append(row)
        print()   # blank line between queries

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Hit@5 and MRR by query type and overall."""
    cfg_keys = [c.split("—")[0].strip() for c, _ in CONFIGS]

    records = []
    for qtype in list(df["type"].unique()) + ["Overall"]:
        subset = df if qtype == "Overall" else df[df["type"] == qtype]
        n = len(subset)
        rec: Dict = {"Query Type": qtype, "N": n}
        for key in cfg_keys:
            rec[f"{key} Hit@5 (%)"] = round(subset[f"{key}_hit"].mean() * 100, 1)
            rec[f"{key} MRR"]       = round(subset[f"{key}_rr"].mean(), 3)
        records.append(rec)

    return pd.DataFrame(records)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  DrugSafe AI — Component Ablation Study")
    print(f"  Queries: {len(QUERIES)}  |  Configs: {len(CONFIGS)}")
    print(f"  Hit threshold: {HIT_THRESH}  (= RELEVANCE_LOW)")
    print("=" * 70, "\n")

    detail_df = run_ablation()

    # Save detail
    detail_path = _OUTPUTS / "ablation_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Detailed results → {detail_path}\n")

    # Build and print summary
    summary_df = summarise(detail_df)
    summary_path = _OUTPUTS / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cfg_keys = [c.split("—")[0].strip() for c, _ in CONFIGS]

    print("=" * 70)
    print("  ABLATION SUMMARY — Hit@5 (%)")
    print("=" * 70)
    print(f"{'Query Type':<20} {'N':>4} ", end="")
    for k in cfg_keys:
        print(f"  {k:>22}", end="")
    print()
    print("-" * 70)

    for _, row in summary_df.iterrows():
        print(f"{row['Query Type']:<20} {int(row['N']):>4} ", end="")
        for k in cfg_keys:
            val = row[f"{k} Hit@5 (%)"]
            print(f"  {val:>21.1f}%", end="")
        print()

    print("=" * 70)

    # Delta column (C4 vs C1)
    overall = summary_df[summary_df["Query Type"] == "Overall"].iloc[0]
    c1_hit = overall["C1 Hit@5 (%)"]
    c4_hit = overall["C4 Hit@5 (%)"]
    print(f"\n  ▲ Full CRAG over Vanilla RAG: +{c4_hit - c1_hit:.1f} pp (overall Hit@5)")

    # Incremental deltas
    prev_val = c1_hit
    for i, k in enumerate(cfg_keys[1:], 2):
        cur_val = overall[f"{k} Hit@5 (%)"]
        delta = cur_val - prev_val
        print(f"  ▲ C{i} over C{i-1}: {delta:+.1f} pp")
        prev_val = cur_val

    print(f"\nSummary table → {summary_path}")
    print("\nDone.")
