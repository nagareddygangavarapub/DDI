"""
data_preprocessing.py — Load and clean the openFDA DDI dataset.

Usage:
    from data_preprocessing import load_and_clean_data, clean_text
    df = load_and_clean_data("./data/clean_ddi_dataset.csv")

    # Clean a single text value (used by fda_sync.py for live API data):
    cleaned = clean_text("Drug Interactions: [1.1] warfarin • interacts...")
"""

import re
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── Bullet-point characters to strip ─────────────────────────────────────────
_BULLET_CODEPOINTS = [
    "\u2022", "\u2023", "\u2043", "\u2219", "\u25e6",
    "\u00b7", "\u204c", "\u204d", "\u25aa", "\u25ab",
    "\u25cf", "\u25cb",
]
_BULLET_PATTERN = re.compile("[" + "".join(_BULLET_CODEPOINTS) + "]")

_BRACKETS_RE   = re.compile(r"[\[\]]")
_SECTION_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
_WHITESPACE_RE  = re.compile(r"\s+")

# ── Text columns we care about ────────────────────────────────────────────────
TEXT_COLS = [
    "drug_interactions",
    "warnings",
    "adverse_reactions",
    "contraindications",
    "clinical_pharmacology",
    "openfda_brand_name",
    "openfda_product_type",
    "openfda_route",
    "openfda_generic_name",
]


def clean_text(text: str, col_name: str = "") -> str:
    """
    Apply the same cleaning pipeline used on the CSV dataset to a single
    text string. Used by fda_sync.py so live API data matches the format
    of the existing ChromaDB index.

    Steps:
        1. Lowercase
        2. Remove brackets [ ]
        3. Strip standalone section numbers (1, 1.1, etc.)
        4. Remove empty parentheses ()
        5. Normalise whitespace
        6. Strip column-name prefix if present at start of value
        7. Remove bullet unicode characters
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = _BRACKETS_RE.sub("", text)
    text = _SECTION_NUM_RE.sub("", text)
    text = _EMPTY_PAREN_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Strip leading column-name prefix (e.g. "drug interactions " at start)
    if col_name:
        col_clean = col_name.replace("_", " ")
        prefix_re = re.compile(rf"^\s*'?\s*{re.escape(col_clean)}\s+", re.IGNORECASE)
        text = prefix_re.sub("", text).strip()

    text = _BULLET_PATTERN.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load the clean_ddi_dataset CSV, apply all text normalization steps,
    derive final_generic_name, and return a ready-to-use DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset — shape: {df.shape}")

    obj_cols = df.select_dtypes(include="object").columns.tolist()

    # ── Apply clean_text() to every string cell ───────────────────────────────
    for col in obj_cols:
        df[col] = df[col].apply(
            lambda val: clean_text(str(val), col_name=col) if pd.notna(val) else val
        )

    # ── Derive final_generic_name via regex over all text columns ─────────────
    if "openfda_generic_name" in df.columns:
        drug_list = (
            df["openfda_generic_name"].dropna().str.lower().unique()
        )
        drug_list    = sorted(drug_list, key=len, reverse=True)
        escaped      = [re.escape(d) for d in drug_list]
        drug_pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE
        )

        search_cols   = [c for c in obj_cols if c != "openfda_generic_name"]
        combined_text = (
            df[search_cols].fillna("").agg(" ".join, axis=1).str.lower()
        )

        df["final_generic_name"] = df["openfda_generic_name"]
        detected = combined_text.str.extract(drug_pattern, expand=False)
        df["final_generic_name"] = df["final_generic_name"].fillna(detected)

        df = df.dropna(subset=["final_generic_name"]).reset_index(drop=True)
        df = df.drop(columns=["openfda_generic_name"])
    else:
        # CSV already has final_generic_name
        df = df.dropna(subset=["final_generic_name"]).reset_index(drop=True)

    # ── Fill remaining NaN helpers ────────────────────────────────────────────
    if "openfda_brand_name" in df.columns:
        df["openfda_brand_name"] = df["openfda_brand_name"].fillna(
            df["final_generic_name"]
        )
    if "warnings" in df.columns:
        df["warnings"] = df["warnings"].fillna("No warning from <FDA data>")

    print(f"Clean dataset — shape: {df.shape}")
    print(f"Remaining nulls:\n{df.isna().sum()[df.isna().sum() > 0]}")
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./data/clean_ddi_dataset.csv"
    df   = load_and_clean_data(path)
    print(df.head(2))
