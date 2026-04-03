"""
data_preprocessing.py — Load and clean the openFDA DDI dataset.

Usage:
    from data_preprocessing import load_and_clean_data
    df = load_and_clean_data("./data/clean_ddi_dataset.csv")
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
_BULLET_PATTERN = "[" + "".join(_BULLET_CODEPOINTS) + "]"

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


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load the clean_ddi_dataset CSV, apply all text normalization steps,
    derive final_generic_name, and return a ready-to-use DataFrame.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset — shape: {df.shape}")

    # ── Lowercase all string columns ──────────────────────────────────────────
    df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

    # ── Select available text columns ─────────────────────────────────────────
    text_cols = [c for c in TEXT_COLS if c in df.columns]
    obj_cols  = df.select_dtypes(include="object").columns.tolist()

    # ── Remove brackets, standalone section numbers, empty parentheses ────────
    df[obj_cols] = df[obj_cols].replace(r"[\[\]]", "", regex=True)
    df[obj_cols] = df[obj_cols].replace(r"\b\d+(\.\d+)?\b", "", regex=True)
    df[obj_cols] = df[obj_cols].replace(r"\(\s*\)", "", regex=True)
    df[obj_cols] = (
        df[obj_cols]
        .replace(r"\s+", " ", regex=True)
        .apply(lambda col: col.str.strip())
    )

    # ── Strip column-name prefixes from cell values ───────────────────────────
    for col in obj_cols:
        col_clean = col.replace("_", " ")
        pattern   = rf"^\s*'?\s*{col_clean}\s+"
        df[col]   = (
            df[col]
            .str.replace(pattern, "", regex=True, case=False)
            .str.strip()
        )

    # ── Remove bullet characters ──────────────────────────────────────────────
    df[obj_cols] = df[obj_cols].replace(_BULLET_PATTERN, " ", regex=True)
    df[obj_cols] = (
        df[obj_cols]
        .replace(r"\s+", " ", regex=True)
        .apply(lambda col: col.str.strip())
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
