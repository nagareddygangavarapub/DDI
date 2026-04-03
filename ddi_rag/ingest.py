import json
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
RAW_FOLDER       = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"
PROCESSED_FOLDER.mkdir(exist_ok=True)

CSV_COLUMNS = [
    "openfda_generic_name",
    "openfda_brand_name",
    "openfda_product_type",
    "openfda_route",
    "drug_interactions",
    "warnings",
    "contraindications",
    "adverse_reactions",
    "clinical_pharmacology",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _first(lst):
    """Return the first element of a list, or None."""
    return lst[0] if lst else None


# ── task 1: write per-drug .txt files (unchanged) ─────────────────────────────

def process_file(file_path):
    saved_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for drug in data.get("results", []):
        openfda      = drug.get("openfda", {})
        generic_names = openfda.get("generic_name")
        interactions  = drug.get("drug_interactions")

        if not generic_names or not interactions:
            continue

        drug_name        = generic_names[0].lower().replace(" ", "_").replace("/", "_")
        interaction_text = "\n".join(interactions)

        readable_text = (
            f"Drug: {drug_name}\n\n"
            f"=== Drug Interactions ===\n"
            f"{interaction_text}\n"
        )

        output_file = PROCESSED_FOLDER / f"{drug_name}.txt"

        if not output_file.exists():
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(readable_text)
            saved_count += 1

    print(f"{file_path.name} → Saved {saved_count} drug txt files")
    return saved_count


# ── task 2: parse one .txt file into a CSV row ───────────────────────────────

def _parse_txt(file_path: Path) -> dict | None:
    """
    Read a processed drug .txt file and return a CSV row dict.

    Expected format:
        Drug: <slug>

        === Drug Interactions ===
        <interaction text ...>
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ── drug name ─────────────────────────────────────────────────────────────
    if not lines or not lines[0].startswith("Drug: "):
        return None

    slug         = lines[0].removeprefix("Drug: ").strip()
    generic_name = slug.replace("_", " ").upper()   # reverse the slug transform

    # ── interaction text (everything after the header line) ──────────────────
    try:
        header_idx = next(
            i for i, ln in enumerate(lines)
            if ln.strip() == "=== Drug Interactions ==="
        )
        interaction_text = "\n".join(lines[header_idx + 1:]).strip()
    except StopIteration:
        interaction_text = ""

    # Wrap in a list so the CSV format matches the original: ['text...']
    drug_interactions_repr = str([interaction_text]) if interaction_text else None

    return {
        "openfda_generic_name":  generic_name,
        "openfda_brand_name":    None,   # not stored in .txt
        "openfda_product_type":  None,
        "openfda_route":         None,
        "drug_interactions":     drug_interactions_repr,
        "warnings":              None,
        "contraindications":     None,
        "adverse_reactions":     None,
        "clinical_pharmacology": None,
    }


# ── task 2: build clean_ddi_dataset.csv from processed .txt files ─────────────

def build_csv(output_path=None):
    """
    Read every .txt file in PROCESSED_FOLDER and write clean_ddi_dataset.csv.

    Each .txt file contributes one row: generic_name + drug_interactions.
    Columns not stored in .txt files (brand_name, warnings, etc.) are left blank.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "clean_ddi_dataset.csv"

    txt_files = sorted(PROCESSED_FOLDER.glob("*.txt"))
    print(f"Found {len(txt_files)} processed .txt files — building CSV …")

    rows = []
    skipped = 0
    for fp in txt_files:
        row = _parse_txt(fp)
        if row:
            rows.append(row)
        else:
            skipped += 1
            print(f"  [SKIP] Could not parse {fp.name}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in CSV_COLUMNS})

    print(f"\nCSV saved → {output_path}  ({len(rows)} rows, {skipped} skipped)")
    return output_path


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Write per-drug .txt files from raw JSON (parallel)
    files = list(RAW_FOLDER.glob("drug-label-*.json"))
    print(f"Found {len(files)} raw JSON files\n")

    total_saved = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(process_file, files)
    total_saved = sum(results)
    print(f"\nTotal drug txt files saved: {total_saved}\n")

    # 2. Build CSV from the processed .txt files
    build_csv()