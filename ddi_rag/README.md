# DDI-RAG — Drug–Drug Interaction Explanation System

**MS Data Science Final Project | University of North Texas**

A production-grade RAG (Retrieval-Augmented Generation) system that answers drug-drug interaction questions using FDA label data, ChromaDB vector search, and BioGPT-Large.

---

## Project Structure

```
ddi_rag/
├── config.py               # All tuneable constants in one place
├── data_preprocessing.py   # Load, clean, and normalise the FDA CSV
├── drug_categorization.py  # Drug → route/product-type mapping + fuzzy lookup
├── visualization.py        # Matplotlib/seaborn chart helpers (EDA notebook use)
├── rag_pipeline.py         # ChromaDB index builder, retriever, BioGPT generator
├── app.py                  # Flask REST API + single-page HTML frontend
├── run.py                  # Entry point — wires everything together
└── requirements.txt
```

---

## Quick Start (Google Colab)

```python
# 1. Install dependencies
!pip install -q -r requirements.txt
!pip install -q sacremoses    # required by BioGPT tokenizer

# 2. Mount Drive and point to your CSV
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["DDI_DATA_CSV"]   = "/content/drive/MyDrive/clean_ddi_dataset.csv"
os.environ["DDI_CHROMA_DIR"] = "/content/chroma_ddi_db"

# 3. Run
!python run.py
```

> **ChromaDB persistence tip:** After the index is built, zip and download the
> entire `chroma_ddi_db/` folder (not just the `.sqlite3` file) so you don't
> need to re-embed on the next session:
>
> ```python
> import shutil
> shutil.make_archive("chroma_backup", "zip", "/content/chroma_ddi_db")
> from google.colab import files
> files.download("chroma_backup.zip")
> ```
>
> Unzip it back to the same path on your next session — `run.py` will detect
> the existing collection and skip the rebuild automatically.

---

## Local Usage

```bash
pip install -r requirements.txt
pip install sacremoses

export DDI_DATA_CSV=./data/clean_ddi_dataset.csv
export DDI_CHROMA_DIR=./chroma_ddi_db

python run.py
# → Flask running at http://localhost:5000
```

---

## API

### `POST /api/query`

**Request**
```json
{
  "prescription": "warfarin 5mg daily, aspirin 81mg",
  "top_k": 5
}
```

**Response**
```json
{
  "detected_drugs": ["warfarin", "aspirin"],
  "results": [
    {
      "drug": "warfarin",
      "answer": "...",
      "sources": [{ "generic_name": "...", "section": "...", "score": 0.82, "text": "..." }]
    }
  ]
}
```

---

## Configuration

All settings are in `config.py`. The most useful overrides via environment variable:

| Variable          | Default                          | Description                  |
|-------------------|----------------------------------|------------------------------|
| `DDI_DATA_CSV`    | `./data/clean_ddi_dataset.csv`   | Path to cleaned FDA CSV      |
| `DDI_CHROMA_DIR`  | `./chroma_ddi_db`                | ChromaDB persistence folder  |
| `DDI_PORT`        | `5000`                           | Flask port                   |

---

## Module Reference

| Module                  | Key exports                                                  |
|-------------------------|--------------------------------------------------------------|
| `data_preprocessing`    | `load_and_clean_data(csv_path)`                              |
| `drug_categorization`   | `lookup_route()`, `categorize_drug()`, `apply_route_column()`, `apply_product_type()` |
| `visualization`         | `setup_plot_style()`, `clean_text()`, `get_ngrams_sklearn()`, `styled_barh()` |
| `rag_pipeline`          | `load_models()`, `build_chunk_df()`, `build_chroma_index()`, `retrieve_chunks()`, `answer_ddi()` |
| `app`                   | `flask_app`, `init_lookups()`, `parse_prescription()`         |

---

> **Disclaimer:** For educational use only. Not intended as medical advice.
