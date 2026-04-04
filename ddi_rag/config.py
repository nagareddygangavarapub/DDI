"""
config.py — Central configuration for the DDI-RAG system.
All tuneable constants live here so every other module imports from one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV   = os.getenv("DDI_DATA_CSV",   "./data/clean_ddi_dataset.csv")
CHROMA_DIR = os.getenv("DDI_CHROMA_DIR", "./chroma_ddi_db")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "fda_drug_labels"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 120   # words per chunk
CHUNK_OVERLAP    = 30    # word overlap between consecutive chunks
EMBED_BATCH_SIZE = 1024  # larger batch = faster on GPU (GTX 1650 has 4GB VRAM)

# ── HuggingFace Inference API (kept for reference) ────────────────────────────
HF_API_TOKEN       = os.getenv("HF_API_TOKEN", "")
HF_MODEL           = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TIMEOUT         = 60

# ── Groq API (primary generation) ─────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GENERATION_MAX_NEW = 512
GROQ_TIMEOUT       = 30

# ── Database (PostgreSQL) ─────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── Auth (JWT) ────────────────────────────────────────────────────────────────
JWT_SECRET_KEY        = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ACCESS_TOKEN_MINS = int(os.getenv("JWT_ACCESS_TOKEN_MINS", 60 * 24))  # 1 day

# ── Redis (optional cache) ────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "")

# ── Flask / API ───────────────────────────────────────────────────────────────
PORT                 = int(os.getenv("DDI_PORT", 5000))
MAX_PRESCRIPTION_LEN = 1_000
DEFAULT_TOP_K        = 5
MAX_TOP_K            = 10

# ── RAG columns ───────────────────────────────────────────────────────────────
TEXT_COLS = [
    "drug_interactions",
    "warnings",
    "adverse_reactions",
    "contraindications",
    "clinical_pharmacology",
]

# ── Drug-name fuzzy matching ──────────────────────────────────────────────────
FUZZY_CUTOFF = 0.82
MIN_ROOT_LEN = 6

# ── OpenFDA sync ──────────────────────────────────────────────────────────────
OPENFDA_BASE_URL  = "https://api.fda.gov/drug/label.json"
OPENFDA_PAGE_SIZE = 100
SYNC_HOUR         = 2   # run nightly at 2 AM

# ── System prompt (used in rag_pipeline) ─────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a clinical pharmacist specialising in drug-drug interactions. "
    "Using ONLY the FDA label excerpts below, explain the interaction risks, "
    "contraindications, and warnings. Be concise and clinically precise. "
    "Do not invent facts!"
)
