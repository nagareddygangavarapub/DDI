"""
run.py — Entry point for the DDI-RAG system.

Steps:
  1. Load & clean the dataset
  2. Apply route and product-type columns
  3. Build the chunk DataFrame
  4. Build (or load) the ChromaDB index
  5. Load embedding model
  6. Initialise DB tables
  7. Initialise prescription-parser lookup tables
  8. Start nightly FDA sync scheduler
  9. Start Flask server in a daemon thread
  10. Start Cloudflare tunnel and print the public URL

Usage (local):
    python run.py
"""

import logging
import os
import re
import subprocess
import sys
import time
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ddi.run")

# ── Kill stale process on port 5000 (Linux/Mac only) ──────────────────────────
try:
    subprocess.run(["fuser", "-k", "5000/tcp"], capture_output=True)
    time.sleep(1)
    log.info("Port 5000 cleared.")
except FileNotFoundError:
    pass  # fuser not available on Windows — that's fine

# ── 1. Load and clean data ────────────────────────────────────────────────────
from config import CHROMA_DIR, COLLECTION_NAME, DATA_CSV, PORT
from data_preprocessing import load_and_clean_data

log.info("Loading dataset from: %s", DATA_CSV)
df = load_and_clean_data(DATA_CSV)

# ── 2. Apply route + product-type columns ─────────────────────────────────────
from drug_categorization import apply_product_type, apply_route_column

df = apply_route_column(df)
df = apply_product_type(df)

# ── 3. Build chunk DataFrame ──────────────────────────────────────────────────
from rag_pipeline import build_chunk_df, build_chroma_index, load_models

log.info("Building chunk DataFrame ...")
chunk_df = build_chunk_df(df)
log.info("chunk_df shape: %s", chunk_df.shape)

# ── 4. Build or load ChromaDB index ──────────────────────────────────────────
import chromadb

_client   = chromadb.PersistentClient(path=CHROMA_DIR)
_existing = [c.name for c in _client.list_collections()]

if COLLECTION_NAME in _existing:
    count = _client.get_collection(COLLECTION_NAME).count()
    log.info(
        'ChromaDB collection "%s" already exists with %d vectors — skipping rebuild.',
        COLLECTION_NAME, count,
    )
else:
    log.info("ChromaDB collection not found — building index (this takes a while) ...")
    load_models()
    build_chroma_index(chunk_df, rebuild=False)

# ── 5. Load embedding model (no-op if already loaded) ────────────────────────
load_models()

# ── 6. Initialise database tables ────────────────────────────────────────────
from database import init_db, ping_db

if ping_db():
    init_db()
    log.info("PostgreSQL connected and tables ready.")
else:
    log.warning(
        "Could not connect to PostgreSQL. "
        "Check DATABASE_URL in your .env file. "
        "Auth and history features will be unavailable."
    )

# ── 7. Init prescription-parser lookup tables ─────────────────────────────────
from app import flask_app, init_lookups

init_lookups(chunk_df)

# ── 8. Start nightly FDA sync scheduler ──────────────────────────────────────
from fda_sync import start_scheduler

_scheduler = start_scheduler()

# ── 9. Start Flask ────────────────────────────────────────────────────────────
def _run_flask():
    flask_app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False)

Thread(target=_run_flask, daemon=True).start()
time.sleep(2)
log.info("Flask listening on port %d.", PORT)

# ── 10. Cloudflare tunnel (Linux/Mac only — skipped on Windows) ───────────────
import platform

def _run_cloudflared():
    CF_BIN = "/usr/local/bin/cloudflared"

    if not os.path.exists(CF_BIN):
        log.info("Downloading cloudflared ...")
        subprocess.run(
            [
                "wget", "-q", "-O", CF_BIN,
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            ],
            check=True,
        )
        subprocess.run(["chmod", "+x", CF_BIN], check=True)

    proc = subprocess.Popen(
        [CF_BIN, "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url_pat  = re.compile(r"https://[\w-]+\.trycloudflare\.com")
    pub_url  = None
    deadline = time.time() + 30

    while time.time() < deadline:
        line  = proc.stdout.readline()
        match = url_pat.search(line)
        if match:
            pub_url = match.group(0)
            break

    if pub_url:
        print("\n" + "=" * 62)
        print(f"  DDI Web App LIVE  :  {pub_url}")
        print(f"  API endpoint      :  {pub_url}/api/query")
        print("=" * 62 + "\n")
    else:
        log.warning("Cloudflare tunnel did not start — app available locally.")
        print(f"\nFlask running at http://localhost:{PORT}\n")

    return proc


if platform.system() == "Windows":
    # Cloudflared not supported on Windows dev — use localhost directly
    print("\n" + "=" * 62)
    print(f"  DDI Web App LIVE  :  http://localhost:{PORT}")
    print(f"  API endpoint      :  http://localhost:{PORT}/api/query")
    print("=" * 62 + "\n")
    proc = None
else:
    try:
        proc = _run_cloudflared()
    except Exception as exc:
        log.warning("Cloudflare tunnel failed: %s — running locally.", exc)
        print(f"\nFlask running at http://localhost:{PORT}\n")
        proc = None

# Keep main thread alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    log.info("Shutting down.")
    if proc:
        proc.terminate()
    if _scheduler:
        _scheduler.shutdown(wait=False)
    sys.exit(0)
