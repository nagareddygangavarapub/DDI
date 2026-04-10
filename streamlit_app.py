"""
streamlit_app.py — DrugSafe AI · Streamlit UI

Access model:
    • Everyone can use the app freely (no login required)
    • Logged-in users get persistent query history + medications saved to DB
    • Guests get the full RAG / AI assistant — history is not saved

Usage:
    cd C:\\Users\\C V REDDY\\Downloads\\ddi_rag
    streamlit run streamlit_app.py
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "ddi_rag"
sys.path.insert(0, str(_SRC))

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="DrugSafe AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# FONT + CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&'
    'family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&'
    'family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ═══════════════════════════════════════════
   CSS VARIABLES
═══════════════════════════════════════════ */
:root {
  --bg:            #F7F5F0;
  --bg-2:          #EFECE5;
  --bg-3:          #E6E2D9;
  --surface:       #FFFFFF;
  --border:        #E0DCD3;
  --border-strong: #C8C3B8;
  --text-1:        #1A1714;
  --text-2:        #6B6560;
  --text-3:        #A09A94;
  --teal:          #0B7B6E;
  --teal-mid:      #0FA898;
  --teal-light:    #E3F7F4;
  --teal-border:   rgba(11,123,110,0.22);
  --amber:         #B45309;
  --amber-light:   #FEF9EC;
  --amber-border:  #F5D07A;
  --r-xs:          6px;
  --r-sm:          10px;
  --r:             16px;
  --r-lg:          22px;
  --sh-sm:         0 1px 4px rgba(26,23,20,0.07);
  --sh:            0 3px 14px rgba(26,23,20,0.08);
  --sh-lg:         0 10px 40px rgba(26,23,20,0.11);
}

/* ═══════════════════════════════════════════
   GLOBAL RESET
═══════════════════════════════════════════ */
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text-1) !important;
}

/* Dot-grid texture on the outer canvas */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(circle, #C8C3B8 1px, transparent 1px) !important;
  background-size: 26px 26px !important;
  background-color: var(--bg) !important;
}

/* Main content column — white card floating over the grid */
.main .block-container {
  background: var(--surface) !important;
  max-width: 880px !important;
  padding: 0 2.25rem 5rem !important;
  margin: 0 auto !important;
  box-shadow: var(--sh-lg) !important;
  border-left:  1px solid var(--border) !important;
  border-right: 1px solid var(--border) !important;
  min-height: 100vh !important;
}

/* ═══════════════════════════════════════════
   HIDE CHROME JUNK
═══════════════════════════════════════════ */
#MainMenu, footer, [data-testid="stDecoration"] {
  visibility: hidden !important;
}
[data-testid="stHeader"] {
  background: transparent !important;
  backdrop-filter: none !important;
}

/* ═══════════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Playfair Display', Georgia, serif !important;
  color: var(--text-1) !important;
  letter-spacing: -0.01em !important;
}
p, li, span, label, caption {
  font-family: 'DM Sans', system-ui, sans-serif !important;
}

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg-2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.75rem !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-1) !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown small {
  color: var(--text-3) !important;
  font-size: 0.72rem !important;
}

/* Sidebar toggle */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
  background-color: var(--teal) !important;
  color: #fff !important;
  border-radius: 50% !important;
  border: none !important;
  box-shadow: var(--sh) !important;
}

/* ═══════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════ */
div[data-testid="stButton"] > button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: 99px !important;
  padding: 0.32rem 0.95rem !important;
  transition: border-color 0.14s, color 0.14s, background 0.14s !important;
  box-shadow: var(--sh-sm) !important;
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--teal) !important;
  color: var(--teal) !important;
  background: var(--teal-light) !important;
}
div[data-testid="stButton"] > button:active {
  transform: scale(0.98) !important;
}

/* Form submit — filled teal */
div[data-testid="stFormSubmitButton"] > button {
  background: var(--teal) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 8px rgba(11,123,110,0.28) !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
  background: var(--teal-mid) !important;
  color: #fff !important;
}

/* ═══════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--surface) !important;
  color: var(--text-1) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--r-xs) !important;
  font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(11,123,110,0.12) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  color: var(--text-2) !important;
}

/* ═══════════════════════════════════════════
   CHAT
═══════════════════════════════════════════ */
[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg-2) !important;
  color: var(--text-1) !important;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--r) !important;
  font-size: 0.93rem !important;
  line-height: 1.6 !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(11,123,110,0.10) !important;
  outline: none !important;
}
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 0.2rem 0 !important;
  border: none !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text-1) !important;
  font-size: 0.93rem !important;
  line-height: 1.72 !important;
}

/* ═══════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════ */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  background: var(--bg-2) !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  font-size: 0.83rem !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
}

/* ═══════════════════════════════════════════
   ALERTS
═══════════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: var(--r-sm) !important;
  font-size: 0.86rem !important;
}

/* ═══════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════ */
.stProgress > div > div > div {
  background: var(--teal) !important;
  border-radius: 99px !important;
}
.stProgress > div > div {
  background: var(--bg-3) !important;
  border-radius: 99px !important;
}

/* ═══════════════════════════════════════════
   TABS
═══════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--teal) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
  background: var(--teal) !important;
}

/* ═══════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════ */
hr {
  border-color: var(--border) !important;
  margin: 1.25rem 0 !important;
}

/* ═══════════════════════════════════════════
   CAPTION / SMALL TEXT
═══════════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"] p {
  font-size: 0.76rem !important;
  color: var(--text-3) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ═══════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════ */

/* ── Welcome hero ── */
.welcome-hero {
  padding: 3.5rem 1rem 2.25rem;
  text-align: center;
  position: relative;
}
.welcome-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal);
  background: var(--teal-light);
  border: 1px solid var(--teal-border);
  border-radius: 99px;
  padding: 5px 16px;
  margin-bottom: 1.4rem;
  font-family: 'JetBrains Mono', monospace;
}
.welcome-title {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 3.4rem !important;
  font-weight: 700 !important;
  color: var(--text-1) !important;
  line-height: 1.1 !important;
  letter-spacing: -0.025em !important;
  margin: 0 0 0.6rem !important;
}
.welcome-title em {
  font-style: italic !important;
  color: var(--teal) !important;
}
.welcome-sub {
  font-size: 1.05rem;
  color: var(--text-2);
  line-height: 1.65;
  max-width: 460px;
  margin: 0 auto 2.25rem;
  font-weight: 300;
}
.welcome-stats {
  display: inline-flex;
  gap: 0;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  margin-bottom: 0.5rem;
}
.stat-cell {
  padding: 1rem 1.75rem;
  border-right: 1px solid var(--border);
  text-align: center;
}
.stat-cell:last-child { border-right: none; }
.stat-num {
  display: block;
  font-family: 'Playfair Display', serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--text-1);
  line-height: 1;
  margin-bottom: 4px;
}
.stat-label {
  display: block;
  font-size: 0.63rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}

/* ── Drug / AI badges ── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.66rem;
  font-weight: 500;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 4px 11px;
  border-radius: var(--r-xs);
  margin-bottom: 0.7rem;
}
.badge-rag {
  background: var(--teal-light);
  color: var(--teal);
  border: 1px solid var(--teal-border);
}
.badge-ai {
  background: var(--amber-light);
  color: var(--amber);
  border: 1px solid rgba(180,83,9,0.2);
}

/* ── CRAG status micro-pill ── */
.crag-pill {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.60rem;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--bg-3);
  color: var(--text-3);
  border: 1px solid var(--border-strong);
  margin-left: 7px;
  vertical-align: middle;
  letter-spacing: 0.04em;
}
.crag-correct   { background: #E3F7F4; color: #0B7B6E; border-color: rgba(11,123,110,0.2); }
.crag-ambiguous { background: #FEF9EC; color: #B45309; border-color: rgba(180,83,9,0.2);   }
.crag-incorrect { background: #FEF2F2; color: #B91C1C; border-color: rgba(185,28,28,0.2);  }

/* ── User chip (sidebar) ── */
.user-chip {
  background: var(--teal-light);
  border: 1px solid var(--teal-border);
  border-radius: var(--r-sm);
  padding: 0.65rem 0.9rem;
  margin-bottom: 0.8rem;
  font-size: 0.84rem;
  line-height: 1.55;
}
.user-chip strong { color: var(--teal) !important; font-weight: 600; }
.user-chip small  { color: var(--text-3) !important; font-size: 0.75rem; }

/* ── Guest notice ── */
.guest-notice {
  background: var(--amber-light);
  border: 1px solid var(--amber-border);
  border-radius: var(--r-sm);
  padding: 0.6rem 0.9rem;
  margin-bottom: 0.8rem;
  font-size: 0.79rem;
  color: var(--amber);
  line-height: 1.55;
}

/* ── Pharmacy card ── */
.pharm-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1rem 1.15rem;
  margin: 0.4rem 0;
  font-size: 0.85rem;
  line-height: 1.72;
  box-shadow: var(--sh-sm);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.pharm-card:hover {
  border-color: var(--teal) !important;
  box-shadow: 0 4px 18px rgba(11,123,110,0.10) !important;
}
.pharm-card strong {
  font-size: 0.9rem;
  color: var(--text-1);
  font-weight: 600;
}
.pharm-card a {
  color: var(--teal) !important;
  text-decoration: none !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
}
.pharm-card a:hover { text-decoration: underline !important; }

/* ── Sidebar section label ── */
.sidebar-label {
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: 0.5rem;
  display: block;
}

/* ── Suggestion chips row ── */
.chips-label {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
  margin: 1.5rem 0 0.6rem;
  display: block;
}

/* ── Source entry in expander ── */
.source-entry {
  padding: 0.55rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.82rem;
  line-height: 1.6;
}
.source-entry:last-child { border-bottom: none; }
.source-entry .src-drug {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--teal);
}
.source-entry .src-section {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  color: var(--text-3);
  margin-left: 6px;
}
.source-entry .src-score {
  float: right;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  color: var(--text-3);
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 6px;
}
.source-entry .src-text {
  color: var(--text-2);
  font-size: 0.80rem;
  margin-top: 4px;
  font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INIT
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _init_db():
    try:
        from database import init_db
        init_db()
        return True
    except Exception:
        return False

_db_ok = _init_db()


# ══════════════════════════════════════════════════════════════════════════════
# RAG / MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

_DRUG_NAMES_JSON = _SRC / "drug_names.json"


@st.cache_resource(show_spinner="Initialising DrugSafe AI…")
def _load_system():
    import json
    from rag_pipeline import load_models

    if not _DRUG_NAMES_JSON.exists():
        st.error("❌ drug_names.json not found. Run ddi_rag/upload_to_qdrant.py locally first.")
        st.stop()

    with open(_DRUG_NAMES_JSON, encoding="utf-8") as f:
        lookup = json.load(f)

    b2g      = lookup.get("brand_to_generic", {})
    generics = set(lookup.get("generic_names", []))

    all_names    = generics | set(b2g.keys())
    sorted_names = sorted(all_names, key=len, reverse=True)
    patterns     = {n: re.compile(r"\b" + re.escape(n) + r"\b") for n in sorted_names}

    load_models()
    return b2g, sorted_names, patterns


_B2G, _SORTED, _PATTERNS = _load_system()

from rag_pipeline import answer_ddi, answer_general
from pharmacy_search import find_pharmacies
from streamlit_js_eval import get_geolocation


def parse_prescription(text: str) -> List[str]:
    text_lower = text.lower()
    found: List[str] = []
    consumed: List[Tuple[int, int]] = []
    for name in _SORTED:
        for m in _PATTERNS[name].finditer(text_lower):
            s, e = m.start(), m.end()
            if not any(cs <= s < ce or cs < e <= ce for cs, ce in consumed):
                generic = _B2G.get(name, name)
                if generic not in found:
                    found.append(generic)
                consumed.append((s, e))
    return found


# ══════════════════════════════════════════════════════════════════════════════
# AUTH + DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

from auth import hash_password, verify_password
from database import get_db
from models import Medication, QueryHistory, User


def _register(email, full_name, password):
    with get_db() as db:
        if db.query(User).filter(User.email == email.lower()).first():
            return None, "Email already registered."
        user = User(email=email.lower().strip(),
                    full_name=full_name.strip(),
                    password=hash_password(password))
        db.add(user)
        db.flush()
        return user.to_dict(), None


def _login(email, password):
    with get_db() as db:
        user = db.query(User).filter(User.email == email.lower()).first()
        if not user or not verify_password(password, user.password):
            return None, "Invalid email or password."
        return user.to_dict(), None


def _get_medications(user_id) -> List[Dict]:
    with get_db() as db:
        meds = (db.query(Medication)
                .filter(Medication.user_id == user_id, Medication.is_active == True)
                .order_by(Medication.created_at).all())
        return [m.to_dict() for m in meds]


def _add_medication(user_id, drug_name, dosage="", frequency=""):
    with get_db() as db:
        db.add(Medication(user_id=user_id, drug_name=drug_name.strip(),
                          dosage=dosage.strip() or None,
                          frequency=frequency.strip() or None, is_active=True))


def _delete_medication(med_id):
    with get_db() as db:
        med = db.query(Medication).filter(Medication.id == med_id).first()
        if med:
            med.is_active = False


def _get_history(user_id, limit=20) -> List[Dict]:
    with get_db() as db:
        rows = (db.query(QueryHistory)
                .filter(QueryHistory.user_id == user_id)
                .order_by(QueryHistory.created_at.desc())
                .limit(limit).all())
        return [r.to_dict() for r in rows]


def _save_history(user_id, prescription, detected, results):
    with get_db() as db:
        db.add(QueryHistory(user_id=user_id, prescription=prescription,
                            detected_drugs=detected, results=results))


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k, v in {
    "user_id": None, "user_name": None, "user_email": None,
    "messages": [], "show_auth": False, "_pending_prompt": None,
    "user_lat": None, "user_lon": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

logged_in = bool(st.session_state.user_id)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Playfair Display\',serif;font-size:1.25rem;'
        'font-weight:700;color:#1A1714;letter-spacing:-0.01em;margin-bottom:2px;">'
        'DrugSafe <em style="color:#0B7B6E;font-style:italic;">AI</em></div>'
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
        'color:#A09A94;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:0.5rem;">'
        'FDA · CRAG · Qdrant · LLaMA</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Auth section ──────────────────────────────────────────────────────────
    if logged_in:
        st.markdown(
            f'<div class="user-chip">'
            f'<strong>{st.session_state.user_name}</strong>'
            f'<br><small>{st.session_state.user_email}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        if col1.button("＋ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if col2.button("Sign Out", use_container_width=True):
            st.session_state.update({"user_id": None, "user_name": None,
                                     "user_email": None, "messages": []})
            st.rerun()

    else:
        st.markdown(
            '<div class="guest-notice">'
            '🔓 Browsing as <strong>guest</strong>. '
            'Sign in to save history &amp; medications.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("🔑 Login / Register", expanded=st.session_state.show_auth):
            tab_l, tab_r = st.tabs(["Login", "Register"])

            with tab_l:
                with st.form("login_form"):
                    email_l = st.text_input("Email", key="l_email")
                    pass_l  = st.text_input("Password", type="password", key="l_pass")
                    ok_l    = st.form_submit_button("Login", use_container_width=True)
                if ok_l:
                    user, err = _login(email_l, pass_l)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.update({
                            "user_id":    user["id"],
                            "user_name":  user["full_name"],
                            "user_email": user["email"],
                        })
                        st.rerun()

            with tab_r:
                with st.form("reg_form"):
                    name_r  = st.text_input("Full Name", key="r_name")
                    email_r = st.text_input("Email",     key="r_email")
                    pass_r  = st.text_input("Password",  type="password", key="r_pass")
                    pass_r2 = st.text_input("Confirm",   type="password", key="r_pass2")
                    ok_r    = st.form_submit_button("Create Account", use_container_width=True)
                if ok_r:
                    if pass_r != pass_r2:
                        st.error("Passwords don't match.")
                    elif len(pass_r) < 6:
                        st.error("Password must be ≥ 6 characters.")
                    else:
                        user, err = _register(email_r, name_r, pass_r)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.update({
                                "user_id":    user["id"],
                                "user_name":  user["full_name"],
                                "user_email": user["email"],
                            })
                            st.rerun()

    st.divider()

    # ── Query History ─────────────────────────────────────────────────────────
    if logged_in:
        st.markdown('<span class="sidebar-label">Recent Queries</span>',
                    unsafe_allow_html=True)
        try:
            history_rows = _get_history(st.session_state.user_id, limit=15)
            if history_rows:
                for row in history_rows:
                    label = row["prescription"][:46] + ("…" if len(row["prescription"]) > 46 else "")
                    ts    = (row["created_at"] or "")[:10]
                    if st.button(label, key=f"h_{row['id']}", use_container_width=True,
                                 help=f"Asked on {ts}"):
                        saved = row.get("results") or []
                        st.session_state.messages = [
                            {"role": "user",      "content": row["prescription"]},
                            {"role": "assistant", "content": "\n\n".join(
                                r.get("answer", "") for r in saved),
                             "data": saved},
                        ]
                        st.rerun()
            else:
                st.caption("No history yet — start chatting!")
        except Exception:
            st.caption("History unavailable.")

        st.divider()

        # ── Medications ───────────────────────────────────────────────────────
        with st.expander("💊 My Medications", expanded=True):
            try:
                db_meds = _get_medications(st.session_state.user_id)
            except Exception:
                db_meds = []

            with st.form("add_med", clear_on_submit=True):
                nd = st.text_input("Drug name *", placeholder="e.g. Metformin")
                c1, c2 = st.columns(2)
                dos = c1.text_input("Dosage",    placeholder="500 mg")
                frq = c2.text_input("Frequency", placeholder="twice daily")
                if st.form_submit_button("Add Medication", use_container_width=True):
                    if nd.strip():
                        _add_medication(st.session_state.user_id, nd, dos, frq)
                        st.rerun()
                    else:
                        st.warning("Drug name required.")

            for med in db_meds:
                name = med["drug_name"]
                info = " · ".join(filter(None, [med.get("dosage"), med.get("frequency")]))
                mc1, mc2 = st.columns([5, 1])
                mc1.markdown(
                    f"**{name}**" + (f"  \n<small style='color:#A09A94'>{info}</small>" if info else ""),
                    unsafe_allow_html=True,
                )
                if mc2.button("✕", key=f"d_{med['id']}"):
                    _delete_medication(med["id"])
                    st.rerun()

            if not db_meds:
                st.caption("No medications saved yet.")


# ══════════════════════════════════════════════════════════════════════════════
# LOCATION BAR
# ══════════════════════════════════════════════════════════════════════════════

def _location_bar():
    has_loc = st.session_state.user_lat is not None

    if has_loc:
        lat = st.session_state.user_lat
        lon = st.session_state.user_lon
        col1, col2 = st.columns([6, 1])
        col1.markdown(
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
            f'color:#A09A94;">📍 {lat:.4f}, {lon:.4f} — pharmacies enabled</span>',
            unsafe_allow_html=True,
        )
        if col2.button("Change", key="loc_change"):
            st.session_state.user_lat = None
            st.session_state.user_lon = None
            st.rerun()
    else:
        st.info("📍 **Share your location** to find nearby pharmacies after each drug query.")
        col1, col2, col3 = st.columns([2, 1, 1])

        if col1.button("🌐 Detect my location", key="loc_auto"):
            st.session_state["_detect_loc"] = True
            st.rerun()

        if st.session_state.get("_detect_loc"):
            loc = get_geolocation()
            if loc and loc.get("coords"):
                st.session_state.user_lat = loc["coords"]["latitude"]
                st.session_state.user_lon = loc["coords"]["longitude"]
                st.session_state["_detect_loc"] = False
                st.rerun()
            else:
                st.caption("⏳ Waiting for browser permission…")

        with st.expander("Or enter manually", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 1])
            mlat = c1.number_input("Latitude",  value=17.3850, format="%.4f", key="m_lat")
            mlon = c2.number_input("Longitude", value=78.4867, format="%.4f", key="m_lon")
            if c3.button("Set", key="loc_manual"):
                st.session_state.user_lat = mlat
                st.session_state.user_lon = mlon
                st.rerun()


_location_bar()
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PHARMACY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _show_pharmacies(drug_name: str):
    lat = st.session_state.user_lat
    lon = st.session_state.user_lon
    if lat is None or lon is None:
        return

    with st.spinner(f"Finding pharmacies near you for **{drug_name}**…"):
        results = find_pharmacies(lat=lat, lon=lon, radius_m=5000, drug_name=drug_name)

    if not results:
        st.warning(f"No pharmacies found within 5 km for {drug_name}.")
        return

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
        f'letter-spacing:0.09em;text-transform:uppercase;color:#A09A94;margin-bottom:0.6rem;">'
        f'Nearby Pharmacies · {drug_name.title()}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, p in enumerate(results[:6]):
        name     = p.get("name", "Unknown Pharmacy")
        addr     = p.get("address", "")
        dist     = p.get("distance_label", "")
        maps_url = p.get("maps_url", "#")
        dir_url  = p.get("directions_url", maps_url)
        phone    = p.get("phone", "")
        hours    = p.get("opening_hours", "")
        with cols[i % 2]:
            st.markdown(
                f'<div class="pharm-card">'
                f'<strong>{name}</strong><br>'
                f'{"📍 " + addr + "<br>" if addr else ""}'
                f'🚶 {dist}<br>'
                f'{"📞 " + phone + "<br>" if phone else ""}'
                f'{"🕐 " + hours + "<br>" if hours else ""}'
                f'<a href="{maps_url}" target="_blank">🗺 Map</a>&nbsp;&nbsp;'
                f'<a href="{dir_url}" target="_blank">🧭 Directions</a>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    greeting = f"Hello, {st.session_state.user_name}. " if logged_in else ""
    st.markdown(
        f'<div class="welcome-hero">'
        f'<div class="welcome-eyebrow">⬡ CRAG &nbsp;·&nbsp; FDA &nbsp;·&nbsp; 745K Vectors</div>'
        f'<h1 class="welcome-title">Drug<em>Safe</em> AI</h1>'
        f'<p class="welcome-sub">{greeting}'
        f'Clinical-grade drug interaction intelligence.<br>'
        f'Ask about interactions, side effects, or any health question.</p>'
        f'<div class="welcome-stats">'
        f'<div class="stat-cell"><span class="stat-num">745K</span><span class="stat-label">FDA Vectors</span></div>'
        f'<div class="stat-cell"><span class="stat-num">CRAG</span><span class="stat-label">Retrieval</span></div>'
        f'<div class="stat-cell"><span class="stat-num">LLaMA</span><span class="stat-label">Generation</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not logged_in:
        st.info("💡 Sign in from the sidebar to save your query history and medications.")

    st.markdown(
        '<span class="chips-label">Try asking</span>',
        unsafe_allow_html=True,
    )
    examples = [
        "Is it safe to take ibuprofen with warfarin?",
        "What are the side effects of metformin?",
        "Can I take aspirin and clopidogrel together?",
        "What should I know about amoxicillin?",
        "What should I do for a mild fever at home?",
        "Are there foods to avoid with lisinopril?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["_pending_prompt"] = ex
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _crag_pill_html(status: str) -> str:
    """Return a colour-coded CRAG status micro-pill."""
    cls_map = {
        "correct":              "crag-correct",
        "ambiguous:broadened":  "crag-ambiguous",
        "incorrect:rewritten":  "crag-incorrect",
        "incorrect:no_evidence":"crag-incorrect",
    }
    cls = cls_map.get(status, "")
    label = status.replace(":", " · ")
    return f'<span class="crag-pill {cls}">{label}</span>'


def _render_results(results: list):
    for i, item in enumerate(results):
        mode        = item.get("mode", "rag")
        drug        = item.get("drug", "")
        answer      = item.get("answer", "")
        sources     = item.get("sources", [])
        crag_status = item.get("crag_status", "")

        badge_cls = "badge-ai" if mode == "general" else "badge-rag"
        badge_lbl = "AI Assistant" if mode == "general" else drug.upper()

        header_html = f'<span class="badge {badge_cls}">{badge_lbl}</span>'
        if crag_status and mode != "general":
            header_html += _crag_pill_html(crag_status)

        st.markdown(header_html, unsafe_allow_html=True)
        st.markdown(answer)

        if sources:
            with st.expander(f"📚 {len(sources)} FDA source(s)", expanded=False):
                for src in sources[:4]:
                    score_val = src.get("score", 0)
                    st.markdown(
                        f'<div class="source-entry">'
                        f'<span class="src-drug">{src.get("generic_name","")}</span>'
                        f'<span class="src-section">{src.get("section","").replace("_"," ")}</span>'
                        f'<span class="src-score">{score_val:.3f}</span>'
                        f'<div class="src-text">{src.get("text","")[:280]}…</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if i < len(results) - 1:
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY RENDER
# ══════════════════════════════════════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            data = msg.get("data")
            if data:
                _render_results(data)
            else:
                st.markdown(msg.get("content", ""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAT INPUT + PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

_typed   = st.chat_input("Ask about drug interactions or any health question…")
_pending = st.session_state.pop("_pending_prompt", None)
prompt   = _typed or _pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build medication context
    med_ctx = ""
    if logged_in:
        try:
            db_meds = _get_medications(st.session_state.user_id)
            if db_meds:
                med_ctx = "Patient's current medications: " + ", ".join(
                    m["drug_name"] + (f" {m['dosage']}" if m.get("dosage") else "")
                    for m in db_meds
                )
        except Exception:
            pass

    recent = [m for m in st.session_state.messages[-8:]
              if m["role"] in ("user", "assistant") and "content" in m]
    history_ctx = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:300]}"
        for m in recent[:-1]
    )
    if med_ctx:
        history_ctx = med_ctx + "\n" + history_ctx

    detected      = parse_prescription(prompt)
    response_data = []

    with st.chat_message("assistant"):
        placeholder = st.empty()

        if detected:
            for drug in detected:
                with st.status(f"🔍 Searching FDA database for **{drug}**…",
                               expanded=True) as status:
                    st.write("Retrieving chunks from Qdrant Cloud…")
                    prog = st.progress(0, text="Embedding query…")
                    prog.progress(30, text="Running vector search + CRAG grading…")
                    res  = answer_ddi(drug_name=drug, section="drug_interactions",
                                     top_k=5, history_context=history_ctx)
                    prog.progress(80, text="Generating answer via LLaMA…")
                    prog.progress(100, text="Done ✓")
                    status.update(label=f"✅ {drug.upper()} — done",
                                  state="complete", expanded=False)
                response_data.append({
                    "drug":        drug,
                    "answer":      res.get("answer", ""),
                    "sources":     res.get("sources", []),
                    "crag_status": res.get("crag_status", ""),
                    "mode":        "rag",
                })
        else:
            with st.status("💬 Thinking…", expanded=True) as status:
                st.write("Calling LLaMA via Groq…")
                prog = st.progress(0, text="Preparing response…")
                res  = answer_general(prompt, history_context=history_ctx)
                prog.progress(100, text="Done ✓")
                status.update(label="✅ Response ready", state="complete", expanded=False)
            response_data.append({
                "drug":    "DrugSafe AI",
                "answer":  res.get("answer", ""),
                "sources": [],
                "mode":    "general",
            })

        placeholder.empty()
        _render_results(response_data)

        if detected and st.session_state.user_lat is not None:
            st.divider()
            for drug in detected:
                _show_pharmacies(drug)

    plain = "\n\n".join(r["answer"] for r in response_data)
    st.session_state.messages.append({"role": "assistant", "content": plain,
                                      "data": response_data})

    if logged_in:
        try:
            _save_history(
                user_id      = st.session_state.user_id,
                prescription = prompt,
                detected     = detected,
                results      = [{k: v for k, v in r.items() if k != "sources"}
                                for r in response_data],
            )
        except Exception:
            pass
