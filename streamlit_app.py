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

# Use /tmp on Streamlit Cloud (source dir is read-only); local dev uses repo data dir
_DATASETS_DIR = Path("/tmp/ddi_datasets")
_FDA_CSV      = _DATASETS_DIR / "clean_ddi_dataset.csv"
_DDI_CSV      = _DATASETS_DIR / "fully_processed_dataset.csv"
_CHROMA_DIR   = Path("/tmp/ddi_chroma")

_HF_FILES = {
    "clean_ddi_dataset.csv":       "https://huggingface.co/datasets/wolfrum/ddi-data/resolve/main/clean_ddi_dataset.csv",
    "fully_processed_dataset.csv": "https://huggingface.co/datasets/wolfrum/ddi-data/resolve/main/fully_processed_dataset.csv",
}


def _download_data():
    """Download datasets from Hugging Face into /tmp if not already present."""
    import requests
    _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in _HF_FILES.items():
        dest = _DATASETS_DIR / filename
        if not dest.exists():
            with st.spinner(f"Downloading {filename} from Hugging Face…"):
                resp = requests.get(url, stream=True, timeout=600)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)


_download_data()


@st.cache_resource(show_spinner="Loading DrugSafe AI models…")
def _load_system():
    from config import COLLECTION_NAME
    from data_preprocessing import load_and_clean_data
    from drug_categorization import apply_product_type, apply_route_column
    from rag_pipeline import build_chunk_df, build_chroma_index, load_models
    import chromadb

    DATA_CSV   = str(_FDA_CSV)
    CHROMA_DIR = str(_CHROMA_DIR)

    df       = load_and_clean_data(DATA_CSV)
    df       = apply_route_column(df)
    df       = apply_product_type(df)
    chunk_df = build_chunk_df(df)

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    load_models()
    if COLLECTION_NAME not in existing:
        build_chroma_index(chunk_df, rebuild=False)

    b2g: Dict[str, str] = {}
    for _, row in chunk_df[["brand_name", "generic_name"]].drop_duplicates().iterrows():
        b = str(row["brand_name"]).strip().lower()
        g = str(row["generic_name"]).strip().lower()
        if b and b != "nan":
            b2g[b] = g

    generics     = set(chunk_df["generic_name"].str.lower().str.strip().dropna().unique())
    all_names    = generics | set(b2g.keys())
    sorted_names = sorted(all_names, key=len, reverse=True)
    patterns     = {n: re.compile(r"\b" + re.escape(n) + r"\b") for n in sorted_names}

    return chunk_df, b2g, sorted_names, patterns


chunk_df, _B2G, _SORTED, _PATTERNS = _load_system()

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
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main, .block-container {
    background-color: #ffffff !important; color: #1a1a1a !important;
}
/* Only hide the hamburger menu and footer — leave header/toggle untouched */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 820px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f7f7f8 !important;
    border-right: 1px solid #e5e5e5 !important;
}
[data-testid="stSidebar"] * { color: #1a1a1a !important; }

/* Style the native sidebar toggle buttons green */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {
    background-color: #10a37f !important;
    color: white !important;
    border-radius: 50% !important;
}

/* Chat messages */
[data-testid="stChatMessage"] { background: #ffffff !important; color: #1a1a1a !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li { color: #1a1a1a !important; }
[data-testid="stChatMessage"][data-role="user"] {
    background: #f7f7f8 !important; border-radius: 12px; padding: 0.6rem 1rem;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important; color: #1a1a1a !important;
    border: 1px solid #d1d5db !important; border-radius: 12px !important;
}

/* Badges */
.badge {
    display: inline-block; font-size: 0.72rem; font-weight: 700;
    padding: 3px 14px; border-radius: 99px; letter-spacing: 0.05em; margin-bottom: 0.5rem;
}
.badge-rag { background: #10a37f; color: #fff; }
.badge-ai  { background: #7c3aed; color: #fff; }

/* Cards */
.user-chip {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 0.5rem; font-size: 0.88rem;
}
.guest-notice {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 0.5rem; font-size: 0.82rem; color: #92400e;
}
.pharm-card {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px;
    padding: 0.65rem 1rem; margin: 0.35rem 0; font-size: 0.88rem; line-height: 1.6;
}
.welcome-wrap { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.welcome-wrap h1 { font-size: 2.2rem; color: #1a1a1a; }
.welcome-wrap p  { color: #6b7280; max-width: 520px; margin: auto; }
div[data-testid="stButton"] button {
    border-radius: 20px !important; font-size: 0.85rem !important;
    color: #1a1a1a !important; background: #f7f7f8 !important;
    border: 1px solid #e5e5e5 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💊 DrugSafe AI")
    st.caption("FDA · 745K vectors · Groq LLaMA")
    st.divider()

    # ── Auth section ──────────────────────────────────────────────────────────
    if logged_in:
        st.markdown(
            f'<div class="user-chip">👤 <strong>{st.session_state.user_name}</strong>'
            f'<br><small>{st.session_state.user_email}</small></div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        if col1.button("＋ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if col2.button("Logout", use_container_width=True):
            st.session_state.update({"user_id": None, "user_name": None,
                                     "user_email": None, "messages": []})
            st.rerun()

    else:
        # Guest — show login/register inline
        st.markdown(
            '<div class="guest-notice">🔓 You\'re browsing as a <strong>guest</strong>.'
            ' Sign in to save history &amp; medications.</div>',
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
                            "user_id": user["id"],
                            "user_name": user["full_name"],
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
                                "user_id": user["id"],
                                "user_name": user["full_name"],
                                "user_email": user["email"],
                            })
                            st.rerun()

    st.divider()

    # ── Query History (only for logged-in users) ───────────────────────────────
    if logged_in:
        st.caption("📋  MY HISTORY")
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
        except Exception as e:
            st.caption(f"History unavailable.")

        st.divider()

        # ── Medications (only for logged-in users) ─────────────────────────────
        with st.expander("💊 My Medications", expanded=True):
            try:
                db_meds = _get_medications(st.session_state.user_id)
            except Exception:
                db_meds = []

            with st.form("add_med", clear_on_submit=True):
                nd = st.text_input("Drug name *", placeholder="e.g. Metformin")
                c1, c2 = st.columns(2)
                dos = c1.text_input("Dosage",    placeholder="500mg")
                frq = c2.text_input("Frequency", placeholder="twice daily")
                if st.form_submit_button("Add", use_container_width=True):
                    if nd.strip():
                        _add_medication(st.session_state.user_id, nd, dos, frq)
                        st.rerun()
                    else:
                        st.warning("Drug name required.")

            for med in db_meds:
                name = med["drug_name"]
                info = " · ".join(filter(None, [med.get("dosage"), med.get("frequency")]))
                mc1, mc2 = st.columns([5, 1])
                mc1.markdown(f"**{name}**" + (f"  \n<small>{info}</small>" if info else ""),
                             unsafe_allow_html=True)
                if mc2.button("✕", key=f"d_{med['id']}"):
                    _delete_medication(med["id"])
                    st.rerun()

            if not db_meds:
                st.caption("No medications saved yet.")



# ══════════════════════════════════════════════════════════════════════════════
# LOCATION BAR — auto-detect or manual, shown at top of main area
# ══════════════════════════════════════════════════════════════════════════════

def _location_bar():
    """Compact location row at top of main content."""
    has_loc = st.session_state.user_lat is not None

    if has_loc:
        lat = st.session_state.user_lat
        lon = st.session_state.user_lon
        loc_label = f"📍 {lat:.4f}, {lon:.4f}"
        col1, col2 = st.columns([5, 1])
        col1.caption(f"{loc_label} — pharmacies will show after drug queries")
        if col2.button("Change", key="loc_change"):
            st.session_state.user_lat = None
            st.session_state.user_lon = None
            st.rerun()
    else:
        st.info("📍 **Share your location** to automatically find nearby pharmacies after each drug query.")
        col1, col2, col3 = st.columns([2, 1, 1])

        # Auto-detect via browser
        if col1.button("🌐 Detect my location automatically", key="loc_auto"):
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

        # Manual fallback
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
    """Fetch and render nearby pharmacies for a drug inline in the chat."""
    lat = st.session_state.user_lat
    lon = st.session_state.user_lon
    if lat is None or lon is None:
        return

    with st.spinner(f"🏪 Finding pharmacies near you with **{drug_name}**…"):
        results = find_pharmacies(lat=lat, lon=lon, radius_m=5000, drug_name=drug_name)

    if not results:
        st.warning(f"No pharmacies found within 5 km. Try expanding your search radius.")
        return

    st.markdown(f"**🏪 Nearby pharmacies where you can get {drug_name.title()}:**")
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
                f'<a href="{maps_url}" target="_blank">🗺 Map</a> &nbsp; '
                f'<a href="{dir_url}" target="_blank">🧭 Directions</a>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    greeting = (f"Hello, <strong>{st.session_state.user_name}</strong>! "
                if logged_in else "")
    st.markdown(
        f'<div class="welcome-wrap"><h1>💊 DrugSafe AI</h1>'
        f'<p>{greeting}Ask about drug interactions, side effects, '
        f'or any health question — free for everyone.</p></div>',
        unsafe_allow_html=True,
    )
    if not logged_in:
        st.info("💡 Sign in from the sidebar to save your query history and medications.")

    st.markdown("##### Try asking:")
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


# ── Render helpers ─────────────────────────────────────────────────────────────

def _render_results(results: list):
    for i, item in enumerate(results):
        mode    = item.get("mode", "rag")
        drug    = item.get("drug", "")
        answer  = item.get("answer", "")
        sources = item.get("sources", [])

        badge_cls = "badge-ai" if mode == "general" else "badge-rag"
        badge_lbl = "AI Assistant" if mode == "general" else drug.upper()

        st.markdown(f'<span class="badge {badge_cls}">{badge_lbl}</span>',
                    unsafe_allow_html=True)
        st.markdown(answer)

        if sources:
            with st.expander(f"📚 {len(sources)} FDA source(s)", expanded=False):
                for src in sources[:4]:
                    st.caption(
                        f"**{src.get('generic_name','')}** · "
                        f"{src.get('section','')} · score {src.get('score',0):.3f}"
                    )
                    st.caption(f"_{src.get('text','')[:300]}…_")

        if i < len(results) - 1:
            st.divider()


# ── Render chat ────────────────────────────────────────────────────────────────

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
# CHAT INPUT
# ══════════════════════════════════════════════════════════════════════════════

# Pick up typed input OR a pending prompt from an example button click
_typed   = st.chat_input("Ask about drug interactions or health questions…")
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
                    st.write("Retrieving chunks from ChromaDB…")
                    prog = st.progress(0, text="Embedding query…")
                    prog.progress(30, text="Running vector search…")
                    res  = answer_ddi(drug_name=drug, section="drug_interactions",
                                     top_k=5, history_context=history_ctx)
                    prog.progress(80, text="Generating answer via LLaMA…")
                    prog.progress(100, text="Done ✓")
                    status.update(label=f"✅ {drug.upper()} — done",
                                  state="complete", expanded=False)
                response_data.append({"drug": drug, "answer": res.get("answer", ""),
                                      "sources": res.get("sources", []), "mode": "rag"})
        else:
            with st.status("💬 Thinking…", expanded=True) as status:
                st.write("Calling LLaMA via Groq…")
                prog = st.progress(0, text="Preparing response…")
                res  = answer_general(prompt, history_context=history_ctx)
                prog.progress(100, text="Done ✓")
                status.update(label="✅ Response ready", state="complete", expanded=False)
            response_data.append({"drug": "DrugSafe AI", "answer": res.get("answer", ""),
                                  "sources": [], "mode": "general"})

        placeholder.empty()
        _render_results(response_data)

        # Auto-show nearby pharmacies for each detected drug (if location is set)
        if detected and st.session_state.user_lat is not None:
            st.divider()
            for drug in detected:
                _show_pharmacies(drug)

    plain = "\n\n".join(r["answer"] for r in response_data)
    st.session_state.messages.append({"role": "assistant", "content": plain,
                                      "data": response_data})

    # Save history only for logged-in users
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
