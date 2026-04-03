"""
app.py — Flask REST API + single-page web UI for DDI-RAG.

Endpoints:
    Auth
        POST /api/auth/register
        POST /api/auth/login

    Medications  [JWT required]
        GET    /api/medications
        POST   /api/medications
        PUT    /api/medications/<id>
        DELETE /api/medications/<id>

    Query  [JWT optional]
        POST /api/query

    History  [JWT required]
        GET /api/history

    Sync  [no auth — restrict in prod]
        POST /api/sync
"""

import json
import logging
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from flask import Flask, make_response, request
from flask_cors import CORS
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    verify_jwt_in_request,
)

from auth import hash_password, init_jwt, verify_password
from config import DEFAULT_TOP_K, MAX_PRESCRIPTION_LEN, MAX_TOP_K, PORT
from database import get_db, init_db
from models import Medication, QueryHistory, User
from rag_pipeline import answer_ddi, safe_str

log = logging.getLogger("ddi.app")

flask_app = Flask("ddi")
CORS(flask_app)
init_jwt(flask_app)

# ── Lookup tables (populated by init_lookups) ─────────────────────────────────
_BRAND_TO_GENERIC:  Dict[str, str]        = {}
_SORTED_NAMES:      List[str]             = []
_COMPILED_PATTERNS: Dict[str, re.Pattern] = {}


def init_lookups(chunk_df) -> None:
    global _BRAND_TO_GENERIC, _SORTED_NAMES, _COMPILED_PATTERNS

    b2g: Dict[str, str] = {}
    for _, row in chunk_df[["brand_name", "generic_name"]].drop_duplicates().iterrows():
        b = str(row["brand_name"]).strip().lower()
        g = str(row["generic_name"]).strip().lower()
        if b and b != "nan":
            b2g[b] = g

    _BRAND_TO_GENERIC  = b2g
    generics           = set(chunk_df["generic_name"].str.lower().str.strip().dropna().unique())
    all_names          = generics | set(b2g.keys())
    _SORTED_NAMES      = sorted(all_names, key=len, reverse=True)
    _COMPILED_PATTERNS = {
        n: re.compile(r"\b" + re.escape(n) + r"\b") for n in _SORTED_NAMES
    }
    log.info("Lookup tables built: %d names indexed.", len(_SORTED_NAMES))


def parse_prescription(text: str) -> List[str]:
    text_lower = text.lower()
    found:    List[str]             = []
    consumed: List[Tuple[int, int]] = []

    for name in _SORTED_NAMES:
        for m in _COMPILED_PATTERNS[name].finditer(text_lower):
            s, e = m.start(), m.end()
            if not any(cs <= s < ce or cs < e <= ce for cs, ce in consumed):
                generic = _BRAND_TO_GENERIC.get(name, name)
                if generic not in found:
                    found.append(generic)
                consumed.append((s, e))
    return found


# ── JSON helper ───────────────────────────────────────────────────────────────
def _json(obj, status: int = 200):
    resp = make_response(json.dumps(obj, ensure_ascii=True, default=str), status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp


# ── History warning builder ───────────────────────────────────────────────────
def _build_history_warnings(detected: List[str], active_meds: List[Medication]) -> List[dict]:
    """
    Cross-reference newly detected drugs against the user's active medications.
    Returns a list of warning dicts when a new drug matches a past medication.
    """
    warnings = []
    active_names = {m.drug_name.lower().strip() for m in active_meds}

    for drug in detected:
        if drug.lower().strip() in active_names:
            matched = next(
                m for m in active_meds
                if m.drug_name.lower().strip() == drug.lower().strip()
            )
            detail = []
            if matched.dosage:
                detail.append(matched.dosage)
            if matched.frequency:
                detail.append(matched.frequency)
            if matched.start_date:
                detail.append(f"since {matched.start_date.isoformat()}")

            warnings.append({
                "drug":    drug,
                "message": (
                    f"You are already taking {drug}"
                    + (f" ({', '.join(detail)})" if detail else "")
                    + ". Please review interactions with your doctor."
                ),
                "start_date": matched.start_date.isoformat() if matched.start_date else None,
                "dosage":     matched.dosage,
                "frequency":  matched.frequency,
            })

    return warnings


def _format_history_context(active_meds: List[Medication]) -> str:
    """Format active medications as a text block for the RAG prompt."""
    if not active_meds:
        return ""
    lines = []
    for m in active_meds:
        parts = [m.drug_name]
        if m.dosage:
            parts.append(m.dosage)
        if m.frequency:
            parts.append(m.frequency)
        if m.start_date:
            parts.append(f"started {m.start_date.isoformat()}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


# ── Flask error handler ───────────────────────────────────────────────────────
@flask_app.errorhandler(Exception)
def handle_exception(e):
    log.exception("Unhandled exception")
    return _json({"error": safe_str(e)}, 500)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/auth/register", methods=["POST"])
def register():
    payload   = request.get_json(force=True, silent=True) or {}
    email     = safe_str(payload.get("email", "")).strip().lower()
    password  = safe_str(payload.get("password", "")).strip()
    full_name = safe_str(payload.get("full_name", "")).strip()

    if not email or not password or not full_name:
        return _json({"error": "full_name, email, and password are required."}, 400)
    if len(password) < 8:
        return _json({"error": "Password must be at least 8 characters."}, 400)

    with get_db() as db:
        if db.query(User).filter(User.email == email).first():
            return _json({"error": "An account with this email already exists."}, 409)

        user = User(
            email     = email,
            password  = hash_password(password),
            full_name = full_name,
        )
        db.add(user)
        db.flush()
        user_dict = user.to_dict()

    token = create_access_token(identity=user_dict["id"])
    return _json({"message": "Account created.", "user": user_dict, "access_token": token}, 201)


@flask_app.route("/api/auth/login", methods=["POST"])
def login():
    payload  = request.get_json(force=True, silent=True) or {}
    email    = safe_str(payload.get("email", "")).strip().lower()
    password = safe_str(payload.get("password", "")).strip()

    if not email or not password:
        return _json({"error": "email and password are required."}, 400)

    with get_db() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password):
            return _json({"error": "Invalid email or password."}, 401)
        user_dict = user.to_dict()

    token = create_access_token(identity=user_dict["id"])
    return _json({"access_token": token, "user": user_dict})


# ══════════════════════════════════════════════════════════════════════════════
# MEDICATION ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/medications", methods=["GET"])
@jwt_required()
def list_medications():
    user_id = get_jwt_identity()
    with get_db() as db:
        meds = (
            db.query(Medication)
            .filter(Medication.user_id == user_id)
            .order_by(Medication.created_at.desc())
            .all()
        )
        return _json([m.to_dict() for m in meds])


@flask_app.route("/api/medications", methods=["POST"])
@jwt_required()
def add_medication():
    user_id = get_jwt_identity()
    payload = request.get_json(force=True, silent=True) or {}

    drug_name = safe_str(payload.get("drug_name", "")).strip().lower()
    if not drug_name:
        return _json({"error": "drug_name is required."}, 400)

    # Parse optional date fields safely
    def _parse_date(val):
        if not val:
            return None
        try:
            return date.fromisoformat(str(val))
        except ValueError:
            return None

    med = Medication(
        user_id    = user_id,
        drug_name  = drug_name,
        dosage     = safe_str(payload.get("dosage", "")).strip() or None,
        frequency  = safe_str(payload.get("frequency", "")).strip() or None,
        start_date = _parse_date(payload.get("start_date")),
        end_date   = _parse_date(payload.get("end_date")),
        is_active  = payload.get("is_active", True),
        notes      = safe_str(payload.get("notes", "")).strip() or None,
    )

    with get_db() as db:
        db.add(med)
        db.flush()
        result = med.to_dict()

    return _json(result, 201)


@flask_app.route("/api/medications/<med_id>", methods=["PUT"])
@jwt_required()
def update_medication(med_id):
    user_id = get_jwt_identity()
    payload = request.get_json(force=True, silent=True) or {}

    def _parse_date(val):
        if not val:
            return None
        try:
            return date.fromisoformat(str(val))
        except ValueError:
            return None

    with get_db() as db:
        med = (
            db.query(Medication)
            .filter(Medication.id == med_id, Medication.user_id == user_id)
            .first()
        )
        if not med:
            return _json({"error": "Medication not found."}, 404)

        if "drug_name"  in payload:
            med.drug_name  = safe_str(payload["drug_name"]).strip().lower()
        if "dosage"     in payload:
            med.dosage     = safe_str(payload["dosage"]).strip() or None
        if "frequency"  in payload:
            med.frequency  = safe_str(payload["frequency"]).strip() or None
        if "start_date" in payload:
            med.start_date = _parse_date(payload["start_date"])
        if "end_date"   in payload:
            med.end_date   = _parse_date(payload["end_date"])
        if "is_active"  in payload:
            med.is_active  = bool(payload["is_active"])
        if "notes"      in payload:
            med.notes      = safe_str(payload["notes"]).strip() or None

        result = med.to_dict()

    return _json(result)


@flask_app.route("/api/medications/<med_id>", methods=["DELETE"])
@jwt_required()
def delete_medication(med_id):
    user_id = get_jwt_identity()
    with get_db() as db:
        med = (
            db.query(Medication)
            .filter(Medication.id == med_id, Medication.user_id == user_id)
            .first()
        )
        if not med:
            return _json({"error": "Medication not found."}, 404)
        db.delete(med)
    return _json({"message": "Medication deleted."})


# ══════════════════════════════════════════════════════════════════════════════
# QUERY ROUTE  (JWT optional — history only used when logged in)
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/query", methods=["POST"])
def query_api():
    # Try to get user identity — don't fail if no token
    user_id = None
    try:
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
    except Exception:
        pass

    payload      = request.get_json(force=True, silent=True) or {}
    prescription = safe_str(payload.get("prescription", "")).strip()
    top_k        = min(int(payload.get("top_k", DEFAULT_TOP_K)), MAX_TOP_K)

    if not prescription:
        return _json({"error": "prescription field is required."}, 400)
    if len(prescription) > MAX_PRESCRIPTION_LEN:
        return _json({"error": f"prescription must be <= {MAX_PRESCRIPTION_LEN} chars."}, 400)

    # Detect drugs
    try:
        detected = parse_prescription(prescription)
    except Exception:
        log.exception("parse_prescription failed")
        detected = []

    # Fetch user's active medications (if logged in)
    active_meds:    List[Medication] = []
    history_context: str             = ""
    history_warnings: List[dict]     = []

    if user_id:
        try:
            with get_db() as db:
                active_meds = (
                    db.query(Medication)
                    .filter(
                        Medication.user_id  == user_id,
                        Medication.is_active != False,
                    )
                    .all()
                )
                # Detach from session so we can use outside context
                db.expunge_all()
        except Exception:
            log.exception("Failed to fetch active medications for user %s", user_id)

        history_context  = _format_history_context(active_meds)
        history_warnings = _build_history_warnings(detected, active_meds)

    # Run RAG for each detected drug
    results = []
    for drug in (detected if detected else [None]):
        try:
            res = answer_ddi(
                drug_name       = drug,
                top_k           = top_k,
                history_context = history_context,
            )
            results.append({
                "drug"   : safe_str(drug or prescription),
                "answer" : res["answer"],
                "sources": res["sources"],
            })
        except Exception as exc:
            log.exception("Processing failed for drug=%s", drug)
            results.append({
                "drug"   : safe_str(drug or prescription),
                "answer" : f"Processing error: {safe_str(exc)}",
                "sources": [],
            })

    # Save to query history (if logged in)
    if user_id:
        try:
            with get_db() as db:
                db.add(QueryHistory(
                    user_id        = user_id,
                    prescription   = prescription,
                    detected_drugs = detected,
                    warnings       = history_warnings,
                    results        = [
                        {"drug": r["drug"], "answer": r["answer"]}
                        for r in results
                    ],
                ))
        except Exception:
            log.exception("Failed to save query history for user %s", user_id)

    return _json({
        "detected_drugs":   detected,
        "history_warnings": history_warnings,
        "results":          results,
    })


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY ROUTE
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/history", methods=["GET"])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    limit   = min(int(request.args.get("limit", 20)), 100)

    with get_db() as db:
        entries = (
            db.query(QueryHistory)
            .filter(QueryHistory.user_id == user_id)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
            .all()
        )
        return _json([e.to_dict() for e in entries])


# ══════════════════════════════════════════════════════════════════════════════
# FDA SYNC ROUTE
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/sync", methods=["POST"])
def trigger_sync():
    from fda_sync import run_sync
    payload = request.get_json(force=True, silent=True) or {}
    full    = bool(payload.get("full", False))
    result  = run_sync(full=full)
    return _json(result)


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND — Single-page app with auth + medication history
# ══════════════════════════════════════════════════════════════════════════════

FULL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>DDI — Drug Interaction Checker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Oxanium:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&family=Sora:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:#070B10; --surface:#0D1520; --surface-2:#111D2E;
      --border:#1C2D3F; --border-2:#243850;
      --accent:#00B4D8; --accent-dim:rgba(0,180,216,0.10);
      --success:#00C896; --warning:#F5A623; --danger:#FF4D6D;
      --text:#D8EAF8; --text-muted:#5E7A94; --text-dim:#2E4560;
      --mono:'JetBrains Mono',monospace; --sans:'Sora',sans-serif; --display:'Oxanium',sans-serif;
    }
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;
      background-image:linear-gradient(rgba(0,180,216,0.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,180,216,0.025) 1px,transparent 1px);
      background-size:48px 48px;pointer-events:none;z-index:0}
    .layout{position:relative;z-index:1;max-width:920px;margin:0 auto;padding:2.5rem 2rem 4rem}

    /* Nav */
    .nav{display:flex;align-items:center;justify-content:space-between;margin-bottom:2rem;flex-wrap:wrap;gap:.75rem}
    .nav-brand{font-family:var(--display);font-size:1.25rem;font-weight:600;color:#E8F4FE;letter-spacing:-.015em}
    .nav-brand em{color:var(--accent);font-style:normal}
    .nav-right{display:flex;align-items:center;gap:.6rem}
    .nav-user{font-family:var(--mono);font-size:11px;color:var(--text-muted)}

    /* Buttons */
    .btn{display:inline-flex;align-items:center;gap:7px;border:none;border-radius:8px;padding:9px 18px;
      font-family:var(--display);font-size:13px;font-weight:600;letter-spacing:.04em;cursor:pointer;
      transition:background .18s,transform .15s;white-space:nowrap}
    .btn:hover{transform:translateY(-1px)}.btn:active{transform:translateY(0)}
    .btn:disabled{opacity:.45;cursor:not-allowed;transform:none}
    .btn-primary{background:var(--accent);color:#060A0F}.btn-primary:hover{background:#00C9EE}
    .btn-outline{background:transparent;color:var(--accent);border:1px solid rgba(0,180,216,.35)}
    .btn-outline:hover{background:var(--accent-dim)}
    .btn-danger{background:rgba(255,77,109,.15);color:var(--danger);border:1px solid rgba(255,77,109,.3)}
    .btn-sm{padding:6px 12px;font-size:11.5px}

    /* Cards */
    .card{background:var(--surface);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:1.25rem}
    .card-accent{height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:.55}
    .card-body{padding:1.5rem}
    .card-title{font-family:var(--display);font-size:14px;font-weight:500;color:#E8F4FE;margin-bottom:1rem;
      letter-spacing:.03em;display:flex;align-items:center;gap:8px}

    /* Auth modal */
    .modal-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;
      align-items:center;justify-content:center}
    .modal-backdrop.open{display:flex}
    .modal{background:var(--surface);border:1px solid var(--border);border-radius:16px;
      padding:2rem;width:100%;max-width:420px;position:relative}
    .modal-title{font-family:var(--display);font-size:1.2rem;font-weight:600;color:#E8F4FE;
      margin-bottom:1.5rem}
    .modal-close{position:absolute;top:1rem;right:1rem;background:none;border:none;
      color:var(--text-muted);cursor:pointer;font-size:1.2rem;line-height:1}

    /* Forms */
    .field{margin-bottom:1rem}
    .field-label{display:block;font-family:var(--mono);font-size:10px;text-transform:uppercase;
      letter-spacing:.12em;color:var(--text-muted);margin-bottom:.5rem}
    input,textarea,select{width:100%;background:#06090D;color:var(--text);
      border:1px solid var(--border);border-radius:8px;padding:.65rem 1rem;
      font-family:var(--mono);font-size:13px;outline:none;transition:border-color .2s}
    input::placeholder,textarea::placeholder{color:var(--text-dim)}
    input:focus,textarea:focus,select:focus{border-color:rgba(0,180,216,.4)}
    textarea{min-height:110px;resize:vertical;line-height:1.75}
    .tab-bar{display:flex;gap:0;margin-bottom:1.5rem;border:1px solid var(--border);
      border-radius:8px;overflow:hidden}
    .tab{flex:1;padding:8px;font-family:var(--mono);font-size:11px;text-align:center;
      cursor:pointer;background:transparent;border:none;color:var(--text-muted);
      letter-spacing:.06em;text-transform:uppercase;transition:background .15s,color .15s}
    .tab.active{background:var(--accent-dim);color:var(--accent)}

    /* Sections */
    .section{display:none}.section.active{display:block}
    .page-title{font-family:var(--display);font-size:clamp(1.6rem,3.5vw,2.4rem);font-weight:600;
      letter-spacing:-.02em;color:#E8F4FE;line-height:1.15;margin-bottom:.4rem}
    .page-title em{color:var(--accent);font-style:normal}
    .page-sub{font-family:var(--mono);font-size:11px;color:var(--text-muted);letter-spacing:.04em;margin-bottom:2rem}

    /* Medication list */
    .med-grid{display:grid;gap:.75rem}
    .med-card{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;
      padding:.9rem 1.1rem;display:flex;align-items:center;justify-content:space-between;gap:.75rem}
    .med-name{font-family:var(--display);font-size:14px;color:#E8F4FE;text-transform:capitalize}
    .med-meta{font-family:var(--mono);font-size:10.5px;color:var(--text-muted);margin-top:3px}
    .med-actions{display:flex;gap:.4rem;flex-shrink:0}
    .badge-active{display:inline-block;font-family:var(--mono);font-size:9.5px;color:var(--success);
      background:rgba(0,200,150,.1);border:1px solid rgba(0,200,150,.25);border-radius:4px;
      padding:2px 7px;letter-spacing:.06em;text-transform:uppercase}
    .badge-inactive{display:inline-block;font-family:var(--mono);font-size:9.5px;color:var(--text-muted);
      background:var(--surface-2);border:1px solid var(--border);border-radius:4px;
      padding:2px 7px;letter-spacing:.06em;text-transform:uppercase}

    /* Query results */
    .drug-tags{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem}
    .drug-tag{display:inline-flex;align-items:center;gap:6px;font-family:var(--mono);
      font-size:12px;color:var(--accent);background:var(--accent-dim);
      border:1px solid rgba(0,180,216,.28);border-radius:100px;padding:4px 12px 4px 8px}
    .tag-dot{width:5px;height:5px;background:var(--accent);border-radius:50%;flex-shrink:0}
    .history-warn{display:flex;align-items:flex-start;gap:10px;background:rgba(245,166,35,.07);
      border:1px solid rgba(245,166,35,.25);border-radius:8px;padding:10px 14px;
      font-size:13px;color:var(--warning);line-height:1.55;margin-bottom:.75rem}
    .result-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;
      margin-bottom:1.25rem;overflow:hidden;opacity:0;animation:fadeUp .35s ease forwards}
    @keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
    .result-header{display:flex;align-items:center;padding:.9rem 1.25rem;
      border-bottom:1px solid var(--border);background:var(--surface-2)}
    .drug-name-label{font-family:var(--display);font-size:15px;font-weight:500;color:#E8F4FE}
    .rx-chip{font-family:var(--mono);font-size:10px;color:var(--accent);background:var(--accent-dim);
      border:1px solid rgba(0,180,216,.22);border-radius:4px;padding:2px 7px;margin-right:9px}
    .answer-box{background:#06090D;border:1px solid var(--border);border-radius:8px;
      padding:1rem 1.2rem;font-size:13.5px;line-height:1.85;color:var(--text);white-space:pre-wrap}
    .result-body{padding:1.25rem}
    .sources-btn{display:flex;align-items:center;gap:7px;background:none;border:none;padding:0;
      margin-top:1rem;cursor:pointer;font-family:var(--mono);font-size:10.5px;
      text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);transition:color .15s}
    .sources-btn:hover{color:var(--text)}
    .chevron{transition:transform .2s;font-size:9px}.chevron.open{transform:rotate(180deg)}
    .sources-list{display:none;margin-top:.75rem;border-top:1px solid var(--border);padding-top:.75rem}
    .sources-list.open{display:block}
    .src-row{display:grid;grid-template-columns:42px 1fr;gap:0 10px;
      padding:10px 0;border-bottom:1px solid var(--border)}
    .src-row:last-child{border-bottom:none}
    .src-num{font-family:var(--mono);font-size:10px;color:var(--accent);background:var(--accent-dim);
      border:1px solid rgba(0,180,216,.2);border-radius:4px;padding:2px 5px;text-align:center;height:fit-content;margin-top:1px}
    .src-meta{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:5px}
    .meta-tag{font-family:var(--mono);font-size:10px;color:var(--text-muted);background:var(--surface-2);
      border:1px solid var(--border);border-radius:4px;padding:2px 7px}
    .score-pill{font-family:var(--mono);font-size:10px;border-radius:4px;border:1px solid;padding:2px 7px}
    .sc-hi{color:var(--success);border-color:rgba(0,200,150,.3);background:rgba(0,200,150,.07)}
    .sc-md{color:var(--warning);border-color:rgba(245,166,35,.3);background:rgba(245,166,35,.07)}
    .sc-lo{color:var(--text-muted);border-color:var(--border);background:transparent}
    .src-text{font-family:var(--mono);font-size:11.5px;color:#7A9BB5;line-height:1.65}

    /* History list */
    .hist-card{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;
      padding:1rem 1.2rem;margin-bottom:.75rem}
    .hist-rx{font-family:var(--mono);font-size:12.5px;color:var(--text);margin-bottom:.4rem}
    .hist-meta{font-family:var(--mono);font-size:10px;color:var(--text-muted)}
    .hist-drugs{display:flex;flex-wrap:wrap;gap:6px;margin-top:.5rem}
    .hist-tag{font-family:var(--mono);font-size:10.5px;color:var(--accent);
      background:var(--accent-dim);border:1px solid rgba(0,180,216,.2);border-radius:100px;
      padding:2px 9px}

    /* Loading / error */
    #loadingPanel{display:none;background:var(--surface);border:1px solid var(--border);
      border-radius:14px;padding:1.75rem;text-align:center;margin-bottom:1.25rem}
    .scan-wrap{width:260px;height:2px;background:var(--border);border-radius:1px;overflow:hidden;margin:1rem auto}
    .scan-fill{height:100%;width:35%;background:var(--accent);border-radius:1px;animation:sweep 1.4s ease-in-out infinite}
    @keyframes sweep{0%{transform:translateX(-200%)}100%{transform:translateX(450%)}}
    .loading-label{font-family:var(--mono);font-size:11px;color:var(--text-muted);letter-spacing:.06em}
    #errorPanel{display:none;background:rgba(255,77,109,.07);border:1px solid rgba(255,77,109,.28);
      border-radius:14px;padding:1rem 1.25rem;margin-bottom:1.25rem}
    .err-label{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--danger);margin-bottom:6px}
    #errorMsg{font-size:13px;color:#C07080;white-space:pre-wrap}
    .empty-state{font-family:var(--mono);font-size:12px;color:var(--text-muted);
      padding:2rem;text-align:center;border:1px dashed var(--border);border-radius:10px}
    .footer{margin-top:3rem;padding-top:1.25rem;border-top:1px solid var(--border);
      font-family:var(--mono);font-size:10.5px;color:var(--text-dim);letter-spacing:.04em;
      display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem}
    @media(max-width:600px){.layout{padding:1.5rem 1rem 3rem}.nav{flex-direction:column;align-items:flex-start}}
  </style>
</head>
<body>
<div class="layout">

  <!-- Nav -->
  <nav class="nav">
    <div class="nav-brand">DDI <em>Checker</em></div>
    <div class="nav-right">
      <span class="nav-user" id="navUser"></span>
      <button class="btn btn-outline btn-sm" id="authBtn" onclick="openAuth()">Sign In</button>
      <button class="btn btn-danger btn-sm" id="logoutBtn" style="display:none" onclick="logout()">Sign Out</button>
    </div>
  </nav>

  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab active" onclick="showSection('querySection',this)">// Query</button>
    <button class="tab" onclick="showSection('medsSection',this)">// My Medications</button>
    <button class="tab" onclick="showSection('histSection',this);loadHistory()">// History</button>
  </div>

  <!-- ── QUERY SECTION ───────────────────────────────── -->
  <div class="section active" id="querySection">
    <div class="page-title">Drug <em>Interaction</em> Checker</div>
    <p class="page-sub">// ChromaDB &nbsp;&middot;&nbsp; BioMistral-7B &nbsp;&middot;&nbsp; FDA Label Data</p>

    <div class="card">
      <div class="card-accent"></div>
      <div class="card-body">
        <label class="field-label" for="rxInput">// Prescription Input</label>
        <textarea id="rxInput" placeholder="e.g. warfarin 5mg daily, aspirin 81mg, amoxicillin 500mg TID"></textarea>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-top:1rem;gap:.75rem;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--text-muted)">
            Evidence depth
            <select id="topKSelect" style="width:auto">
              <option value="3">3 sources</option>
              <option value="5" selected>5 sources</option>
              <option value="8">8 sources</option>
              <option value="10">10 sources</option>
            </select>
          </div>
          <button class="btn btn-primary" id="analyzeBtn" onclick="analyze()">
            <svg viewBox="0 0 16 16" width="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="5" r="3.2"/><path d="M8.8 7.6 2 14.4"/><path d="M5 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2"/>
            </svg>
            Analyse Interactions
          </button>
        </div>
      </div>
    </div>

    <div id="loadingPanel">
      <div class="loading-label">RETRIEVING FDA EVIDENCE — RUNNING INFERENCE</div>
      <div class="scan-wrap"><div class="scan-fill"></div></div>
      <div class="loading-label" style="opacity:.55;font-size:10.5px;margin-top:4px">This may take 20–40 seconds per drug</div>
    </div>
    <div id="errorPanel"><div class="err-label">// Error</div><div id="errorMsg"></div></div>
    <div id="queryResults"></div>
  </div>

  <!-- ── MEDICATIONS SECTION ────────────────────────── -->
  <div class="section" id="medsSection">
    <div class="page-title">My <em>Medications</em></div>
    <p class="page-sub">// Track your current medication list</p>

    <div class="card" id="addMedCard">
      <div class="card-accent"></div>
      <div class="card-body">
        <div class="card-title">
          <svg viewBox="0 0 16 16" width="14" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 3v10M3 8h10"/></svg>
          Add Medication
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.75rem">
          <div class="field" style="grid-column:1/-1">
            <label class="field-label">Drug Name *</label>
            <input id="medDrug" placeholder="e.g. warfarin" />
          </div>
          <div class="field">
            <label class="field-label">Dosage (optional)</label>
            <input id="medDosage" placeholder="e.g. 5mg" />
          </div>
          <div class="field">
            <label class="field-label">Frequency (optional)</label>
            <input id="medFreq" placeholder="e.g. once daily" />
          </div>
          <div class="field">
            <label class="field-label">Start Date (optional)</label>
            <input id="medStart" type="date" />
          </div>
          <div class="field">
            <label class="field-label">Notes (optional)</label>
            <input id="medNotes" placeholder="Any extra notes" />
          </div>
        </div>
        <button class="btn btn-primary" onclick="addMedication()" style="margin-top:.25rem">Add Medication</button>
      </div>
    </div>

    <div id="loginPromptMeds" style="display:none">
      <div class="empty-state">Sign in to manage your medication list.</div>
    </div>
    <div id="medList" class="med-grid"></div>
  </div>

  <!-- ── HISTORY SECTION ────────────────────────────── -->
  <div class="section" id="histSection">
    <div class="page-title">Query <em>History</em></div>
    <p class="page-sub">// Past prescription queries</p>
    <div id="loginPromptHist" style="display:none">
      <div class="empty-state">Sign in to view your query history.</div>
    </div>
    <div id="histList"></div>
  </div>

  <footer class="footer">
    <span>DDI CHECKER &middot; FDA LABEL DATA</span>
    <span>FOR EDUCATIONAL USE ONLY &mdash; NOT MEDICAL ADVICE</span>
  </footer>
</div>

<!-- Auth Modal -->
<div class="modal-backdrop" id="authModal">
  <div class="modal">
    <button class="modal-close" onclick="closeAuth()">&times;</button>
    <div class="tab-bar" style="margin-bottom:1.25rem">
      <button class="tab active" id="loginTab" onclick="switchAuthTab('login')">Sign In</button>
      <button class="tab" id="registerTab" onclick="switchAuthTab('register')">Register</button>
    </div>

    <!-- Login form -->
    <div id="loginForm">
      <div class="modal-title">Welcome back</div>
      <div class="field"><label class="field-label">Email</label><input id="loginEmail" type="email" placeholder="you@example.com"/></div>
      <div class="field"><label class="field-label">Password</label><input id="loginPassword" type="password" placeholder="••••••••"/></div>
      <div id="loginError" style="color:var(--danger);font-size:12px;margin-bottom:.75rem;font-family:var(--mono);display:none"></div>
      <button class="btn btn-primary" style="width:100%" onclick="doLogin()">Sign In</button>
    </div>

    <!-- Register form -->
    <div id="registerForm" style="display:none">
      <div class="modal-title">Create account</div>
      <div class="field"><label class="field-label">Full Name</label><input id="regName" placeholder="Your name"/></div>
      <div class="field"><label class="field-label">Email</label><input id="regEmail" type="email" placeholder="you@example.com"/></div>
      <div class="field"><label class="field-label">Password</label><input id="regPassword" type="password" placeholder="Min 8 characters"/></div>
      <div id="regError" style="color:var(--danger);font-size:12px;margin-bottom:.75rem;font-family:var(--mono);display:none"></div>
      <button class="btn btn-primary" style="width:100%" onclick="doRegister()">Create Account</button>
    </div>
  </div>
</div>

<script>
/* ── State ─────────────────────────────────────────── */
let TOKEN = localStorage.getItem('ddi_token') || '';
let USER  = JSON.parse(localStorage.getItem('ddi_user') || 'null');

/* ── Utils ─────────────────────────────────────────── */
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function scoreClass(n){ const v=parseFloat(n); return v>=0.75?'sc-hi':v>=0.50?'sc-md':'sc-lo'; }
function authHeaders(){ return TOKEN?{'Content-Type':'application/json','Authorization':'Bearer '+TOKEN}:{'Content-Type':'application/json'}; }

async function api(path,opts={}){
  const r = await fetch(path,{headers:authHeaders(),...opts});
  return r.json();
}

/* ── Auth state ─────────────────────────────────────── */
function updateNavAuth(){
  const loggedIn = !!TOKEN;
  document.getElementById('authBtn').style.display    = loggedIn?'none':'inline-flex';
  document.getElementById('logoutBtn').style.display  = loggedIn?'inline-flex':'none';
  document.getElementById('navUser').textContent      = loggedIn&&USER ? USER.full_name : '';
  document.getElementById('addMedCard').style.display = loggedIn?'block':'none';
  document.getElementById('loginPromptMeds').style.display = loggedIn?'none':'block';
  document.getElementById('loginPromptHist').style.display = loggedIn?'none':'block';
}

function openAuth(){ document.getElementById('authModal').classList.add('open'); }
function closeAuth(){ document.getElementById('authModal').classList.remove('open'); }

function switchAuthTab(tab){
  document.getElementById('loginForm').style.display    = tab==='login'?'block':'none';
  document.getElementById('registerForm').style.display = tab==='register'?'block':'none';
  document.getElementById('loginTab').classList.toggle('active', tab==='login');
  document.getElementById('registerTab').classList.toggle('active', tab==='register');
}

async function doLogin(){
  const email=document.getElementById('loginEmail').value.trim();
  const pass=document.getElementById('loginPassword').value;
  const el=document.getElementById('loginError');
  el.style.display='none';
  const data = await api('/api/auth/login',{method:'POST',body:JSON.stringify({email,password:pass})});
  if(data.error){ el.textContent=data.error; el.style.display='block'; return; }
  TOKEN=data.access_token; USER=data.user;
  localStorage.setItem('ddi_token',TOKEN);
  localStorage.setItem('ddi_user',JSON.stringify(USER));
  closeAuth(); updateNavAuth(); loadMedications();
}

async function doRegister(){
  const full_name=document.getElementById('regName').value.trim();
  const email=document.getElementById('regEmail').value.trim();
  const password=document.getElementById('regPassword').value;
  const el=document.getElementById('regError');
  el.style.display='none';
  const data = await api('/api/auth/register',{method:'POST',body:JSON.stringify({full_name,email,password})});
  if(data.error){ el.textContent=data.error; el.style.display='block'; return; }
  TOKEN=data.access_token; USER=data.user;
  localStorage.setItem('ddi_token',TOKEN);
  localStorage.setItem('ddi_user',JSON.stringify(USER));
  closeAuth(); updateNavAuth(); loadMedications();
}

function logout(){
  TOKEN=''; USER=null;
  localStorage.removeItem('ddi_token');
  localStorage.removeItem('ddi_user');
  updateNavAuth();
  document.getElementById('medList').innerHTML='';
  document.getElementById('histList').innerHTML='';
}

/* ── Sections ───────────────────────────────────────── */
function showSection(id,btn){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if(id==='medsSection') loadMedications();
}

/* ── Query ──────────────────────────────────────────── */
function setLoading(on){
  document.getElementById('loadingPanel').style.display=on?'block':'none';
  const btn=document.getElementById('analyzeBtn');
  btn.disabled=on;
  btn.innerHTML=on?'Analysing\u2026':'<svg viewBox="0 0 16 16" width="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="5" r="3.2"/><path d="M8.8 7.6 2 14.4"/><path d="M5 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2"/></svg> Analyse Interactions';
}
function showError(msg){ document.getElementById('errorMsg').textContent=msg; document.getElementById('errorPanel').style.display='block'; }
function clearUI(){ document.getElementById('errorPanel').style.display='none'; document.getElementById('queryResults').innerHTML=''; }

function toggleSources(btn,id){
  const list=document.getElementById(id);
  const open=list.classList.toggle('open');
  btn.querySelector('.chevron').classList.toggle('open',open);
  btn.querySelector('.src-lbl').textContent=open?'Hide sources':'Show sources';
}

async function analyze(){
  const text=document.getElementById('rxInput').value.trim();
  if(!text){alert('Please enter prescription text.');return;}
  const topK=parseInt(document.getElementById('topKSelect').value,10);
  clearUI(); setLoading(true);
  try{
    const data = await api('/api/query',{method:'POST',body:JSON.stringify({prescription:text,top_k:topK})});
    if(data.error){showError(data.error);return;}
    renderResults(data);
  }catch(e){showError('Network error: '+e.message);}
  finally{setLoading(false);}
}

function renderResults(data){
  const container=document.getElementById('queryResults');
  container.innerHTML='';

  // Drug tags
  if(data.detected_drugs&&data.detected_drugs.length){
    const tags=data.detected_drugs.map(d=>`<span class="drug-tag"><span class="tag-dot"></span>${esc(d)}</span>`).join('');
    container.innerHTML+=`<div class="drug-tags">${tags}</div>`;
  }

  // History warnings
  (data.history_warnings||[]).forEach(w=>{
    container.innerHTML+=`<div class="history-warn">&#x26A0;&nbsp; ${esc(w.message)}</div>`;
  });

  // Results
  (data.results||[]).forEach((res,i)=>{
    const sid='sources-'+i;
    const sourcesHtml=(res.sources&&res.sources.length)?`
      <button class="sources-btn" onclick="toggleSources(this,'${sid}')">
        <span class="chevron">&#x25BC;</span>
        <span class="src-lbl">Show sources</span>
        &nbsp;(${res.sources.length} FDA excerpt${res.sources.length>1?'s':''})
      </button>
      <div class="sources-list" id="${sid}">
        ${res.sources.map((s,j)=>`
          <div class="src-row">
            <span class="src-num">SRC${j+1}</span>
            <div>
              <div class="src-meta">
                <span class="meta-tag">${esc((s.section||'').replace(/_/g,' '))}</span>
                <span class="score-pill ${scoreClass(s.score)}">sim&nbsp;${parseFloat(s.score).toFixed(3)}</span>
              </div>
              <p class="src-text">${esc((s.text||'').substring(0,440))}${(s.text||'').length>440?'&hellip;':''}</p>
            </div>
          </div>`).join('')}
      </div>`:'';
    const card=document.createElement('div');
    card.className='result-card';
    card.style.animationDelay=(i*110)+'ms';
    card.innerHTML=`
      <div class="result-header"><span class="drug-name-label"><span class="rx-chip">Rx</span>${esc(res.drug)}</span></div>
      <div class="result-body">
        <div class="answer-box">${esc(res.answer)}</div>
        ${sourcesHtml}
      </div>`;
    container.appendChild(card);
  });
}

/* ── Medications ────────────────────────────────────── */
async function loadMedications(){
  if(!TOKEN){ return; }
  const data = await api('/api/medications');
  const list=document.getElementById('medList');
  if(!data||data.error||!data.length){
    list.innerHTML='<div class="empty-state">No medications added yet.</div>';
    return;
  }
  list.innerHTML=data.map(m=>`
    <div class="med-card">
      <div>
        <div class="med-name">${esc(m.drug_name)}</div>
        <div class="med-meta">
          ${m.dosage?esc(m.dosage)+'&nbsp;&middot;&nbsp;':''}
          ${m.frequency?esc(m.frequency)+'&nbsp;&middot;&nbsp;':''}
          ${m.start_date?'Since '+esc(m.start_date):''}
        </div>
      </div>
      <div class="med-actions">
        ${m.is_active!==false
          ?`<span class="badge-active">Active</span>`
          :`<span class="badge-inactive">Inactive</span>`}
        ${m.is_active!==false
          ?`<button class="btn btn-outline btn-sm" onclick="stopMed('${m.id}')">Stop</button>`:''}
        <button class="btn btn-danger btn-sm" onclick="deleteMed('${m.id}')">Remove</button>
      </div>
    </div>`).join('');
}

async function addMedication(){
  if(!TOKEN){openAuth();return;}
  const drug_name=document.getElementById('medDrug').value.trim();
  if(!drug_name){alert('Drug name is required.');return;}
  const payload={
    drug_name,
    dosage:    document.getElementById('medDosage').value.trim()||undefined,
    frequency: document.getElementById('medFreq').value.trim()||undefined,
    start_date:document.getElementById('medStart').value||undefined,
    notes:     document.getElementById('medNotes').value.trim()||undefined,
  };
  const data=await api('/api/medications',{method:'POST',body:JSON.stringify(payload)});
  if(data.error){alert(data.error);return;}
  ['medDrug','medDosage','medFreq','medStart','medNotes'].forEach(id=>document.getElementById(id).value='');
  loadMedications();
}

async function stopMed(id){
  await api('/api/medications/'+id,{method:'PUT',body:JSON.stringify({is_active:false,end_date:new Date().toISOString().split('T')[0]})});
  loadMedications();
}

async function deleteMed(id){
  if(!confirm('Remove this medication?'))return;
  await api('/api/medications/'+id,{method:'DELETE'});
  loadMedications();
}

/* ── History ────────────────────────────────────────── */
async function loadHistory(){
  if(!TOKEN){return;}
  const data=await api('/api/history?limit=20');
  const list=document.getElementById('histList');
  if(!data||data.error||!data.length){
    list.innerHTML='<div class="empty-state">No query history yet.</div>';
    return;
  }
  list.innerHTML=data.map(h=>`
    <div class="hist-card">
      <div class="hist-rx">${esc(h.prescription)}</div>
      <div class="hist-drugs">
        ${(h.detected_drugs||[]).map(d=>`<span class="hist-tag">${esc(d)}</span>`).join('')}
      </div>
      <div class="hist-meta" style="margin-top:.5rem">${h.created_at?new Date(h.created_at).toLocaleString():''}</div>
    </div>`).join('');
}

/* ── Init ───────────────────────────────────────────── */
updateNavAuth();
if(TOKEN) loadMedications();
</script>
</body>
</html>"""


@flask_app.route("/")
def index():
    resp = make_response(FULL_HTML)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False)
