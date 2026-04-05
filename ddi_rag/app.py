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
from rag_pipeline import answer_ddi, answer_general, safe_str

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


# ── Date parse helper ─────────────────────────────────────────────────────────
def _parse_date(val):
    """Parse an ISO date string to a date object, returning None on failure."""
    if not val:
        return None
    try:
        return date.fromisoformat(str(val))
    except ValueError:
        return None


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

    # ── Route: drug query → RAG pipeline / general question → medical assistant ──
    results = []

    if detected:
        # Drug names found → use RAG for each drug
        for drug in detected:
            try:
                res = answer_ddi(
                    drug_name       = drug,
                    top_k           = top_k,
                    history_context = history_context,
                )
                results.append({
                    "drug"   : safe_str(drug),
                    "answer" : res["answer"],
                    "sources": res["sources"],
                    "mode"   : "rag",
                })
            except Exception as exc:
                log.exception("RAG failed for drug=%s", drug)
                results.append({
                    "drug"   : safe_str(drug),
                    "answer" : f"Processing error: {safe_str(exc)}",
                    "sources": [],
                    "mode"   : "rag",
                })
    else:
        # No drugs detected → general medical assistant via Groq directly
        try:
            res = answer_general(
                query           = prescription,
                history_context = history_context,
            )
            results.append({
                "drug"   : "DrugSafe AI",
                "answer" : res["answer"],
                "sources": [],
                "mode"   : "general",
            })
        except Exception as exc:
            log.exception("General answer failed")
            results.append({
                "drug"   : "DrugSafe AI",
                "answer" : f"Processing error: {safe_str(exc)}",
                "sources": [],
                "mode"   : "general",
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
# NEARBY PHARMACY ROUTE
# ══════════════════════════════════════════════════════════════════════════════

@flask_app.route("/api/nearby-pharmacy", methods=["GET"])
def nearby_pharmacy():
    """
    Find pharmacies near the given GPS coordinates using OpenStreetMap.
    Query params: lat, lon, radius (metres, default 5000), drug (optional label)
    """
    try:
        lat    = float(request.args.get("lat", 0))
        lon    = float(request.args.get("lon", 0))
        radius = min(int(request.args.get("radius", 5000)), 20000)
        drug   = safe_str(request.args.get("drug", "")).strip()
    except (ValueError, TypeError):
        return _json({"error": "lat and lon must be valid numbers."}, 400)

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return _json({"error": "Invalid coordinates."}, 400)

    from pharmacy_search import find_pharmacies
    pharmacies = find_pharmacies(lat=lat, lon=lon, radius_m=radius, drug_name=drug)
    return _json({"drug": drug, "count": len(pharmacies), "pharmacies": pharmacies})


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
  <title>DrugSafe AI — FDA-Powered Drug Interaction Checker</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #ffffff; --sidebar-bg: #f9f9f9; --surface: #ffffff;
      --border: #e5e5e5; --border-dark: #d1d1d1;
      --accent: #10a37f; --accent-hover: #0d8f6f; --accent-light: #f0faf6;
      --danger: #ef4444; --warning: #f59e0b; --success: #10b981;
      --text: #0d0d0d; --text-muted: #6b7280; --text-light: #9ca3af;
      --user-bubble: #f4f4f4; --font: 'Inter', -apple-system, sans-serif;
    }
    *,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: var(--font); background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; display: flex; }

    /* ── Sidebar ── */
    .sidebar { width: 260px; min-width: 260px; background: var(--sidebar-bg); border-right: 1px solid var(--border); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    .sidebar-top { padding: 12px; }
    .new-query-btn { width: 100%; display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: white; color: var(--text); font-size: 14px; font-weight: 500; cursor: pointer; transition: background .15s; }
    .new-query-btn:hover { background: var(--border); }
    .new-query-btn svg { color: var(--text-muted); flex-shrink: 0; }
    .sidebar-section-label { padding: 8px 12px 4px; font-size: 11px; font-weight: 600; color: var(--text-light); text-transform: uppercase; letter-spacing: .06em; }
    .sidebar-history { flex: 1; overflow-y: auto; padding: 4px 8px; }
    .sidebar-history::-webkit-scrollbar { width: 4px; }
    .sidebar-history::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 2px; }
    .hist-item { padding: 8px 10px; border-radius: 8px; cursor: pointer; font-size: 13px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: background .12s; }
    .hist-item:hover { background: var(--border); color: var(--text); }
    .hist-item .hist-time { font-size: 11px; color: var(--text-light); margin-top: 2px; }
    .sidebar-footer { border-top: 1px solid var(--border); padding: 10px; }
    .sidebar-footer-btn { width: 100%; display: flex; align-items: center; gap: 10px; padding: 9px 10px; border-radius: 8px; background: none; border: none; color: var(--text-muted); font-size: 13px; font-family: var(--font); cursor: pointer; transition: background .12s; }
    .sidebar-footer-btn:hover { background: var(--border); color: var(--text); }
    .user-avatar { width: 28px; height: 28px; border-radius: 50%; background: var(--accent); color: white; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; flex-shrink: 0; }

    /* ── Main area ── */
    .main { flex: 1; display: flex; flex-direction: column; height: 100vh; overflow: hidden; position: relative; }

    /* ── Topbar ── */
    .topbar { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 1px solid var(--border); background: white; flex-shrink: 0; }
    .topbar-brand { font-size: 16px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px; }
    .topbar-brand .brand-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
    .topbar-right { display: flex; align-items: center; gap: 10px; }
    .topbar-user { font-size: 13px; font-weight: 500; color: var(--text-muted); }
    .btn { display: inline-flex; align-items: center; gap: 7px; border: none; border-radius: 8px; padding: 8px 16px; font-family: var(--font); font-size: 13px; font-weight: 500; cursor: pointer; transition: background .15s; white-space: nowrap; }
    .btn:disabled { opacity: .5; cursor: not-allowed; }
    .btn-primary { background: var(--accent); color: white; }
    .btn-primary:hover { background: var(--accent-hover); }
    .btn-outline { background: white; color: var(--text); border: 1px solid var(--border-dark); }
    .btn-outline:hover { background: var(--user-bubble); }
    .btn-ghost { background: none; color: var(--text-muted); border: 1px solid transparent; }
    .btn-ghost:hover { background: var(--user-bubble); }
    .btn-danger { background: #fef2f2; color: var(--danger); border: 1px solid #fecaca; }
    .btn-danger:hover { background: #fee2e2; }
    .btn-sm { padding: 6px 12px; font-size: 12px; }

    /* ── Content area ── */
    .content-area { flex: 1; overflow-y: auto; padding: 0; display: flex; flex-direction: column; }
    .content-area::-webkit-scrollbar { width: 6px; }
    .content-area::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 3px; }

    /* Welcome state */
    .welcome-state { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; text-align: center; }
    .welcome-logo { width: 56px; height: 56px; border-radius: 16px; background: var(--accent); display: flex; align-items: center; justify-content: center; margin: 0 auto 1.5rem; }
    .welcome-title { font-size: 28px; font-weight: 700; color: var(--text); margin-bottom: .5rem; }
    .welcome-sub { font-size: 14px; color: var(--text-muted); margin-bottom: 2rem; }
    .welcome-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; max-width: 500px; }
    .welcome-pill { background: white; border: 1px solid var(--border); border-radius: 20px; padding: 8px 16px; font-size: 13px; color: var(--text-muted); cursor: pointer; transition: border-color .15s, color .15s; }
    .welcome-pill:hover { border-color: var(--accent); color: var(--accent); }

    /* Results */
    .results-wrap { padding: 20px 24px; max-width: 760px; margin: 0 auto; width: 100%; }
    .query-bubble { background: var(--user-bubble); border-radius: 18px 18px 4px 18px; padding: 12px 18px; font-size: 14px; margin-bottom: 20px; display: inline-block; max-width: 85%; }
    .drug-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
    .drug-chip { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; font-weight: 500; color: var(--accent); background: var(--accent-light); border: 1px solid #a7f3d0; border-radius: 100px; padding: 3px 10px; }
    .history-warn-box { display: flex; align-items: flex-start; gap: 10px; background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 12px 16px; font-size: 13.5px; color: #92400e; line-height: 1.6; margin-bottom: 12px; }
    .result-card { background: white; border: 1px solid var(--border); border-radius: 12px; margin-bottom: 16px; overflow: hidden; animation: fadeUp .3s ease forwards; }
    @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    .result-header { display: flex; align-items: center; gap: 10px; padding: 12px 18px; border-bottom: 1px solid var(--border); background: #fafafa; }
    .rx-pill { background: var(--accent-light); color: var(--accent); font-size: 11px; font-weight: 600; border-radius: 6px; padding: 2px 8px; border: 1px solid #a7f3d0; }
    .drug-name-label { font-size: 15px; font-weight: 600; color: var(--text); text-transform: capitalize; }
    .result-body { padding: 16px 18px; }
    .answer-text { font-size: 14px; line-height: 1.8; color: var(--text); }
    .answer-text h3 { font-size: 15px; font-weight: 700; margin: .9rem 0 .3rem; color: var(--text); }
    .answer-text h4 { font-size: 14px; font-weight: 600; margin: .75rem 0 .25rem; color: var(--text); }
    .answer-text h5 { font-size: 13px; font-weight: 600; margin: .6rem 0 .2rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; }
    .answer-text p  { margin: .45rem 0; }
    .answer-text ul,.answer-text ol { margin: .5rem 0; padding-left: 1.4rem; }
    .answer-text li { margin: .25rem 0; }
    .answer-text strong { font-weight: 600; color: var(--text); }
    .answer-text em { font-style: italic; color: var(--text-muted); }
    .answer-text code { background: #f3f4f6; border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; font-size: 12px; font-family: 'Courier New', monospace; }
    .answer-text hr { border: none; border-top: 1px solid var(--border); margin: .75rem 0; }
    .sources-toggle { display: inline-flex; align-items: center; gap: 6px; margin-top: 14px; background: none; border: none; padding: 0; font-size: 12px; color: var(--text-muted); cursor: pointer; font-family: var(--font); transition: color .15s; }
    .sources-toggle:hover { color: var(--text); }
    .sources-panel { display: none; margin-top: 12px; border-top: 1px solid var(--border); padding-top: 12px; }
    .sources-panel.open { display: block; }
    .src-item { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--border); }
    .src-item:last-child { border-bottom: none; }
    .src-num { flex-shrink: 0; width: 22px; height: 22px; border-radius: 50%; background: var(--accent-light); color: var(--accent); font-size: 11px; font-weight: 600; display: flex; align-items: center; justify-content: center; margin-top: 2px; }
    .src-info { flex: 1; }
    .src-badges { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; }
    .src-badge { font-size: 11px; background: var(--user-bubble); border: 1px solid var(--border); border-radius: 4px; padding: 1px 7px; color: var(--text-muted); }
    .score-hi { background: #f0fdf4; border-color: #bbf7d0; color: #166534; }
    .score-md { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .src-text { font-size: 12px; color: var(--text-muted); line-height: 1.6; }

    /* Loading */
    .loading-card { background: white; border: 1px solid var(--border); border-radius: 12px; padding: 20px 18px; margin-bottom: 16px; display: none; }
    .loading-dots { display: flex; gap: 4px; align-items: center; }
    .loading-dots span { width: 6px; height: 6px; border-radius: 50%; background: var(--text-light); animation: dot-bounce .8s ease-in-out infinite; }
    .loading-dots span:nth-child(2) { animation-delay: .15s; }
    .loading-dots span:nth-child(3) { animation-delay: .3s; }
    @keyframes dot-bounce { 0%,80%,100% { transform:scale(.8); opacity:.5; } 40% { transform:scale(1); opacity:1; } }
    .loading-text { font-size: 13px; color: var(--text-muted); margin-left: 10px; }
    .error-box { background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 12px 16px; font-size: 13px; color: var(--danger); margin-bottom: 12px; display: none; }

    /* ── Input area ── */
    .input-area { flex-shrink: 0; padding: 16px 24px 20px; background: white; border-top: 1px solid var(--border); }
    .input-wrap { max-width: 760px; margin: 0 auto; position: relative; }
    .input-box { width: 100%; border: 1px solid var(--border-dark); border-radius: 14px; padding: 14px 52px 14px 16px; font-family: var(--font); font-size: 14px; color: var(--text); resize: none; outline: none; line-height: 1.5; max-height: 160px; overflow-y: auto; background: white; transition: border-color .2s, box-shadow .2s; }
    .input-box:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(16,163,127,.12); }
    .input-box::placeholder { color: var(--text-light); }
    .send-btn { position: absolute; right: 10px; bottom: 10px; width: 34px; height: 34px; border-radius: 8px; border: none; background: var(--accent); color: white; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: background .15s, opacity .15s; }
    .send-btn:hover { background: var(--accent-hover); }
    .send-btn:disabled { background: var(--border-dark); cursor: not-allowed; }
    .input-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }
    .input-meta-left { font-size: 11px; color: var(--text-light); }
    .depth-select { font-size: 12px; color: var(--text-muted); background: none; border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-family: var(--font); cursor: pointer; outline: none; }

    /* ── Modals ── */
    .modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 200; align-items: center; justify-content: center; }
    .modal-backdrop.open { display: flex; }
    .modal { background: white; border-radius: 16px; padding: 28px; width: 100%; max-width: 420px; position: relative; box-shadow: 0 20px 60px rgba(0,0,0,.15); }
    .modal-close { position: absolute; top: 16px; right: 16px; background: none; border: none; font-size: 18px; color: var(--text-muted); cursor: pointer; line-height: 1; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border-radius: 6px; }
    .modal-close:hover { background: var(--user-bubble); }
    .modal-title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
    .modal-sub { font-size: 13px; color: var(--text-muted); margin-bottom: 20px; }
    .auth-tabs { display: flex; gap: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
    .auth-tab { flex: 1; padding: 8px; font-size: 13px; font-weight: 500; text-align: center; cursor: pointer; background: transparent; border: none; color: var(--text-muted); font-family: var(--font); transition: background .15s, color .15s; }
    .auth-tab.active { background: var(--accent); color: white; }
    .field { margin-bottom: 14px; }
    .field-label { display: block; font-size: 12px; font-weight: 500; color: var(--text-muted); margin-bottom: 5px; }
    input[type=text],input[type=email],input[type=password],input[type=date],textarea,select { width: 100%; background: white; color: var(--text); border: 1px solid var(--border-dark); border-radius: 8px; padding: 9px 12px; font-family: var(--font); font-size: 13px; outline: none; transition: border-color .2s; }
    input:focus,textarea:focus,select:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(16,163,127,.1); }
    input::placeholder,textarea::placeholder { color: var(--text-light); }
    .form-error { font-size: 12px; color: var(--danger); margin-bottom: 10px; display: none; }

    /* ── Meds modal ── */
    .meds-modal { max-width: 580px; }
    .meds-grid { display: grid; gap: 8px; margin-top: 12px; }
    .med-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 14px; border: 1px solid var(--border); border-radius: 10px; gap: 10px; background: #fafafa; }
    .med-name { font-size: 14px; font-weight: 500; text-transform: capitalize; }
    .med-detail { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
    .med-actions { display: flex; gap: 6px; flex-shrink: 0; }
    .badge-active { font-size: 11px; font-weight: 500; color: var(--success); background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 100px; padding: 2px 8px; }
    .badge-inactive { font-size: 11px; color: var(--text-light); background: var(--user-bubble); border: 1px solid var(--border); border-radius: 100px; padding: 2px 8px; }
    .empty-state { text-align: center; padding: 2rem 1rem; color: var(--text-muted); font-size: 13px; border: 1px dashed var(--border); border-radius: 10px; }

    /* Pharmacy cards */
    .pharmacy-card { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; background: #fafafa; }
    .pharmacy-name { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
    .pharmacy-dist { display: inline-block; font-size: 11px; font-weight: 600; color: var(--accent); background: var(--accent-light); border: 1px solid #a7f3d0; border-radius: 100px; padding: 2px 8px; margin-left: 8px; }
    .pharmacy-detail { font-size: 12px; color: var(--text-muted); margin-top: 4px; line-height: 1.6; }
    .pharmacy-links { display: flex; gap: 8px; margin-top: 8px; }
    .pharmacy-link { font-size: 12px; font-weight: 500; color: var(--accent); text-decoration: none; background: var(--accent-light); border: 1px solid #a7f3d0; border-radius: 6px; padding: 4px 10px; }
    .pharmacy-link:hover { background: #d1fae5; }
    .find-pharmacy-btn { display: inline-flex; align-items: center; gap: 6px; margin-top: 14px; font-size: 13px; font-weight: 500; color: var(--accent); background: var(--accent-light); border: 1px solid #a7f3d0; border-radius: 8px; padding: 7px 14px; cursor: pointer; border-style: solid; transition: background .15s; }
    .find-pharmacy-btn:hover { background: #d1fae5; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

    @media(max-width:700px){
      .sidebar { display: none; }
      .results-wrap, .input-wrap { padding-left: 1rem; padding-right: 1rem; }
    }
  </style>
</head>
<body>

<!-- ── Sidebar ── -->
<aside class="sidebar">
  <div class="sidebar-top">
    <button class="new-query-btn" onclick="newQuery()">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M8 3v10M3 8h10"/></svg>
      New Query
    </button>
  </div>
  <div class="sidebar-section-label">Recent</div>
  <div class="sidebar-history" id="sidebarHistory">
    <div style="padding:8px 10px;font-size:12px;color:var(--text-light)">No history yet</div>
  </div>
  <div class="sidebar-footer">
    <button class="sidebar-footer-btn" onclick="openMeds()">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="12" height="8" rx="2"/><path d="M5 6V4a3 3 0 0 1 6 0v2"/><circle cx="8" cy="10" r="1"/></svg>
      My Medications
    </button>
    <div id="sidebarUser" style="display:none">
      <button class="sidebar-footer-btn" onclick="logout()">
        <div class="user-avatar" id="userAvatar">U</div>
        <div style="flex:1;text-align:left">
          <div style="font-size:13px;font-weight:500;color:var(--text)" id="userName">User</div>
          <div style="font-size:11px;color:var(--text-light)">Sign out</div>
        </div>
      </button>
    </div>
  </div>
</aside>

<!-- ── Main ── -->
<main class="main">

  <!-- Topbar -->
  <div class="topbar">
    <div class="topbar-brand">
      <div class="brand-dot"></div>
      DrugSafe AI
    </div>
    <div class="topbar-right">
      <span class="topbar-user" id="topbarUser"></span>
      <button class="btn btn-outline btn-sm" id="authBtn" onclick="openAuth()">Log in</button>
    </div>
  </div>

  <!-- Scrollable content -->
  <div class="content-area" id="contentArea">

    <!-- Welcome state (shown when no results) -->
    <div class="welcome-state" id="welcomeState">
      <div class="welcome-logo">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
        </svg>
      </div>
      <div class="welcome-title">DrugSafe AI</div>
      <div class="welcome-sub">Medical assistant &middot; FDA drug interactions &middot; Llama 3.1 &middot; 930k+ vectors</div>
      <div class="welcome-pills">
        <span class="welcome-pill" onclick="setQuery('warfarin 5mg and aspirin 81mg')">warfarin + aspirin</span>
        <span class="welcome-pill" onclick="setQuery('amoxicillin 500mg')">amoxicillin</span>
        <span class="welcome-pill" onclick="setQuery('What should I do if the nearest pharmacy is closed?')">Pharmacy is closed, what do I do?</span>
        <span class="welcome-pill" onclick="setQuery('I feel dizzy after taking my medication, is this normal?')">I feel dizzy after my medication</span>
        <span class="welcome-pill" onclick="setQuery('metformin and ibuprofen')">metformin + ibuprofen</span>
        <span class="welcome-pill" onclick="setQuery('What are the signs of a drug allergic reaction?')">Signs of drug allergy?</span>
      </div>
    </div>

    <!-- Results (hidden initially) -->
    <div class="results-wrap" id="resultsWrap" style="display:none">
      <div id="loadingCard" class="loading-card">
        <div style="display:flex;align-items:center">
          <div class="loading-dots"><span></span><span></span><span></span></div>
          <span class="loading-text" id="loadingText">Thinking…</span>
        </div>
      </div>
      <div class="error-box" id="errorBox"></div>
      <div id="queryResults"></div>
    </div>

  </div>

  <!-- Input area -->
  <div class="input-area">
    <div class="input-wrap">
      <textarea class="input-box" id="rxInput" rows="1" placeholder="Ask anything — drug interactions, health questions, pharmacy help… (e.g. warfarin + aspirin, or 'what if I miss a dose?')"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();analyze();}"
        oninput="autoResize(this)"></textarea>
      <button class="send-btn" id="analyzeBtn" onclick="analyze()" title="Analyze">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="8" y1="14" x2="8" y2="2"/><polyline points="3 7 8 2 13 7"/>
        </svg>
      </button>
    </div>
    <div class="input-meta">
      <span class="input-meta-left">Press Enter to send &middot; Shift+Enter for new line</span>
      <select class="depth-select" id="topKSelect">
        <option value="3">3 sources</option>
        <option value="5" selected>5 sources</option>
        <option value="8">8 sources</option>
        <option value="10">10 sources</option>
      </select>
    </div>
  </div>

</main>

<!-- ── Auth Modal ── -->
<div class="modal-backdrop" id="authModal">
  <div class="modal">
    <button class="modal-close" onclick="closeAuth()">&times;</button>
    <div class="auth-tabs">
      <button class="auth-tab active" id="loginTab" onclick="switchAuthTab('login')">Sign In</button>
      <button class="auth-tab" id="registerTab" onclick="switchAuthTab('register')">Create Account</button>
    </div>
    <div id="loginForm">
      <div class="modal-title">Welcome back</div>
      <div class="modal-sub">Sign in to your DrugSafe AI account</div>
      <div class="field"><label class="field-label">Email</label><input id="loginEmail" type="email" placeholder="you@example.com"/></div>
      <div class="field"><label class="field-label">Password</label><input id="loginPassword" type="password" placeholder="••••••••" onkeydown="if(event.key==='Enter')doLogin()"/></div>
      <div class="form-error" id="loginError"></div>
      <button class="btn btn-primary" style="width:100%" onclick="doLogin()">Sign In</button>
    </div>
    <div id="registerForm" style="display:none">
      <div class="modal-title">Create account</div>
      <div class="modal-sub">Start tracking your medications safely</div>
      <div class="field"><label class="field-label">Full Name</label><input id="regName" placeholder="Your name"/></div>
      <div class="field"><label class="field-label">Email</label><input id="regEmail" type="email" placeholder="you@example.com"/></div>
      <div class="field"><label class="field-label">Password</label><input id="regPassword" type="password" placeholder="Min 8 characters" onkeydown="if(event.key==='Enter')doRegister()"/></div>
      <div class="form-error" id="regError"></div>
      <button class="btn btn-primary" style="width:100%" onclick="doRegister()">Create Account</button>
    </div>
  </div>
</div>

<!-- ── Pharmacy Modal ── -->
<div class="modal-backdrop" id="pharmacyModal">
  <div class="modal" style="max-width:520px">
    <button class="modal-close" onclick="document.getElementById('pharmacyModal').classList.remove('open')">&times;</button>
    <div class="modal-title" id="pharmacyModalTitle">Nearby Pharmacies</div>
    <div class="modal-sub" id="pharmacyModalSub">Finding pharmacies near you…</div>
    <div id="pharmacyResults" style="max-height:420px;overflow-y:auto;margin-top:8px"></div>
  </div>
</div>

<!-- ── Medications Modal ── -->
<div class="modal-backdrop" id="medsModal">
  <div class="modal meds-modal">
    <button class="modal-close" onclick="document.getElementById('medsModal').classList.remove('open')">&times;</button>
    <div class="modal-title">My Medications</div>
    <div class="modal-sub">Your active medication list is used to detect interaction warnings</div>
    <div id="medsLoginPrompt" style="display:none">
      <div class="empty-state">Please <a href="#" onclick="closeMeds();openAuth();return false" style="color:var(--accent)">sign in</a> to manage your medications.</div>
    </div>
    <div id="medsContent" style="display:none">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
        <div class="field" style="grid-column:1/-1">
          <label class="field-label">Drug Name *</label>
          <input id="medDrug" placeholder="e.g. warfarin"/>
        </div>
        <div class="field">
          <label class="field-label">Dosage (optional)</label>
          <input id="medDosage" placeholder="e.g. 5mg"/>
        </div>
        <div class="field">
          <label class="field-label">Frequency (optional)</label>
          <input id="medFreq" placeholder="e.g. once daily"/>
        </div>
        <div class="field">
          <label class="field-label">Start Date (optional)</label>
          <input id="medStart" type="date"/>
        </div>
        <div class="field">
          <label class="field-label">Notes (optional)</label>
          <input id="medNotes" placeholder="Any notes"/>
        </div>
      </div>
      <button class="btn btn-primary btn-sm" onclick="addMedication()" style="margin-bottom:16px">+ Add Medication</button>
      <div class="meds-grid" id="medList"></div>
    </div>
  </div>
</div>

<script>
/* ── Markdown renderer ── */
function applyInline(text){
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code>$1</code>');
}

function renderMarkdown(raw){
  // Escape HTML to prevent XSS, then render markdown on top
  const e = s => String(s||'')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  const lines = e(raw).split('\\n');
  let html = '', inUl = false, inOl = false;

  const closeList = () => {
    if(inUl){ html += '</ul>'; inUl = false; }
    if(inOl){ html += '</ol>'; inOl = false; }
  };

  for(const line of lines){
    if(/^### /.test(line)){
      closeList();
      html += `<h5>${applyInline(line.slice(4))}</h5>`;
    } else if(/^## /.test(line)){
      closeList();
      html += `<h4>${applyInline(line.slice(3))}</h4>`;
    } else if(/^# /.test(line)){
      closeList();
      html += `<h3>${applyInline(line.slice(2))}</h3>`;
    } else if(/^---+$/.test(line.trim())){
      closeList();
      html += '<hr>';
    } else if(/^[-*] (.+)/.test(line)){
      if(inOl){ html += '</ol>'; inOl = false; }
      if(!inUl){ html += '<ul>'; inUl = true; }
      html += `<li>${applyInline(line.replace(/^[-*] /,''))}</li>`;
    } else if(/^\d+\. (.+)/.test(line)){
      if(inUl){ html += '</ul>'; inUl = false; }
      if(!inOl){ html += '<ol>'; inOl = true; }
      html += `<li>${applyInline(line.replace(/^\d+\. /,''))}</li>`;
    } else if(line.trim() === ''){
      closeList();
      html += '<br>';
    } else {
      closeList();
      html += `<p>${applyInline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

/* ── State ── */
let TOKEN = localStorage.getItem('ddi_token') || '';
let USER  = JSON.parse(localStorage.getItem('ddi_user') || 'null');

/* ── Utils ── */
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function scoreClass(n){ const v=parseFloat(n); return v>=0.75?'score-hi':v>=0.50?'score-md':''; }
function authHdr(){ return TOKEN?{'Content-Type':'application/json','Authorization':'Bearer '+TOKEN}:{'Content-Type':'application/json'}; }
async function api(path,opts={}){ const r=await fetch(path,{headers:authHdr(),...opts}); return r.json(); }

function autoResize(el){
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,160)+'px';
}

/* ── Auth UI ── */
function updateUI(){
  const in_ = !!TOKEN;
  document.getElementById('authBtn').style.display       = in_?'none':'inline-flex';
  document.getElementById('sidebarUser').style.display   = in_?'block':'none';
  document.getElementById('topbarUser').textContent      = '';
  if(in_ && USER){
    document.getElementById('userName').textContent      = USER.full_name;
    document.getElementById('userAvatar').textContent    = USER.full_name[0].toUpperCase();
    document.getElementById('topbarUser').textContent    = USER.full_name;
  }
  if(in_) { loadSidebarHistory(); }
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
  const data=await api('/api/auth/login',{method:'POST',body:JSON.stringify({email,password:pass})});
  if(data.error){ el.textContent=data.error; el.style.display='block'; return; }
  TOKEN=data.access_token; USER=data.user;
  localStorage.setItem('ddi_token',TOKEN);
  localStorage.setItem('ddi_user',JSON.stringify(USER));
  closeAuth(); updateUI();
}

async function doRegister(){
  const full_name=document.getElementById('regName').value.trim();
  const email=document.getElementById('regEmail').value.trim();
  const password=document.getElementById('regPassword').value;
  const el=document.getElementById('regError');
  el.style.display='none';
  const data=await api('/api/auth/register',{method:'POST',body:JSON.stringify({full_name,email,password})});
  if(data.error){ el.textContent=data.error; el.style.display='block'; return; }
  TOKEN=data.access_token; USER=data.user;
  localStorage.setItem('ddi_token',TOKEN);
  localStorage.setItem('ddi_user',JSON.stringify(USER));
  closeAuth(); updateUI();
}

function logout(){
  TOKEN=''; USER=null;
  localStorage.removeItem('ddi_token');
  localStorage.removeItem('ddi_user');
  updateUI();
  document.getElementById('sidebarHistory').innerHTML='<div style="padding:8px 10px;font-size:12px;color:var(--text-light)">No history yet</div>';
  newQuery();
}

/* ── Query flow ── */
function newQuery(){
  document.getElementById('welcomeState').style.display='flex';
  document.getElementById('resultsWrap').style.display='none';
  document.getElementById('rxInput').value='';
  document.getElementById('rxInput').style.height='auto';
  document.getElementById('queryResults').innerHTML='';
  document.getElementById('errorBox').style.display='none';
}

function setQuery(text){
  document.getElementById('rxInput').value=text;
  document.getElementById('rxInput').focus();
}

function setLoading(on, text){
  document.getElementById('loadingCard').style.display=on?'block':'none';
  document.getElementById('analyzeBtn').disabled=on;
  document.getElementById('welcomeState').style.display='none';
  document.getElementById('resultsWrap').style.display='block';
  if(text) document.getElementById('loadingText').textContent=text;
}

function showError(msg){
  const el=document.getElementById('errorBox');
  el.textContent=msg; el.style.display='block';
}

function toggleSources(btn,id){
  const panel=document.getElementById(id);
  const open=panel.classList.toggle('open');
  btn.textContent=open?'▲ Hide sources':'▼ Show sources ('+btn.dataset.count+' excerpts)';
}

async function analyze(){
  const text=document.getElementById('rxInput').value.trim();
  if(!text) return;
  const topK=parseInt(document.getElementById('topKSelect').value,10);
  document.getElementById('queryResults').innerHTML='';
  document.getElementById('errorBox').style.display='none';
  // Give a hint about what mode we're in
  const looksLikeDrug = ['mg','mcg','ml','tablet','capsule','dose','daily','twice','bid','tid','qid'].some(w=>text.toLowerCase().includes(w)) || text.trim().split(' ').filter(Boolean).length <= 4;
  setLoading(true, looksLikeDrug ? 'Retrieving FDA evidence & running inference…' : 'Thinking…');
  try{
    const data=await api('/api/query',{method:'POST',body:JSON.stringify({prescription:text,top_k:topK})});
    if(data.error){ showError(data.error); return; }
    renderResults(data, text);
    if(TOKEN) loadSidebarHistory();
  } catch(e){ showError('Network error: '+e.message); }
  finally{ setLoading(false); }
}

function renderResults(data, query){
  const container=document.getElementById('queryResults');
  container.innerHTML='';

  // Query bubble
  container.innerHTML+=`<div style="display:flex;justify-content:flex-end;margin-bottom:16px">
    <div class="query-bubble">${esc(query)}</div></div>`;

  // Detected drug chips
  if(data.detected_drugs&&data.detected_drugs.length){
    const chips=data.detected_drugs.map(d=>`<span class="drug-chip">&#x2022; ${esc(d)}</span>`).join('');
    container.innerHTML+=`<div class="drug-chips">${chips}</div>`;
  }

  // History warnings (serious alerts)
  (data.history_warnings||[]).forEach(w=>{
    container.innerHTML+=`
      <div class="history-warn-box">
        <span style="font-size:18px;flex-shrink:0">&#x26A0;&#xFE0F;</span>
        <div><strong>Interaction Warning:</strong> ${esc(w.message)}</div>
      </div>`;
  });

  // Result cards
  (data.results||[]).forEach((res,i)=>{
    const sid='src-'+i;
    const srcCount=res.sources?res.sources.length:0;
    const srcHtml=srcCount?`
      <button class="sources-toggle" onclick="toggleSources(this,'${sid}')" data-count="${srcCount}">
        &#x25BC; Show sources (${srcCount} excerpt${srcCount>1?'s':''})
      </button>
      <div class="sources-panel" id="${sid}">
        ${res.sources.map((s,j)=>`
          <div class="src-item">
            <div class="src-num">${j+1}</div>
            <div class="src-info">
              <div class="src-badges">
                <span class="src-badge">${esc((s.section||'').replace(/_/g,' '))}</span>
                <span class="src-badge ${scoreClass(s.score)}">sim ${parseFloat(s.score||0).toFixed(3)}</span>
                ${s.generic_name?`<span class="src-badge">${esc(s.generic_name)}</span>`:''}
              </div>
              <div class="src-text">${esc((s.text||'').substring(0,440))}${(s.text||'').length>440?'…':''}</div>
            </div>
          </div>`).join('')}
      </div>`:'';

    const card=document.createElement('div');
    card.className='result-card';
    card.style.animationDelay=(i*80)+'ms';
    const isGeneral = res.mode === 'general';
    card.innerHTML=`
      <div class="result-header">
        <span class="rx-pill" style="${isGeneral?'background:#f0f0ff;color:#6366f1;border-color:#c7d2fe':''}">
          ${isGeneral?'AI':'Rx'}
        </span>
        <span class="drug-name-label">${esc(res.drug)}</span>
        ${isGeneral?'<span style="margin-left:auto;font-size:11px;color:var(--text-light)">General response · not FDA-sourced</span>':''}
      </div>
      <div class="result-body">
        <div class="answer-text">${renderMarkdown(res.answer)}</div>
        ${srcHtml}
      </div>`;
    container.appendChild(card);
  });

  // "Find Nearby Pharmacy" button — uses first detected drug name
  const primaryDrug = (data.detected_drugs&&data.detected_drugs.length) ? data.detected_drugs[0] : query;
  const pharmacyBtn = document.createElement('button');
  pharmacyBtn.className = 'find-pharmacy-btn';
  pharmacyBtn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg> Find Nearest Pharmacy`;
  pharmacyBtn.onclick = () => findNearbyPharmacy(primaryDrug);
  container.appendChild(pharmacyBtn);

  // Scroll to top of results
  document.getElementById('contentArea').scrollTop=0;
}

/* ── Sidebar history ── */
async function loadSidebarHistory(){
  if(!TOKEN) return;
  const data=await api('/api/history?limit=15');
  const el=document.getElementById('sidebarHistory');
  if(!data||data.error||!data.length){
    el.innerHTML='<div style="padding:8px 10px;font-size:12px;color:var(--text-light)">No history yet</div>';
    return;
  }
  el.innerHTML=data.map(h=>`
    <div class="hist-item" onclick="replayQuery(${JSON.stringify(esc(h.prescription))})">
      <div>${esc((h.prescription||'').substring(0,38))}${(h.prescription||'').length>38?'…':''}</div>
      <div class="hist-time">${h.created_at?new Date(h.created_at).toLocaleDateString():''}</div>
    </div>`).join('');
}

function replayQuery(prescription){
  document.getElementById('rxInput').value=prescription;
  analyze();
}

/* ── Medications modal ── */
function openMeds(){
  document.getElementById('medsModal').classList.add('open');
  if(TOKEN){
    document.getElementById('medsLoginPrompt').style.display='none';
    document.getElementById('medsContent').style.display='block';
    loadMedications();
  } else {
    document.getElementById('medsLoginPrompt').style.display='block';
    document.getElementById('medsContent').style.display='none';
  }
}
function closeMeds(){ document.getElementById('medsModal').classList.remove('open'); }

async function loadMedications(){
  if(!TOKEN) return;
  const data=await api('/api/medications');
  const list=document.getElementById('medList');
  if(!data||data.error||!data.length){
    list.innerHTML='<div class="empty-state">No medications added yet. Add one above.</div>';
    return;
  }
  list.innerHTML=data.map(m=>`
    <div class="med-row">
      <div>
        <div class="med-name">${esc(m.drug_name)}</div>
        <div class="med-detail">
          ${[m.dosage,m.frequency,m.start_date?'Since '+m.start_date:''].filter(Boolean).join(' · ')}
        </div>
      </div>
      <div class="med-actions">
        ${m.is_active!==false?`<span class="badge-active">Active</span>`:`<span class="badge-inactive">Stopped</span>`}
        ${m.is_active!==false?`<button class="btn btn-ghost btn-sm" onclick="stopMed('${m.id}')">Stop</button>`:''}
        <button class="btn btn-danger btn-sm" onclick="deleteMed('${m.id}')">Remove</button>
      </div>
    </div>`).join('');
}

async function addMedication(){
  if(!TOKEN){ closeMeds(); openAuth(); return; }
  const drug_name=document.getElementById('medDrug').value.trim();
  if(!drug_name){ alert('Drug name is required.'); return; }
  const payload={
    drug_name,
    dosage:    document.getElementById('medDosage').value.trim()||undefined,
    frequency: document.getElementById('medFreq').value.trim()||undefined,
    start_date:document.getElementById('medStart').value||undefined,
    notes:     document.getElementById('medNotes').value.trim()||undefined,
  };
  const data=await api('/api/medications',{method:'POST',body:JSON.stringify(payload)});
  if(data.error){ alert(data.error); return; }
  ['medDrug','medDosage','medFreq','medStart','medNotes'].forEach(id=>document.getElementById(id).value='');
  loadMedications();
}

async function stopMed(id){
  await api('/api/medications/'+id,{method:'PUT',body:JSON.stringify({is_active:false,end_date:new Date().toISOString().split('T')[0]})});
  loadMedications();
}

async function deleteMed(id){
  if(!confirm('Remove this medication?')) return;
  await api('/api/medications/'+id,{method:'DELETE'});
  loadMedications();
}

/* ── Nearby Pharmacy ── */
function findNearbyPharmacy(drugName){
  const modal = document.getElementById('pharmacyModal');
  const title = document.getElementById('pharmacyModalTitle');
  const sub   = document.getElementById('pharmacyModalSub');
  const res   = document.getElementById('pharmacyResults');

  title.textContent = drugName ? `Pharmacies stocking ${drugName}` : 'Nearby Pharmacies';
  sub.textContent   = 'Getting your location…';
  res.innerHTML     = '<div style="text-align:center;padding:2rem;color:var(--text-light)">📍 Locating you…</div>';
  modal.classList.add('open');

  if(!navigator.geolocation){
    sub.textContent='Geolocation not supported by your browser.';
    res.innerHTML='';
    return;
  }

  navigator.geolocation.getCurrentPosition(
    pos => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      sub.textContent = `Searching within 5 km of your location…`;
      fetch(`/api/nearby-pharmacy?lat=${lat}&lon=${lon}&radius=5000&drug=${encodeURIComponent(drugName||'')}`)
        .then(r=>r.json())
        .then(data=>{
          if(data.error){ res.innerHTML=`<div style="color:var(--danger);padding:1rem">${esc(data.error)}</div>`; return; }
          renderPharmacies(data, sub);
        })
        .catch(e=>{ res.innerHTML=`<div style="color:var(--danger);padding:1rem">Network error: ${esc(e.message)}</div>`; });
    },
    err => {
      sub.textContent = 'Location access denied.';
      res.innerHTML   = `<div style="padding:1rem;font-size:13px;color:var(--text-muted)">
        Please allow location access in your browser, or search manually on
        <a href="https://www.google.com/maps/search/pharmacy" target="_blank" style="color:var(--accent)">Google Maps</a>.
      </div>`;
    },
    { timeout: 10000 }
  );
}

function renderPharmacies(data, subEl){
  const res = document.getElementById('pharmacyResults');
  const pharmacies = data.pharmacies || [];

  if(!pharmacies.length){
    subEl.textContent = 'No pharmacies found within 5 km.';
    res.innerHTML = '<div class="empty-state">Try a larger radius or search on Google Maps.</div>';
    return;
  }

  subEl.textContent = `Found ${pharmacies.length} pharmacies nearby`;

  res.innerHTML = pharmacies.slice(0,10).map(p=>`
    <div class="pharmacy-card">
      <div class="pharmacy-name">
        🏥 ${esc(p.name)}
        <span class="pharmacy-dist">${esc(p.distance_label)}</span>
      </div>
      ${p.address!=='Address not listed'?`<div class="pharmacy-detail">📍 ${esc(p.address)}</div>`:''}
      ${p.phone?`<div class="pharmacy-detail">📞 ${esc(p.phone)}</div>`:''}
      ${p.opening_hours?`<div class="pharmacy-detail">🕐 ${esc(p.opening_hours)}</div>`:''}
      <div class="pharmacy-links">
        <a class="pharmacy-link" href="${esc(p.maps_url)}" target="_blank">📍 View on Maps</a>
        <a class="pharmacy-link" href="${esc(p.directions_url)}" target="_blank">🧭 Get Directions</a>
      </div>
    </div>`).join('');
}

/* ── Close modals on backdrop click ── */
document.querySelectorAll('.modal-backdrop').forEach(el=>{
  el.addEventListener('click',e=>{ if(e.target===el) el.classList.remove('open'); });
});

/* ── Init ── */
updateUI();
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
