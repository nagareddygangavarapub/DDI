"""
mcp_server.py — DrugSafe AI MCP Server

Exposes DrugSafe AI capabilities as MCP tools so any MCP-compatible client
(Claude Desktop, Cursor, etc.) can query drug interactions and find pharmacies.

Tools:
    query_drug_interactions  — RAG pipeline: retrieve FDA evidence + Groq answer
    find_nearby_pharmacies   — OpenStreetMap pharmacy search by GPS coordinates
    list_drug_warnings       — Quick interaction summary for a drug pair

Run (stdio transport — used by Claude Desktop):
    python mcp_server.py

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "drugsafe": {
          "command": "python",
          "args": ["C:/Users/C V REDDY/Downloads/ddi_rag/ddi_rag/mcp_server.py"]
        }
      }
    }
"""

import json
import logging
import sys
from pathlib import Path

# Allow imports from the ddi_rag package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from config import GROQ_API_KEY, GROQ_MODEL
from pharmacy_search import find_pharmacies

log = logging.getLogger("ddi.mcp")
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# ── Create MCP server ─────────────────────────────────────────────────────────

mcp = FastMCP(
    name        = "DrugSafe AI",
    description = (
        "FDA-powered drug interaction checker. "
        "Query drug-drug interactions using ChromaDB RAG + Llama 3.1 via Groq. "
        "Find nearby pharmacies using OpenStreetMap."
    ),
)

# Lazy-load RAG pipeline (heavy models) only when a tool is first called
_rag_ready = False

def _ensure_rag():
    global _rag_ready
    if not _rag_ready:
        log.info("Loading RAG models for MCP server...")
        from rag_pipeline import load_models
        load_models()
        _rag_ready = True


# ── Tool 1: Query drug interactions ──────────────────────────────────────────

@mcp.tool()
def query_drug_interactions(
    drug_name: str,
    history_context: str = "",
) -> str:
    """
    Query FDA label data and DDI pair database for drug interaction information.

    Uses ChromaDB vector search (930k+ vectors) + Llama 3.1 via Groq to
    provide clinically relevant interaction warnings and contraindications.

    Args:
        drug_name       : Generic drug name to query (e.g. "warfarin", "aspirin")
        history_context : Optional — comma-separated list of patient's current
                          medications (e.g. "warfarin 5mg, lisinopril 10mg")
                          Used to personalise interaction warnings.

    Returns:
        Clinical summary of drug interactions, warnings, and contraindications.
    """
    _ensure_rag()
    from rag_pipeline import answer_ddi

    if not drug_name or not drug_name.strip():
        return "Error: drug_name is required."

    drug_clean = drug_name.strip().lower()
    log.info("MCP tool call: query_drug_interactions('%s')", drug_clean)

    result = answer_ddi(
        drug_name       = drug_clean,
        section         = "drug_interactions",
        top_k           = 5,
        history_context = history_context.strip(),
    )

    answer  = result.get("answer", "No information found.")
    sources = result.get("sources", [])

    # Format sources as a readable list
    src_lines = []
    for i, s in enumerate(sources[:5], 1):
        section  = (s.get("section") or "").replace("_", " ").title()
        score    = float(s.get("score", 0))
        drug     = s.get("generic_name", "")
        src_lines.append(f"  [{i}] {section} | {drug} | similarity={score:.3f}")

    output = f"## Drug Interaction Summary: {drug_name.title()}\n\n{answer}"
    if src_lines:
        output += "\n\n### FDA Evidence Sources\n" + "\n".join(src_lines)

    return output


# ── Tool 2: List drug warnings for a pair ────────────────────────────────────

@mcp.tool()
def list_drug_warnings(
    drug_1: str,
    drug_2: str,
) -> str:
    """
    Check for known interactions between two specific drugs.

    Searches the DDI pairs database (187k+ pairwise interactions) and FDA
    label data for any documented warnings between drug_1 and drug_2.

    Args:
        drug_1 : First drug name  (e.g. "warfarin")
        drug_2 : Second drug name (e.g. "aspirin")

    Returns:
        Known interaction descriptions and severity information.
    """
    _ensure_rag()
    from rag_pipeline import retrieve_chunks

    d1 = drug_1.strip().lower()
    d2 = drug_2.strip().lower()
    log.info("MCP tool call: list_drug_warnings('%s', '%s')", d1, d2)

    query = f"{d1} and {d2} drug interaction warning"
    chunks = retrieve_chunks(query=query, top_k=5)

    if chunks.empty:
        return f"No documented interactions found between {drug_1} and {drug_2} in the database."

    lines = [f"## Interaction Check: {drug_1.title()} + {drug_2.title()}\n"]
    for i, (_, row) in enumerate(chunks.iterrows(), 1):
        section = (row.get("section") or "").replace("_", " ").title()
        text    = (row.get("text") or "")[:400]
        score   = float(row.get("score", 0))
        lines.append(f"**Source {i}** ({section}, relevance={score:.3f}):\n{text}\n")

    return "\n".join(lines)


# ── Tool 3: Find nearby pharmacies ───────────────────────────────────────────

@mcp.tool()
def find_nearby_pharmacies(
    latitude: float,
    longitude: float,
    drug_name: str = "",
    radius_km: float = 5.0,
) -> str:
    """
    Find pharmacies near a GPS location using OpenStreetMap (no API key needed).

    Args:
        latitude  : GPS latitude  (e.g. 17.385 for Hyderabad)
        longitude : GPS longitude (e.g. 78.486 for Hyderabad)
        drug_name : Optional drug name — shown in results for context
        radius_km : Search radius in km (default 5 km, max 20 km)

    Returns:
        Formatted list of nearby pharmacies with addresses, distances, and
        Google Maps links sorted by proximity.
    """
    radius_km = min(float(radius_km), 20.0)
    radius_m  = int(radius_km * 1000)

    log.info(
        "MCP tool call: find_nearby_pharmacies(%.4f, %.4f, radius=%dm)",
        latitude, longitude, radius_m,
    )

    pharmacies = find_pharmacies(
        lat       = latitude,
        lon       = longitude,
        radius_m  = radius_m,
        drug_name = drug_name,
    )

    if not pharmacies:
        return (
            f"No pharmacies found within {radius_km:.0f} km of "
            f"({latitude:.4f}, {longitude:.4f}). "
            "Try increasing the radius."
        )

    drug_line = f" — searching for: **{drug_name}**" if drug_name else ""
    lines = [
        f"## Nearby Pharmacies{drug_line}",
        f"Found {len(pharmacies)} pharmacies within {radius_km:.0f} km\n",
    ]

    for i, p in enumerate(pharmacies[:10], 1):
        lines.append(f"### {i}. {p['name']}  ({p['distance_label']})")
        lines.append(f"📍 {p['address']}")
        if p["phone"]:
            lines.append(f"📞 {p['phone']}")
        if p["opening_hours"]:
            lines.append(f"🕐 {p['opening_hours']}")
        lines.append(f"🗺  [Open in Google Maps]({p['maps_url']})")
        lines.append(f"🧭 [Get Directions]({p['directions_url']})")
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting DrugSafe AI MCP server (stdio transport)...")
    mcp.run(transport="stdio")
