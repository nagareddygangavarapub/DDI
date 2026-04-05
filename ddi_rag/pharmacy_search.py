"""
pharmacy_search.py — Find nearby pharmacies using OpenStreetMap Overpass API.

Free, no API key required.
Returns pharmacies within a configurable radius sorted by distance.

Usage:
    from pharmacy_search import find_pharmacies
    results = find_pharmacies(lat=17.385, lon=78.486, radius_m=5000)
"""

import logging
import math
import requests
from typing import Dict, List

log = logging.getLogger("ddi.pharmacy")

OVERPASS_URL    = "https://overpass-api.de/api/interpreter"
DEFAULT_RADIUS  = 5000   # metres (5 km)
REQUEST_TIMEOUT = 20     # seconds


# ── Distance helper ───────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two (lat, lon) pairs."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


# ── Main search function ──────────────────────────────────────────────────────

def find_pharmacies(
    lat: float,
    lon: float,
    radius_m: int = DEFAULT_RADIUS,
    drug_name: str = "",
) -> List[Dict]:
    """
    Query OpenStreetMap Overpass API for pharmacies near (lat, lon).

    Args:
        lat       : latitude  (e.g. 17.385 for Hyderabad)
        lon       : longitude (e.g. 78.486 for Hyderabad)
        radius_m  : search radius in metres (default 5 km)
        drug_name : optional drug name — included in result metadata only

    Returns:
        List of dicts sorted by distance (nearest first):
            name, address, distance_km, phone, opening_hours,
            lat, lon, maps_url, osm_id
    """
    # Overpass QL — search nodes AND ways tagged as pharmacy
    overpass_query = f"""
[out:json][timeout:15];
(
  node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
  way["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
);
out center body;
"""

    try:
        log.info("Querying Overpass API — centre=(%.4f,%.4f) radius=%dm", lat, lon, radius_m)
        resp = requests.post(
            OVERPASS_URL,
            data    = {"data": overpass_query},
            timeout = REQUEST_TIMEOUT,
            headers = {"Accept": "application/json"},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        log.info("Overpass returned %d elements", len(elements))
    except requests.Timeout:
        log.error("Overpass API timed out")
        return []
    except Exception:
        log.exception("Overpass API request failed")
        return []

    results: List[Dict] = []

    for el in elements:
        tags = el.get("tags", {})

        # Resolve coordinates — nodes have lat/lon directly; ways have a center
        if el["type"] == "node":
            plat, plon = el["lat"], el["lon"]
        else:
            centre = el.get("center", {})
            plat = centre.get("lat", lat)
            plon = centre.get("lon", lon)

        # Build address string from OSM addr tags
        addr_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:suburb", ""),
            tags.get("addr:city", ""),
        ]
        address = ", ".join(p for p in addr_parts if p) or "Address not listed"

        # Format opening hours nicely if present
        hours = tags.get("opening_hours", "")

        distance_km = _haversine(lat, lon, plat, plon)

        results.append({
            "name"          : tags.get("name", "Pharmacy"),
            "address"       : address,
            "distance_km"   : round(distance_km, 2),
            "distance_label": _format_distance(distance_km),
            "phone"         : tags.get("phone") or tags.get("contact:phone") or "",
            "opening_hours" : hours,
            "lat"           : plat,
            "lon"           : plon,
            "maps_url"      : f"https://www.google.com/maps/search/?api=1&query={plat},{plon}",
            "directions_url": f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}",
            "osm_id"        : el.get("id"),
            "drug_query"    : drug_name,
        })

    # Sort nearest first
    return sorted(results, key=lambda x: x["distance_km"])


def _format_distance(km: float) -> str:
    if km < 1.0:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"
