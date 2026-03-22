"""
Derived sensor fields for API responses (dashboard-only heuristics).

Thresholds match the frontend map legend: ≤30 LOW, ≤60 MEDIUM, else HIGH.
"""

from decimal import Decimal
from typing import Optional


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def risk_level_from_score(risk_score) -> str:
    """LOW / MEDIUM / HIGH from numeric risk score (0–100)."""
    s = _to_float(risk_score)
    if s is None:
        return "LOW"
    if s <= 30:
        return "LOW"
    if s <= 60:
        return "MEDIUM"
    return "HIGH"


def spread_rate_kmh_estimate(risk_score, nearest_fire_distance) -> float:
    """
    Dashboard-only spread rate (km/h), not an operational fire-behavior model.
    Combines overall risk score with proximity when nearest_fire_distance is known (≥ 0).
    """
    rs = _to_float(risk_score) or 0.0
    base = (rs / 100.0) * 16.0
    d = _to_float(nearest_fire_distance)
    if d is not None and d >= 0:
        proximity = max(0.0, (100.0 - min(d, 100.0)) / 100.0)
        base += proximity * 9.0
    return round(min(base, 28.0), 2)


def enrich_sensor_record(record: dict) -> dict:
    """Return a shallow copy with riskLevel and spreadRateKmh always derived from score."""
    if not record:
        return record
    out = dict(record)
    rs = out.get("riskScore")
    nf = out.get("nearestFireDistance")
    out["riskLevel"] = risk_level_from_score(rs)
    out["spreadRateKmh"] = spread_rate_kmh_estimate(rs, nf)
    return out


def _normalize_persisted_level(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    u = str(raw).strip().upper()
    if u in ("LOW", "MEDIUM", "HIGH"):
        return u
    return None


def merge_sensor_public_fields(record: dict) -> dict:
    """
    Prefer riskLevel / spreadRateKmh from DynamoDB (written by processing Lambda + Cloud Run).
    If missing (old rows), derive — same thresholds as the UI legend.
    """
    if not record:
        return record
    out = dict(record)
    rs = out.get("riskScore")
    nf = out.get("nearestFireDistance")
    persisted_level = _normalize_persisted_level(out.get("riskLevel"))
    if persisted_level:
        out["riskLevel"] = persisted_level
    else:
        out["riskLevel"] = risk_level_from_score(rs)
    if out.get("spreadRateKmh") is None:
        out["spreadRateKmh"] = spread_rate_kmh_estimate(rs, nf)
    else:
        v = _to_float(out.get("spreadRateKmh"))
        out["spreadRateKmh"] = round(v, 2) if v is not None else spread_rate_kmh_estimate(rs, nf)
    return out
