"""Unit tests for dashboard sensor enrichment (no DynamoDB)."""

from decimal import Decimal

import pytest

from sensor_enrichment import (
    enrich_sensor_record,
    merge_sensor_public_fields,
    risk_level_from_score,
    spread_rate_kmh_estimate,
)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "LOW"),
        (30, "LOW"),
        (31, "MEDIUM"),
        (60, "MEDIUM"),
        (61, "HIGH"),
        (100, "HIGH"),
        (None, "LOW"),
        (Decimal("45"), "MEDIUM"),
    ],
)
def test_risk_level_from_score(score, expected):
    assert risk_level_from_score(score) == expected


def test_spread_rate_monotonic_with_risk():
    low = spread_rate_kmh_estimate(10, None)
    high = spread_rate_kmh_estimate(90, None)
    assert high > low


def test_spread_rate_includes_proximity_when_fire_known():
    without = spread_rate_kmh_estimate(50, 100)
    closer = spread_rate_kmh_estimate(50, 5)
    assert closer >= without


def test_spread_rate_ignores_negative_distance_sentinel():
    """nearestFireDistance -1 means unknown in list sensors."""
    assert spread_rate_kmh_estimate(80, -1) == spread_rate_kmh_estimate(80, None)


def test_enrich_sensor_record_adds_fields():
    row = {"deviceId": "esp-1", "riskScore": 55, "nearestFireDistance": 20.0}
    out = enrich_sensor_record(row)
    assert out["riskLevel"] == "MEDIUM"
    assert isinstance(out["spreadRateKmh"], float)
    assert out["deviceId"] == "esp-1"


def test_merge_prefers_persisted_fields():
    row = {
        "deviceId": "esp-1",
        "riskScore": 55.0,
        "nearestFireDistance": 100.0,
        "riskLevel": "HIGH",
        "spreadRateKmh": 7.5,
    }
    out = merge_sensor_public_fields(row)
    assert out["riskLevel"] == "HIGH"
    assert out["spreadRateKmh"] == 7.5


def test_merge_fills_missing_from_score():
    row = {"deviceId": "esp-1", "riskScore": 25.0, "nearestFireDistance": -1.0}
    out = merge_sensor_public_fields(row)
    assert out["riskLevel"] == "LOW"
    assert isinstance(out["spreadRateKmh"], float)
