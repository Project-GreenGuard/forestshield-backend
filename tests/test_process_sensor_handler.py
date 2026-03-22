"""
Integration-style tests for process_sensor_data.lambda_handler with mocks (no Docker/AWS).
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def _reload_process_sensor(monkeypatch):
    """Fresh module so boto3.resource is patched before import."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://dynamodb-local:8000")
    monkeypatch.delenv("CLOUD_RUN_PREDICT_URL", raising=False)
    monkeypatch.delenv("NASA_MAP_KEY", raising=False)
    monkeypatch.setenv("DYNAMODB_TABLE", "WildfireSensorData")
    sys.modules.pop("process_sensor_data", None)


@pytest.fixture
def mocked_table(monkeypatch):
    mock_table = MagicMock()
    fake_resource = MagicMock()
    fake_resource.Table.return_value = mock_table
    _reload_process_sensor(monkeypatch)
    with patch("boto3.resource", return_value=fake_resource):
        import process_sensor_data as psd

        yield psd, mock_table


def test_lambda_handler_rule_based_writes_risk_level_and_spread(mocked_table):
    psd, mock_table = mocked_table

    fake_resp = MagicMock()
    fake_resp.text = "latitude,longitude\n43.0,-79.0\n"
    fake_resp.raise_for_status = MagicMock()

    with patch.object(psd.requests, "get", return_value=fake_resp):
        event = {
            "deviceId": "test-dev",
            "temperature": 40.0,
            "humidity": 15.0,
            "lat": 43.5,
            "lng": -79.5,
            "timestamp": "2026-03-22T12:00:00Z",
        }
        out = psd.lambda_handler(event, None)

    assert out["statusCode"] == 200
    body = json.loads(out["body"])
    assert body["success"] is True
    assert "riskLevel" in body
    assert "spreadRateKmh" in body

    mock_table.put_item.assert_called_once()
    item = mock_table.put_item.call_args.kwargs["Item"]
    assert item["riskLevel"] in ("LOW", "MEDIUM", "HIGH")
    assert "spreadRateKmh" in item


def test_lambda_handler_uses_cloud_run_when_configured(mocked_table):
    psd, mock_table = mocked_table

    fake_get = MagicMock()
    fake_get.text = "latitude,longitude\n"
    fake_get.raise_for_status = MagicMock()

    fake_post = MagicMock()
    fake_post.json.return_value = {
        "risk_score": 72.5,
        "risk_level": "HIGH",
        "spread_rate": 8.2,
    }
    fake_post.raise_for_status = MagicMock()

    with patch.object(psd, "CLOUD_RUN_PREDICT_URL", "https://example.run.app/predict"):
        with patch.object(psd.requests, "get", return_value=fake_get):
            with patch.object(psd.requests, "post", return_value=fake_post) as mock_post_fn:
                event = {
                    "deviceId": "cr-1",
                    "temperature": 30.0,
                    "humidity": 40.0,
                    "lat": 44.0,
                    "lng": -80.0,
                    "timestamp": "2026-03-22T12:00:00Z",
                }
                out = psd.lambda_handler(event, None)

    assert out["statusCode"] == 200
    mock_post_fn.assert_called_once()
    item = mock_table.put_item.call_args.kwargs["Item"]
    assert str(item["riskLevel"]) == "HIGH"
    assert float(item["riskScore"]) == pytest.approx(72.5)
    assert float(item["spreadRateKmh"]) == pytest.approx(8.2)
