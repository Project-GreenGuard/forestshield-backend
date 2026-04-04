"""
AWS Lambda function to process IoT sensor data from AWS IoT Core.

This function:
1. Receives MQTT messages from IoT Core
2. Fetches NASA FIRMS wildfire data
3. Computes risk (optional GCP Cloud Run ML inference, else rule-based fallback)
4. Stores enriched data in DynamoDB (riskLevel, spreadRateKmh, etc.)
"""

import json
import boto3
import os
import requests
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple

# Initialize AWS clients - support both local and AWS DynamoDB
dynamodb_endpoint = os.getenv("AWS_ENDPOINT_URL")
if dynamodb_endpoint:
    dynamodb = boto3.resource(
        "dynamodb",
        endpoint_url=dynamodb_endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "local"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "local"),
        region_name="us-east-1",
    )
else:
    dynamodb = boto3.resource("dynamodb")

table = dynamodb.Table(os.getenv("DYNAMODB_TABLE", "WildfireSensorData"))

# NASA FIRMS: area API needs MAP key; country CSV works without (broader bbox)
NASA_FIRMS_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_NRT/{bbox}/1"
NASA_FIRMS_COUNTRY = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/{}/MODIS_NRT/1"
_ONTARIO_BBOX = "-95.5,41.5,-74.0,56.9"

# Full URL to POST JSON (e.g. https://your-service.run.app/predict). If unset, rule-based only.
CLOUD_RUN_PREDICT_URL = os.getenv("CLOUD_RUN_PREDICT_URL", "").strip()


def _cloud_run_timeout_sec() -> float:
    """Read on each call so Lambda env updates apply without cold start."""
    return float(os.getenv("CLOUD_RUN_TIMEOUT_SEC", "8"))


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371


def fetch_nasa_firms_data(country_code: str = "CAN") -> list:
    """
    Prefer NASA area CSV when NASA_MAP_KEY is set (Ontario bbox); else country CSV.
    """
    api_key = os.getenv("NASA_MAP_KEY", "").strip()
    if api_key:
        try:
            url = NASA_FIRMS_AREA.format(map_key=api_key, bbox=_ONTARIO_BBOX)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return _parse_firms_csv(response.text)
        except Exception as e:
            print(f"[WARN] NASA area FIRMS failed ({e}), falling back to country CSV")

    try:
        url = NASA_FIRMS_COUNTRY.format(country_code)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return _parse_firms_csv(response.text)
    except Exception as e:
        print(f"Error fetching NASA FIRMS data: {e}")
        return []


def _parse_firms_csv(text: str) -> list:
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    fires = []
    headers = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        if len(values) >= len(headers):
            fire_point = {}
            for i, header in enumerate(headers):
                fire_point[header.strip()] = values[i].strip() if i < len(values) else ""
            fires.append(fire_point)
    return fires


def find_nearest_fire(sensor_lat: float, sensor_lng: float, fires: list) -> Dict[str, Any]:
    if not fires:
        return {"distance": None, "fire_data": None}

    min_distance = float("inf")
    nearest_fire = None

    for fire in fires:
        try:
            fire_lat = float(fire.get("latitude", 0))
            fire_lng = float(fire.get("longitude", 0))
            distance = calculate_distance(sensor_lat, sensor_lng, fire_lat, fire_lng)
            if distance < min_distance:
                min_distance = distance
                nearest_fire = fire
        except (ValueError, KeyError):
            continue

    return {
        "distance": round(min_distance, 2) if nearest_fire else None,
        "fire_data": nearest_fire,
    }


def calculate_risk_score(temperature: float, humidity: float, fire_distance: Optional[float] = None) -> float:
    w1, w2, w3 = 0.4, 0.3, 0.3
    temp_score = min(temperature / 50.0, 1.0) * 100
    humidity_score = (1.0 - min(humidity / 100.0, 1.0)) * 100
    if fire_distance is not None and fire_distance > 0:
        fire_score = max(0, (100 - min(fire_distance, 100)) / 100.0) * 100
    else:
        fire_score = 0
    risk_score = (w1 * temp_score) + (w2 * humidity_score) + (w3 * fire_score)
    return round(min(risk_score, 100.0), 2)


def risk_level_from_score(score: float) -> str:
    if score > 60:
        return "HIGH"
    if score > 30:
        return "MEDIUM"
    return "LOW"


def estimate_spread_rate_kmh(temperature: float, humidity: float, risk_score: float) -> float:
    """Heuristic when Cloud Run is unavailable (aligned with PR #9 scale ~0.5–12 km/h)."""
    temp_factor = min(max(temperature, 0.0), 50.0) / 50.0
    humidity_factor = 1.0 - min(max(humidity, 0.0), 100.0) / 100.0
    risk_factor = min(max(risk_score, 0.0), 100.0) / 100.0
    raw = 12.0 * (0.35 * temp_factor + 0.35 * humidity_factor + 0.30 * risk_factor)
    return round(min(max(raw, 0.5), 12.0), 2)


def generate_ai_insights(temperature, humidity, fire_distance, risk_score):
    """
    Generate AI-style decision-support insights based on sensor features
    and predicted risk score. Mirrors forestshield-ai/inference/predict.py.
    """
    reasons = []

    if temperature >= 35:
        reasons.append("high temperature")
    elif temperature >= 28:
        reasons.append("elevated temperature")

    if humidity <= 30:
        reasons.append("very low humidity")
    elif humidity <= 60:
        reasons.append("moderate humidity")
    else:
        reasons.append("higher humidity conditions")

    if fire_distance is not None and fire_distance <= 10:
        reasons.append("active fire detected nearby")
    elif fire_distance is not None and fire_distance <= 50:
        reasons.append("fire activity within operational range")

    if risk_score >= 75:
        reasons.append("strong model confidence in severe wildfire conditions")
    elif risk_score >= 45 and len(reasons) <= 2:
        reasons.append("moderate environmental risk conditions")

    if risk_score >= 61:
        action = "Dispatch emergency responders, monitor evacuation zones, and issue high-priority alerts."
    elif risk_score >= 31:
        action = "Increase monitoring, prepare response teams, and watch for changing fire conditions."
    else:
        action = "Maintain routine monitoring and continue collecting sensor updates."

    explanation = f"Predicted wildfire risk is driven by {', '.join(reasons)}." if reasons else "Insufficient data for detailed analysis."

    return reasons, action, explanation


def call_cloud_run_predict(
    temperature: float,
    humidity: float,
    lat: float,
    lng: float,
    nearest_fire_dist: Optional[float],
    timestamp: str,
) -> Tuple[float, str, float, list, str, str]:
    """
    POST to Cloud Run predict endpoint. Returns:
    (risk_score, risk_level, spread_rate, risk_factors, recommended_action, explanation)
    """
    if not CLOUD_RUN_PREDICT_URL:
        raise RuntimeError("CLOUD_RUN_PREDICT_URL not set")

    nearest_payload = 100.0
    if nearest_fire_dist is not None and nearest_fire_dist >= 0:
        nearest_payload = float(nearest_fire_dist)

    payload = {
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearestFireDistance": nearest_payload,
        "timestamp": timestamp,
    }

    response = requests.post(
        CLOUD_RUN_PREDICT_URL,
        json=payload,
        timeout=_cloud_run_timeout_sec(),
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    score = float(data.get("risk_score", data.get("riskScore")))
    level_raw = data.get("risk_level", data.get("riskLevel", ""))
    spread = float(data.get("spread_rate", data.get("spreadRateKmh", 0)))

    level = str(level_raw).strip().upper() if level_raw else ""
    if level not in ("LOW", "MEDIUM", "HIGH"):
        level = risk_level_from_score(score)

    # Extract new AI insight fields returned by predict.py
    risk_factors = data.get("risk_factors", data.get("riskFactors", []))
    recommended_action = data.get("recommended_action", data.get("recommendedAction", ""))
    explanation = data.get("explanation", "")

    return score, level, spread, risk_factors, recommended_action, explanation


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        print(f"Received event: {json.dumps(event)}")

        payload = None
        if "deviceId" in event and "temperature" in event:
            payload = event
        elif "temperature" in event:
            payload = event
        elif "body" in event:
            if isinstance(event["body"], str):
                payload = json.loads(event["body"])
            else:
                payload = event["body"]
        else:
            payload = event

        if not payload:
            print(f"ERROR: Could not parse payload from event: {event}")
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid event structure"})}

        print(f"Parsed payload: {json.dumps(payload)}")

        device_id = payload.get("deviceId")
        temperature = float(payload.get("temperature", 0))
        humidity = float(payload.get("humidity", 0))
        lat = float(payload.get("lat", 0))
        lng = float(payload.get("lng", 0))
        timestamp = payload.get("timestamp", datetime.utcnow().isoformat() + "Z")

        if not device_id:
            print(f"ERROR: deviceId is missing from payload: {payload}")
            return {"statusCode": 400, "body": json.dumps({"error": "deviceId is required"})}

        print(f"Processing sensor data for device: {device_id}")

        print("Fetching NASA FIRMS wildfire data...")
        fires = fetch_nasa_firms_data("CAN")
        print(f"Found {len(fires)} active fire rows")

        fire_info = find_nearest_fire(lat, lng, fires)
        fire_distance = fire_info.get("distance")
        print(f"Nearest fire distance: {fire_distance} km")

        risk_score: float
        risk_level: str
        spread_rate_kmh: float

        risk_factors = []
        recommended_action = ""
        explanation = ""
        used_cloud_run = False

        if CLOUD_RUN_PREDICT_URL:
            print(f"Calling Cloud Run: {CLOUD_RUN_PREDICT_URL}")
            try:
                risk_score, risk_level, spread_rate_kmh, risk_factors, recommended_action, explanation = call_cloud_run_predict(
                    temperature, humidity, lat, lng, fire_distance, timestamp
                )
                used_cloud_run = True
                print(f"Cloud Run: score={risk_score}, level={risk_level}, spread={spread_rate_kmh}")
                print(f"Cloud Run AI: factors={risk_factors}, action={recommended_action[:50]}...")
            except Exception as err:
                print(f"[WARN] Cloud Run unavailable, rule-based fallback: {err}")

        if not used_cloud_run:
            if not CLOUD_RUN_PREDICT_URL:
                print("CLOUD_RUN_PREDICT_URL unset; using rule-based scoring only")
            risk_score = calculate_risk_score(temperature, humidity, fire_distance)
            risk_level = risk_level_from_score(risk_score)
            spread_rate_kmh = estimate_spread_rate_kmh(temperature, humidity, risk_score)
            risk_factors, recommended_action, explanation = generate_ai_insights(
                temperature, humidity, fire_distance, risk_score
            )

        print(f"AI Insights: factors={risk_factors}, action={recommended_action[:50]}...")

        dynamodb_item = {
            "deviceId": device_id,
            "timestamp": timestamp,
            "temperature": Decimal(str(temperature)),
            "humidity": Decimal(str(humidity)),
            "lat": Decimal(str(lat)),
            "lng": Decimal(str(lng)),
            "riskScore": Decimal(str(risk_score)),
            "riskLevel": risk_level,
            "spreadRateKmh": Decimal(str(spread_rate_kmh)),
            "nearestFireDistance": Decimal(str(fire_distance)) if fire_distance else Decimal("-1"),
            "nearestFireData": json.dumps(fire_info.get("fire_data")) if fire_info.get("fire_data") else None,
            "riskFactors": risk_factors,
            "recommendedAction": recommended_action,
            "explanation": explanation,
            "ttl": int(datetime.utcnow().timestamp()) + (30 * 24 * 60 * 60),
        }

        print(f"Writing to DynamoDB table: {table.name}")
        table.put_item(Item=dynamodb_item)
        print("Successfully wrote to DynamoDB")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "success": True,
                    "deviceId": device_id,
                    "riskScore": risk_score,
                    "riskLevel": risk_level,
                    "spreadRateKmh": spread_rate_kmh,
                    "fireDistance": fire_distance,
                    "riskFactors": risk_factors,
                    "recommendedAction": recommended_action,
                    "explanation": explanation,
                }
            ),
        }

    except Exception as e:
        print(f"Error processing sensor data: {e}")
        import traceback

        traceback.print_exc()
        return {"statusCode": 500, "body": json.dumps({"success": False, "error": str(e)})}
