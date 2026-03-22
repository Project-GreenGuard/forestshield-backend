"""
Local Flask server for development.
Simulates API Gateway + Lambda for local frontend development.
"""

import json
import os
import sys

# Processing Lambda code lives alongside api-gateway-lambda in the repo
_lambda_dir = os.path.join(os.path.dirname(__file__), "lambda-processing")
if os.path.isdir(_lambda_dir):
    sys.path.insert(0, _lambda_dir)

from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3
from nasa_firms_service import get_nasa_fires

from api_handler import get_all_sensors, get_sensor_by_id, get_risk_map_data

try:
    from process_sensor_data import lambda_handler as process_sensor_lambda
except ImportError:
    process_sensor_lambda = None

app = Flask(__name__)
CORS(app)

dynamodb_endpoint = os.getenv("AWS_ENDPOINT_URL", "http://dynamodb:8000")
dynamodb = boto3.resource(
    "dynamodb",
    endpoint_url=dynamodb_endpoint,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "local"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "local"),
    region_name="us-east-1",
)

_table_name = os.getenv("DYNAMODB_TABLE", "WildfireSensorData")


def ensure_table_exists():
    """Create WildfireSensorData table if missing (local dev only)."""
    client = dynamodb.meta.client
    try:
        client.describe_table(TableName=_table_name)
    except client.exceptions.ResourceNotFoundException:
        print(f"Creating DynamoDB table {_table_name}...")
        dynamodb.create_table(
            TableName=_table_name,
            KeySchema=[
                {"AttributeName": "deviceId", "KeyType": "HASH"},
                {"AttributeName": "timestamp", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "deviceId", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print("Table created.")


ensure_table_exists()


@app.route("/api/sensors", methods=["GET"])
def api_sensors():
    sensors = get_all_sensors()
    return jsonify(sensors)


@app.route("/api/nasa-fires", methods=["GET"])
def nasa_fires():
    fires = get_nasa_fires()
    return jsonify({"source": "NASA FIRMS", "count": len(fires), "fires": fires})


@app.route("/api/sensor/<device_id>", methods=["GET"])
def api_sensor(device_id):
    sensor = get_sensor_by_id(device_id)
    if sensor:
        return jsonify(sensor)
    return jsonify({"error": "Sensor not found"}), 404


@app.route("/api/risk-map", methods=["GET"])
def api_risk_map():
    return jsonify(get_risk_map_data())


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """
    Simulate IoT Core -> processing Lambda (writes riskLevel, spreadRateKmh via Cloud Run or rule-based).
    """
    if process_sensor_lambda is None:
        return jsonify({"error": "process_sensor_data not importable"}), 500
    payload = request.get_json(force=True)
    result = process_sensor_lambda(payload, None)
    body = result.get("body", result)
    if isinstance(body, str):
        body = json.loads(body)
    return jsonify(body), result.get("statusCode", 200)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "forestshield-api"})


if __name__ == "__main__":
    print("Local API server starting...")
    print("API base: http://localhost:5000/api")
    print("  GET  /api/sensors")
    print("  GET  /api/sensor/<id>")
    print("  GET  /api/risk-map")
    print("  POST /api/ingest  (simulate IoT -> processing Lambda)")
    app.run(host="0.0.0.0", port=5000, debug=True)
