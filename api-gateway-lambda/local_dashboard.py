"""
Local Flask server for development.
Simulates API Gateway + Lambda for local frontend development.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import boto3
from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key
from nasa_firms_service import get_nasa_fires
import sys
from pathlib import Path


load_dotenv()  

# Import API handler functions
from api_handler import (
    get_all_sensors, 
    get_sensor_by_id, 
    get_risk_map_data,
    get_detailed_risk,
    get_model_health
)

app = Flask(__name__)
CORS(app)

# Configure DynamoDB client for local development
dynamodb_endpoint = os.getenv('AWS_ENDPOINT_URL', 'http://dynamodb:8000')
dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url=dynamodb_endpoint,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'local'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'local'),
    region_name='us-east-1'
)

# The api_handler will use the environment variables we set above

@app.route('/api/sensors', methods=['GET'])
def api_sensors():
    """Get all sensors endpoint"""
    sensors = get_all_sensors()
    return jsonify(sensors)

@app.route('/api/nasa-fires', methods=['GET'])
def nasa_fires():
    fires = get_nasa_fires()
    return jsonify({
        "source": "NASA FIRMS",
        "count": len(fires),
        "fires": fires
    })

@app.route('/api/sensor/<device_id>', methods=['GET'])
def api_sensor(device_id):
    """Get sensor by ID endpoint"""
    sensor = get_sensor_by_id(device_id)
    if sensor:
        return jsonify(sensor)
    else:
        return jsonify({'error': 'Sensor not found'}), 404

@app.route('/api/risk-map', methods=['GET'])
def api_risk_map():
    """Get risk map data endpoint"""
    map_data = get_risk_map_data()
    return jsonify(map_data)

@app.route('/api/risk-detailed', methods=['POST'])
def api_risk_detailed():
    """Get detailed risk scoring endpoint (PBI-7)"""
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'No payload provided'}), 400
        
        detailed = get_detailed_risk(payload)
        return jsonify(detailed)
    except Exception as e:
        print(f"Error in /api/risk-detailed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-health', methods=['GET'])
def api_model_health():
    """Get model health status endpoint (PBI-7)"""
    health = get_model_health()
    return jsonify(health)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'forestshield-api'})

if __name__ == '__main__':
    print("Local API server starting...")
    print("API available at http://localhost:5000/api")
    print("Endpoints:")
    print("  GET /api/sensors")
    print("  GET /api/sensor/<id>")
    print("  GET /api/risk-map")
    print("  POST /api/risk-detailed (PBI-7)")
    print("  GET /api/model-health (PBI-7)")
    app.run(host='0.0.0.0', port=5000, debug=True)