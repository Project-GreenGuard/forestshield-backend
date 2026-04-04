"""
AWS Lambda function to process IoT sensor data from AWS IoT Core.

This function:
1. Receives MQTT messages from IoT Core
2. Fetches NASA FIRMS wildfire data
3. Computes risk score
4. Stores enriched data in DynamoDB
"""

import json
import os
import boto3
import requests
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any


def call_cloud_run_predict(
    temperature: float,
    humidity: float,
    lat: float,
    lng: float,
    nearest_fire_dist,
    timestamp: str
):
    url = "https://forestshield-ai-8285810750.us-central1.run.app/predict"

    payload = {
        "temperature": temperature,
        "humidity": humidity,
        "lat": lat,
        "lng": lng,
        "nearestFireDistance": nearest_fire_dist if nearest_fire_dist else 100.0,
        "timestamp": timestamp
    }

    response = requests.post(url, json=payload, timeout=5)
    response.raise_for_status()

    data = response.json()

    return (
        data["risk_score"],
        data["risk_level"],
        data["spread_rate"],
        data.get("risk_factors", []),
        data.get("recommended_action", ""),
        data.get("explanation", ""),
    )

# Initialize AWS clients - support both local and AWS DynamoDB
dynamodb_endpoint = os.getenv('AWS_ENDPOINT_URL')
if dynamodb_endpoint:
    # Local development
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=dynamodb_endpoint,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'local'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'local'),
        region_name='us-east-1'
    )
else:
    # AWS production
    dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table('WildfireSensorData')

# NASA FIRMS API endpoint (requires NASA_MAP_KEY env var)
NASA_FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/MODIS_NRT/{bbox}/1"
_ONTARIO_BBOX = "-95.5,41.5,-74.0,56.9"


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula."""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    
    return c * r


def fetch_nasa_firms_data(country_code: str = "CAN") -> list:
    """
    Fetch active wildfire data from NASA FIRMS API.
    Returns list of fire points with lat/lng coordinates.
    """
    api_key = os.getenv("NASA_MAP_KEY")
    if not api_key:
        print("Warning: NASA_MAP_KEY not set. Skipping FIRMS fetch.")
        return []
    try:
        url = NASA_FIRMS_API.format(map_key=api_key, bbox=_ONTARIO_BBOX)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse CSV response
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return []
        
        fires = []
        headers = lines[0].split(',')
        
        for line in lines[1:]:
            values = line.split(',')
            if len(values) >= len(headers):
                fire_point = {}
                for i, header in enumerate(headers):
                    fire_point[header.strip()] = values[i].strip() if i < len(values) else ''
                fires.append(fire_point)
        
        return fires
    except Exception as e:
        print(f"Error fetching NASA FIRMS data: {e}")
        return []


def find_nearest_fire(sensor_lat: float, sensor_lng: float, fires: list) -> Dict[str, Any]:
    """Find the nearest fire to the sensor location."""
    if not fires:
        return {"distance": None, "fire_data": None}
    
    min_distance = float('inf')
    nearest_fire = None
    
    for fire in fires:
        try:
            fire_lat = float(fire.get('latitude', 0))
            fire_lng = float(fire.get('longitude', 0))
            distance = calculate_distance(sensor_lat, sensor_lng, fire_lat, fire_lng)
            
            if distance < min_distance:
                min_distance = distance
                nearest_fire = fire
        except (ValueError, KeyError):
            continue
    
    return {
        "distance": round(min_distance, 2) if nearest_fire else None,
        "fire_data": nearest_fire
    }


def calculate_risk_score(temperature: float, humidity: float, fire_distance: float = None) -> float:
    """
    Simple risk scoring algorithm.
    risk = w1*temperature + w2*(1/humidity) + w3*(1/distance_to_fire)
    """
    w1 = 0.4  # Temperature weight
    w2 = 0.3  # Humidity weight
    w3 = 0.3  # Fire proximity weight
    
    # Normalize temperature (0-50°C range)
    temp_score = min(temperature / 50.0, 1.0) * 100
    
    # Inverse humidity (lower humidity = higher risk)
    humidity_score = (1.0 - min(humidity / 100.0, 1.0)) * 100
    
    # Fire proximity score
    if fire_distance is not None and fire_distance > 0:
        # Closer fire = higher risk (inverse relationship)
        # Normalize: 0-100km range
        fire_score = max(0, (100 - min(fire_distance, 100)) / 100.0) * 100
    else:
        fire_score = 0
    
    risk_score = (w1 * temp_score) + (w2 * humidity_score) + (w3 * fire_score)
    
    return round(min(risk_score, 100.0), 2)


def estimate_spread_rate(temperature: float, humidity: float, risk_score: float) -> float:
    """
    Estimate fire spread rate (km/h) as a heuristic derived from sensor inputs.
    Higher temperature, lower humidity, and higher risk score all increase spread.
    Output is clipped to [0.5, 12.0] km/h.
    """
    temp_factor     = min(max(temperature, 0.0), 50.0) / 50.0
    humidity_factor = 1.0 - min(max(humidity, 0.0), 100.0) / 100.0
    risk_factor     = min(max(risk_score,  0.0), 100.0) / 100.0

    raw = 12.0 * (0.35 * temp_factor + 0.35 * humidity_factor + 0.30 * risk_factor)
    return round(min(max(raw, 0.5), 12.0), 2)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing IoT sensor data.
    
    AWS IoT Core sends events in this format:
    {
        "deviceId": "esp32-01",
        "temperature": 23.4,
        "humidity": 40.2,
        "lat": 43.467,
        "lng": -79.699,
        "timestamp": "2025-12-01T16:20:00Z"
    }
    
    The IoT Core rule uses: SELECT * FROM 'wildfire/sensors/+'
    which passes the message payload directly.
    """
    try:
        # Debug: Log the incoming event structure
        print(f"Received event: {json.dumps(event)}")
        
        # Parse IoT payload - IoT Core SQL SELECT * returns the message payload directly
        # But sometimes it's wrapped in additional fields
        payload = None
        
        # Try different event structures
        if 'deviceId' in event and 'temperature' in event:
            # Direct payload structure
            payload = event
        elif 'temperature' in event:
            # Payload might be at root level
            payload = event
        elif 'body' in event:
            # Wrapped in body (API Gateway style, but handle it)
            if isinstance(event['body'], str):
                payload = json.loads(event['body'])
            else:
                payload = event['body']
        else:
            # Default: use event as payload
            payload = event
        
        if not payload:
            print(f"ERROR: Could not parse payload from event: {event}")
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid event structure'})}
        
        print(f"Parsed payload: {json.dumps(payload)}")
        
        device_id = payload.get('deviceId')
        temperature = float(payload.get('temperature', 0))
        humidity = float(payload.get('humidity', 0))
        lat = float(payload.get('lat', 0))
        lng = float(payload.get('lng', 0))
        timestamp = payload.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        
        print(f"Extracted values - deviceId: {device_id}, temp: {temperature}, humidity: {humidity}, lat: {lat}, lng: {lng}, timestamp: {timestamp}")
        
        # Validate required fields
        if not device_id:
            print(f"ERROR: deviceId is missing from payload: {payload}")
            return {'statusCode': 400, 'body': json.dumps({'error': 'deviceId is required'})}
        
        print(f"Processing sensor data for device: {device_id}")
        
        # Fetch NASA FIRMS wildfire data
        print("Fetching NASA FIRMS wildfire data...")
        fires = fetch_nasa_firms_data("CAN")  # Canada for now, can be parameterized
        print(f"Found {len(fires)} active fires")
        
        # Find nearest fire
        print("Finding nearest fire...")
        fire_info = find_nearest_fire(lat, lng, fires)
        fire_distance = fire_info.get('distance')
        print(f"Nearest fire distance: {fire_distance} km")
        
        # Calculate risk score — call Cloud Run API, fall back to rule-based if unavailable
        print("Calculating risk score via Cloud Run...")
        risk_factors = []
        recommended_action = ""
        explanation = ""
        try:
            risk_score, risk_level, spread_rate_kmh, risk_factors, recommended_action, explanation = call_cloud_run_predict(
                temperature, humidity, lat, lng, fire_distance, timestamp
            )
            print(f"Cloud Run prediction: {risk_score} ({risk_level}), spread: {spread_rate_kmh} km/h")
        except Exception as cloud_run_err:
            print(f"[WARN] Cloud Run unavailable, using rule-based fallback: {cloud_run_err}")
            risk_score = calculate_risk_score(temperature, humidity, fire_distance)
            risk_level = 'HIGH' if risk_score > 60 else ('MEDIUM' if risk_score > 30 else 'LOW')
            spread_rate_kmh = estimate_spread_rate(temperature, humidity, risk_score)
            risk_factors = []
            recommended_action = "Fallback mode — monitor conditions manually."
            explanation = f"Rule-based risk estimate: {risk_level} ({risk_score}/100)."
            print(f"Rule-based risk score: {risk_score} ({risk_level})")

        print(f"Estimated spread rate: {spread_rate_kmh} km/h")

        # Prepare DynamoDB item (convert floats to Decimals for DynamoDB compatibility)
        print("Preparing DynamoDB item...")
        dynamodb_item = {
            'deviceId': device_id,
            'timestamp': timestamp,
            'temperature': Decimal(str(temperature)),
            'humidity': Decimal(str(humidity)),
            'lat': Decimal(str(lat)),
            'lng': Decimal(str(lng)),
            'riskScore': Decimal(str(risk_score)),
            'riskLevel': risk_level,
            'spreadRateKmh': Decimal(str(spread_rate_kmh)),
            'nearestFireDistance': Decimal(str(fire_distance)) if fire_distance else Decimal('-1'),
            'nearestFireData': json.dumps(fire_info.get('fire_data')) if fire_info.get('fire_data') else None,
            'riskFactors': risk_factors if risk_factors else [],
            'recommendedAction': recommended_action or '',
            'explanation': explanation or '',
            'ttl': int(datetime.utcnow().timestamp()) + (30 * 24 * 60 * 60)  # 30 days TTL
        }
        
        # Write to DynamoDB
        print(f"Writing to DynamoDB table: {table.name}")
        table.put_item(Item=dynamodb_item)
        print("Successfully wrote to DynamoDB")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'deviceId': device_id,
                'riskScore': risk_score,
                'riskLevel': risk_level,
                'spreadRateKmh': spread_rate_kmh,
                'fireDistance': fire_distance
            })
        }
        
    except Exception as e:
        print(f"Error processing sensor data: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

