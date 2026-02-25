"""
AWS Lambda function to process IoT sensor data from AWS IoT Core.

This function:
1. Receives MQTT messages from IoT Core
2. Fetches NASA FIRMS wildfire data
3. Computes ML-based risk score
4. Stores enriched data in DynamoDB
"""

import json
import boto3
import os
import requests
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
import tempfile

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

# NASA FIRMS API endpoint
NASA_FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/{}/MODIS_NRT/1"

# ML Model - will be loaded from S3 in production
ml_model = None
model_loaded = False

# Feature columns expected by the ML model
FEATURE_COLUMNS = [
    'temperature', 'humidity', 'wind_speed', 
    'wind_direction', 'pressure', 'vegetation_index'
]

def load_ml_model():
    """Load the trained ML model from S3 or local file"""
    global ml_model, model_loaded
    
    try:
        # In production, you'd load from S3
        # s3 = boto3.client('s3')
        # s3.download_file('your-bucket', 'models/ml_model.joblib', '/tmp/ml_model.joblib')
        # ml_model = joblib.load('/tmp/ml_model.joblib')
        
        # For local development, load from current directory
        model_path = os.path.join(os.path.dirname(__file__), 'ml_model.joblib')
        if os.path.exists(model_path):
            ml_model = joblib.load(model_path)
            model_loaded = True
            print(f"✅ ML Model loaded from {model_path}")
        else:
            print("⚠️ No ML model found, using fallback calculation")
            model_loaded = False
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        model_loaded = False

def calculate_vegetation_index(humidity: float, soil_moisture: float = 30) -> float:
    """Calculate vegetation moisture index (matches training data)"""
    return (humidity * 0.3 + soil_moisture * 0.7) / 100

def calculate_risk_score_ml(temperature: float, humidity: float, wind_speed: float = 10, 
                           wind_direction: float = 0, pressure: float = 1013) -> float:
    """Use ML model to predict risk score"""
    global ml_model, model_loaded
    
    # Load model if not loaded
    if not model_loaded:
        load_ml_model()
    
    # If model loaded, use it
    if model_loaded and ml_model:
        try:
            # Calculate vegetation index
            veg_index = calculate_vegetation_index(humidity)
            
            # Create feature array for prediction
            features = pd.DataFrame([[
                temperature, humidity, wind_speed, wind_direction, pressure, veg_index
            ]], columns=FEATURE_COLUMNS)
            
            # Make prediction (returns 0-1 scale)
            risk_score = float(ml_model.predict(features)[0])
            risk_score = max(0, min(1, risk_score)) * 100  # Convert to 0-100 scale
            
            print(f"✅ ML prediction: {risk_score:.1f}/100")
            return round(risk_score, 2)
        except Exception as e:
            print(f"❌ ML prediction error: {e}, falling back to rule-based")
            return calculate_risk_score_fallback(temperature, humidity, wind_speed)
    else:
        # Fallback to rule-based calculation
        return calculate_risk_score_fallback(temperature, humidity, wind_speed)

def calculate_risk_score_fallback(temperature: float, humidity: float, 
                                  wind_speed: float = 10) -> float:
    """Fallback rule-based risk calculation when ML model unavailable"""
    # Normalize factors (0-1 scale)
    temp_factor = min(temperature / 40.0, 1.0)  # 40°C = max risk
    humidity_factor = 1.0 - min(humidity / 100.0, 1.0)  # Lower humidity = higher risk
    wind_factor = min(wind_speed / 30.0, 1.0)  # 30 km/h = max risk
    
    # Weighted combination (adjust weights based on importance)
    risk_score = (temp_factor * 0.5 + humidity_factor * 0.3 + wind_factor * 0.2) * 100
    
    return round(risk_score, 2)

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
    try:
        api_key = os.getenv('NASA_MAP_KEY', '')
        url = NASA_FIRMS_API.format(country_code)
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

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing IoT sensor data.
    
    AWS IoT Core sends events in this format:
    {
        "deviceId": "esp32-01",
        "temperature": 23.4,
        "humidity": 40.2,
        "wind_speed": 12.5,
        "wind_direction": 180,
        "pressure": 1012,
        "lat": 43.467,
        "lng": -79.699,
        "timestamp": "2025-12-01T16:20:00Z"
    }
    """
    try:
        # Debug: Log the incoming event structure
        print(f"Received event: {json.dumps(event)}")
        
        # Parse IoT payload
        payload = None
        
        if 'deviceId' in event and 'temperature' in event:
            payload = event
        elif 'temperature' in event:
            payload = event
        elif 'body' in event:
            if isinstance(event['body'], str):
                payload = json.loads(event['body'])
            else:
                payload = event['body']
        else:
            payload = event
        
        if not payload:
            print(f"ERROR: Could not parse payload from event: {event}")
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid event structure'})}
        
        print(f"Parsed payload: {json.dumps(payload)}")
        
        # Extract sensor data
        device_id = payload.get('deviceId')
        temperature = float(payload.get('temperature', 0))
        humidity = float(payload.get('humidity', 0))
        wind_speed = float(payload.get('wind_speed', 10))
        wind_direction = float(payload.get('wind_direction', 0))
        pressure = float(payload.get('pressure', 1013))
        lat = float(payload.get('lat', 0))
        lng = float(payload.get('lng', 0))
        timestamp = payload.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        
        print(f"Extracted values - deviceId: {device_id}, temp: {temperature}, humidity: {humidity}, wind: {wind_speed}, lat: {lat}, lng: {lng}")
        
        # Validate required fields
        if not device_id:
            print(f"ERROR: deviceId is missing from payload: {payload}")
            return {'statusCode': 400, 'body': json.dumps({'error': 'deviceId is required'})}
        
        print(f"Processing sensor data for device: {device_id}")
        
        # Fetch NASA FIRMS wildfire data
        print("Fetching NASA FIRMS wildfire data...")
        fires = fetch_nasa_firms_data("CAN")
        print(f"Found {len(fires)} active fires")
        
        # Find nearest fire
        print("Finding nearest fire...")
        fire_info = find_nearest_fire(lat, lng, fires)
        fire_distance = fire_info.get('distance')
        print(f"Nearest fire distance: {fire_distance} km")
        
        # Calculate risk score using ML model (with fallback)
        print("Calculating ML-based risk score...")
        risk_score = calculate_risk_score_ml(
            temperature, humidity, wind_speed, wind_direction, pressure
        )
        print(f"ML Risk score: {risk_score}/100")
        
        # Determine risk level based on score
        if risk_score > 60:
            risk_level = "HIGH"
        elif risk_score > 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Prepare DynamoDB item
        print("Preparing DynamoDB item...")
        dynamodb_item = {
            'deviceId': device_id,
            'timestamp': timestamp,
            'temperature': Decimal(str(temperature)),
            'humidity': Decimal(str(humidity)),
            'wind_speed': Decimal(str(wind_speed)),
            'wind_direction': Decimal(str(wind_direction)),
            'pressure': Decimal(str(pressure)),
            'lat': Decimal(str(lat)),
            'lng': Decimal(str(lng)),
            'riskScore': Decimal(str(risk_score / 100)),  # Store as 0-1 scale
            'riskLevel': risk_level,
            'nearestFireDistance': Decimal(str(fire_distance)) if fire_distance else Decimal('-1'),
            'nearestFireData': json.dumps(fire_info.get('fire_data')) if fire_info.get('fire_data') else None,
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
                'fireDistance': fire_distance,
                'modelUsed': 'ml' if model_loaded else 'fallback'
            })
        }
        
    except Exception as e:
        print(f"Error processing sensor data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

# Load ML model on cold start
load_ml_model()