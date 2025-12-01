"""
AWS Lambda function to process IoT sensor data from AWS IoT Core.

This function:
1. Receives MQTT messages from IoT Core
2. Fetches NASA FIRMS wildfire data
3. Computes risk score
4. Stores enriched data in DynamoDB
"""

import json
import boto3
import requests
from datetime import datetime
from typing import Dict, Any

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('WildfireSensorData')

# NASA FIRMS API endpoint
NASA_FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/{}/MODIS_NRT/1"


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


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing IoT sensor data.
    
    Expected event structure from IoT Core:
    {
        "deviceId": "esp32-01",
        "temperature": 23.4,
        "humidity": 40.2,
        "lat": 43.467,
        "lng": -79.699,
        "timestamp": "2025-12-01T16:20:00Z"
    }
    """
    try:
        # Parse IoT payload
        if 'deviceId' in event:
            payload = event
        else:
            # IoT Core may wrap the payload
            payload = json.loads(event.get('body', json.dumps(event)))
        
        device_id = payload.get('deviceId')
        temperature = float(payload.get('temperature', 0))
        humidity = float(payload.get('humidity', 0))
        lat = float(payload.get('lat', 0))
        lng = float(payload.get('lng', 0))
        timestamp = payload.get('timestamp', datetime.utcnow().isoformat() + 'Z')
        
        # Fetch NASA FIRMS wildfire data
        fires = fetch_nasa_firms_data("CAN")  # Canada for now, can be parameterized
        
        # Find nearest fire
        fire_info = find_nearest_fire(lat, lng, fires)
        fire_distance = fire_info.get('distance')
        
        # Calculate risk score
        risk_score = calculate_risk_score(temperature, humidity, fire_distance)
        
        # Prepare DynamoDB item
        dynamodb_item = {
            'deviceId': device_id,
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'lat': lat,
            'lng': lng,
            'riskScore': risk_score,
            'nearestFireDistance': fire_distance if fire_distance else -1,
            'nearestFireData': json.dumps(fire_info.get('fire_data')) if fire_info.get('fire_data') else None,
            'ttl': int(datetime.utcnow().timestamp()) + (30 * 24 * 60 * 60)  # 30 days TTL
        }
        
        # Write to DynamoDB
        table.put_item(Item=dynamodb_item)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'deviceId': device_id,
                'riskScore': risk_score,
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

