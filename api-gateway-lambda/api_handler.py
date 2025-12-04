"""
API Gateway Lambda handler for dashboard endpoints.

Endpoints:
- GET /api/sensors - List all sensors
- GET /api/sensor/{id} - Get sensor by ID
- GET /api/risk-map - Get risk map data
"""

import json
import boto3
import os
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta

# Support both local and AWS DynamoDB
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


def get_all_sensors():
    """Get all unique sensors with their latest data."""
    try:
        # Scan table to get all devices
        response = table.scan()
        items = response.get('Items', [])
        
        # Group by deviceId and get latest for each
        sensors = {}
        for item in items:
            device_id = item['deviceId']
            timestamp = item['timestamp']
            
            if device_id not in sensors or timestamp > sensors[device_id]['timestamp']:
                sensors[device_id] = {
                    'deviceId': device_id,
                    'temperature': item.get('temperature'),
                    'humidity': item.get('humidity'),
                    'lat': item.get('lat'),
                    'lng': item.get('lng'),
                    'riskScore': item.get('riskScore'),
                    'nearestFireDistance': item.get('nearestFireDistance'),
                    'timestamp': timestamp
                }
        
        return list(sensors.values())
    except Exception as e:
        print(f"Error getting sensors: {e}")
        return []


def get_sensor_by_id(device_id: str):
    """Get latest data for a specific sensor."""
    try:
        response = table.query(
            KeyConditionExpression=Key('deviceId').eq(device_id),
            ScanIndexForward=False,  # Get most recent first
            Limit=1
        )
        
        items = response.get('Items', [])
        if items:
            return items[0]
        return None
    except Exception as e:
        print(f"Error getting sensor {device_id}: {e}")
        return None


def get_risk_map_data():
    """Get risk map data for visualization (last 24 hours)."""
    try:
        # Get data from last 24 hours
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat() + 'Z'
        
        response = table.scan(
            FilterExpression=Key('timestamp').gte(cutoff_time)
        )
        
        items = response.get('Items', [])
        
        # Format for map visualization
        map_data = []
        for item in items:
            map_data.append({
                'deviceId': item['deviceId'],
                'lat': item.get('lat'),
                'lng': item.get('lng'),
                'riskScore': item.get('riskScore'),
                'temperature': item.get('temperature'),
                'humidity': item.get('humidity'),
                'nearestFireDistance': item.get('nearestFireDistance'),
                'timestamp': item.get('timestamp')
            })
        
        return map_data
    except Exception as e:
        print(f"Error getting risk map data: {e}")
        return []


def lambda_handler(event, context):
    """API Gateway Lambda handler."""
    try:
        http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
        path = event.get('path') or event.get('requestContext', {}).get('path') or event.get('rawPath', '')
        
        # Normalize path - API Gateway may send /api/sensors or /sensors depending on resource structure
        # Handle both formats for compatibility
        normalized_path = path
        if not path.startswith('/api'):
            normalized_path = f'/api{path}' if path.startswith('/') else f'/api/{path}'
        
        # Parse path parameters
        path_params = event.get('pathParameters') or {}
        
        # Route requests
        if (path == '/api/sensors' or normalized_path == '/api/sensors' or path == '/sensors') and http_method == 'GET':
            sensors = get_all_sensors()
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(sensors)
            }
        
        elif (path.startswith('/api/sensor/') or normalized_path.startswith('/api/sensor/') or path.startswith('/sensors/')) and http_method == 'GET':
            device_id = path_params.get('id') or path.split('/')[-1]
            sensor = get_sensor_by_id(device_id)
            
            if sensor:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(sensor)
                }
            else:
                return {
                    'statusCode': 404,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({'error': 'Sensor not found'})
                }
        
        elif (path == '/api/risk-map' or normalized_path == '/api/risk-map' or path == '/risk-map') and http_method == 'GET':
            map_data = get_risk_map_data()
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(map_data)
            }
        
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Not found'})
            }
            
    except Exception as e:
        print(f"Error in API handler: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }

