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
from decimal import Decimal

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


def decimal_default(obj):
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def get_all_sensors():
    """Get all unique sensors with their latest data."""
    try:
        # Scan table with pagination to get all devices
        items = []
        last_evaluated_key = None
        
        while True:
            if last_evaluated_key:
                response = table.scan(ExclusiveStartKey=last_evaluated_key)
            else:
                response = table.scan()
            
            items.extend(response.get('Items', []))
            
            # Check if there are more items to scan
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        # Group by deviceId and get latest for each
        sensors = {}
        for item in items:
            device_id = item['deviceId']
            timestamp = item['timestamp']
            
            # Compare timestamps (ISO format strings compare correctly)
            if device_id not in sensors or timestamp > sensors[device_id]['timestamp']:
                sensors[device_id] = {
                    'deviceId': device_id,
                    'temperature': float(item.get('temperature', 0)) if isinstance(item.get('temperature'), Decimal) else item.get('temperature'),
                    'humidity': float(item.get('humidity', 0)) if isinstance(item.get('humidity'), Decimal) else item.get('humidity'),
                    'lat': float(item.get('lat', 0)) if isinstance(item.get('lat'), Decimal) else item.get('lat'),
                    'lng': float(item.get('lng', 0)) if isinstance(item.get('lng'), Decimal) else item.get('lng'),
                    'riskScore': float(item.get('riskScore', 0)) if isinstance(item.get('riskScore'), Decimal) else item.get('riskScore'),
                    'nearestFireDistance': float(item.get('nearestFireDistance', -1)) if isinstance(item.get('nearestFireDistance'), Decimal) else item.get('nearestFireDistance'),
                    'timestamp': timestamp
                }
        
        return list(sensors.values())
    except Exception as e:
        print(f"Error getting sensors: {e}")
        import traceback
        traceback.print_exc()
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
            item = items[0]
            # Convert Decimal to float for JSON serialization
            result = {}
            for key, value in item.items():
                if isinstance(value, Decimal):
                    result[key] = float(value)
                else:
                    result[key] = value
            return result
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
        
        print(f"Parsed - Method: {http_method}, Path: {path}")
        
        # Normalize path - API Gateway may send /api/sensors or /sensors depending on resource structure
        # Handle both formats for compatibility
        if not path:
            path = '/'
        normalized_path = path
        if path and not path.startswith('/api'):
            normalized_path = f'/api{path}' if path.startswith('/') else f'/api/{path}'
        
        # Parse path parameters
        path_params = event.get('pathParameters') or {}
        
        # Route requests - check both original path and normalized path
        if http_method == 'GET':
            if path == '/api/sensors' or normalized_path == '/api/sensors' or path == '/sensors':
                sensors = get_all_sensors()
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(sensors, default=decimal_default)
                }
            
            elif path.startswith('/api/sensor/') or normalized_path.startswith('/api/sensor/') or path.startswith('/sensors/'):
                device_id = path_params.get('id') or path.split('/')[-1]
                sensor = get_sensor_by_id(device_id)
                
                if sensor:
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(sensor, default=decimal_default)
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
            
            elif path == '/api/risk-map' or normalized_path == '/api/risk-map' or path == '/risk-map':
                map_data = get_risk_map_data()
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(map_data, default=decimal_default)
                }
            
            else:
                return {
                    'statusCode': 404,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({'error': 'Not found', 'path': path, 'method': http_method})
                }
        else:
            return {
                'statusCode': 405,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Method not allowed'})
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

