"""
API Gateway Lambda handler for dashboard endpoints.

Endpoints:
- GET /api/sensors - List all sensors
- GET /api/sensor/{id} - Get sensor by ID
- GET /api/risk-map - Get risk map data
- GET /api/nasa-fires - NASA FIRMS hotspots (needs NASA_MAP_KEY on this Lambda)
"""

import json
import boto3
import os
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime, timedelta
from decimal import Decimal
from nasa_firms_service import get_nasa_fires
from sensor_enrichment import merge_sensor_public_fields


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

table = dynamodb.Table(os.getenv('DYNAMODB_TABLE', 'WildfireSensorData'))


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
                def _f(v, default=None):
                    if v is None:
                        return default
                    if isinstance(v, Decimal):
                        return float(v)
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return v

                sensors[device_id] = {
                    'deviceId': device_id,
                    'temperature': _f(item.get('temperature'), 0),
                    'humidity': _f(item.get('humidity'), 0),
                    'lat': _f(item.get('lat'), 0),
                    'lng': _f(item.get('lng'), 0),
                    'riskScore': _f(item.get('riskScore'), 0),
                    'nearestFireDistance': _f(item.get('nearestFireDistance'), -1),
                    'timestamp': timestamp,
                    'riskLevel': item.get('riskLevel'),
                    'spreadRateKmh': _f(item.get('spreadRateKmh'), None) if item.get('spreadRateKmh') is not None else None,
                    'riskFactors': item.get('riskFactors'),
                    'recommendedAction': item.get('recommendedAction'),
                    'explanation': item.get('explanation'),
                }

        return [merge_sensor_public_fields(s) for s in sensors.values()]
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
            return merge_sensor_public_fields(result)
        return None
    except Exception as e:
        print(f"Error getting sensor {device_id}: {e}")
        return None


def get_risk_map_data():
    """Get risk map data for visualization (last 24 hours)."""
    try:
        # Get data from last 24 hours (timestamp is not the partition key — use Attr, not Key)
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat() + 'Z'
        items = []
        last_key = None
        while True:
            kwargs = {'FilterExpression': Attr('timestamp').gte(cutoff_time)}
            if last_key:
                kwargs['ExclusiveStartKey'] = last_key
            response = table.scan(**kwargs)
            items.extend(response.get('Items', []))
            last_key = response.get('LastEvaluatedKey')
            if not last_key:
                break

        def _f(v, default=None):
            if v is None:
                return default
            if isinstance(v, Decimal):
                return float(v)
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        map_data = []
        for item in items:
            rs = item.get('riskScore')
            nf = item.get('nearestFireDistance')
            row = {
                'deviceId': item['deviceId'],
                'lat': item.get('lat'),
                'lng': item.get('lng'),
                'riskScore': rs,
                'temperature': item.get('temperature'),
                'humidity': item.get('humidity'),
                'nearestFireDistance': nf,
                'timestamp': item.get('timestamp'),
                'riskLevel': item.get('riskLevel'),
                'spreadRateKmh': _f(item.get('spreadRateKmh'), None) if item.get('spreadRateKmh') is not None else None,
                'riskFactors': item.get('riskFactors'),
                'recommendedAction': item.get('recommendedAction'),
                'explanation': item.get('explanation'),
            }
            map_data.append(merge_sensor_public_fields(row))

        return map_data
    except Exception as e:
        print(f"Error getting risk map data: {e}")
        return []
    
def get_nasa_data(event, context):
    fires = get_nasa_fires()

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps({
            "source": "NASA FIRMS",
            "count": len(fires),
            "fires": fires
        }, default=decimal_default)
    }


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
            
            elif path == '/api/nasa-fires' or normalized_path == '/api/nasa-fires' or path == '/nasa-fires':
                return get_nasa_data(event, context)

            
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



