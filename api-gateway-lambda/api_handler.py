"""
API Gateway Lambda handler for dashboard endpoints.

Endpoints:
- GET /api/sensors - List all sensors
- GET /api/sensor/{id} - Get sensor by ID
- GET /api/risk-map - Get risk map data
- POST /api/risk-detailed - Get detailed risk scoring (PBI-7)
- GET /api/model-health - Get model health status (PBI-7)
"""

import json
import boto3
import os
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from decimal import Decimal
from nasa_firms_service import get_nasa_fires
from pathlib import Path
import sys

# Add parent directory for ML imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from inference.hybrid_scoring import predict_risk_hybrid
    from training.drift_detection import calculate_baseline_metrics, detect_drift
    from inference.store_prediction import store_prediction
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARN] ML modules not available - detailed scoring disabled")


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


def get_detailed_risk(payload):
    """
    Get detailed risk scoring with ML breakdown.
    Supports PBI-7: Enhanced Risk Scoring UI
    
    Args:
        payload: Dict with sensor data
    
    Returns:
        Dict with ml_score, rule_score, combined_score, etc.
    """
    if not ML_AVAILABLE:
        return {
            'error': 'ML modules not available',
            'combined_score': payload.get('riskScore', 0)
        }
    
    try:
        # Get hybrid prediction (ML + rules)
        result = predict_risk_hybrid(payload)
        
        # Check for model drift
        baseline = calculate_baseline_metrics(Path("training/logs/baseline_predictions.csv"))
        drift_result = detect_drift(Path("training/logs/predictions.csv"), baseline)
        
        # Store this prediction for monitoring
        store_prediction(result['combined_score'], result['combined_level'])
        
        response = {
            'ml_score': round(result['ml_score'], 2),
            'ml_level': result['ml_level'],
            'ml_confidence': round(result['ml_confidence'], 2),
            'rule_score': round(result['rule_score'], 2),
            'rule_level': result['rule_level'],
            'combined_score': round(result['combined_score'], 2),
            'combined_level': result['combined_level'],
            'model_version': result['model_version'],
            'model_drift': {
                'has_drift': drift_result.get('has_drift', False),
                'rmse_change_pct': round(drift_result.get('rmse_increase_pct', 0), 1),
                'num_predictions': drift_result.get('num_predictions', 0)
            }
        }
        
        return response
    
    except Exception as e:
        print(f"[ERROR] get_detailed_risk failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def get_model_health():
    """
    Get current model health status.
    Includes drift detection, prediction count, and recommendations.
    """
    if not ML_AVAILABLE:
        return {
            'status': 'UNKNOWN',
            'drift_detected': False,
            'predictions_count': 0,
            'error': 'ML modules not available'
        }
    
    try:
        baseline = calculate_baseline_metrics(Path("training/logs/baseline_predictions.csv"))
        drift_result = detect_drift(Path("training/logs/predictions.csv"), baseline)
        
        # Determine health status
        status = "HEALTHY"
        recommendations = []
        
        if drift_result.get('rmse_increase_pct', 0) > 25:
            status = "CRITICAL"
            recommendations.append("RMSE increased >25% - Retrain immediately")
        elif drift_result.get('has_drift', False):
            status = "WARNING"
            recommendations.append("Model drift detected - Consider retraining")
        
        if drift_result.get('num_predictions', 0) > 500:
            recommendations.append("High prediction volume - Monitor performance closely")
        
        response = {
            'status': status,
            'drift_detected': drift_result.get('has_drift', False),
            'predictions_count': drift_result.get('num_predictions', 0),
            'rmse': round(drift_result.get('current_rmse', 0), 2),
            'accuracy': round(drift_result.get('current_accuracy', 0), 3),
            'baseline_rmse': round(baseline.get('rmse', 0), 2),
            'recommendations': recommendations
        }
        
        return response
    
    except Exception as e:
        print(f"[ERROR] get_model_health failed: {e}")
        return {
            'status': 'UNKNOWN',
            'drift_detected': False,
            'predictions_count': 0,
            'error': str(e)
        }

    
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
            
            elif path == '/api/model-health' or normalized_path == '/api/model-health' or path == '/model-health':
                health = get_model_health()
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps(health, default=decimal_default)
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
        
        elif http_method == 'POST':
            if path == '/api/risk-detailed' or normalized_path == '/api/risk-detailed' or path == '/risk-detailed':
                try:
                    payload = json.loads(event.get('body', '{}'))
                    detailed = get_detailed_risk(payload)
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(detailed, default=decimal_default)
                    }
                except json.JSONDecodeError:
                    return {
                        'statusCode': 400,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps({'error': 'Invalid JSON'})
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