"""
API Gateway Lambda handler for dashboard endpoints.

Endpoints:
- GET /api/sensors - List all sensors
- GET /api/sensor/{id} - Get sensor by ID
- GET /api/risk-map - Get risk map data
- GET /api/nasa-fires - Get NASA FIRMS data
- POST /api/ml/predict - ML prediction endpoint
- POST /api/ml/train - Train ML model
"""

import json
import boto3
import os
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from decimal import Decimal
from nasa_firms_service import get_nasa_fires
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ==================== ML Components ====================

class WildfireML:
    """ML prediction for wildfire risk"""
    
    def __init__(self):
        self.model = None
        # Save model in the same directory as this script
        self.model_path = os.path.join(os.path.dirname(__file__), 'ml_model.joblib')
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 
            'wind_direction', 'pressure', 'vegetation_index'
        ]
        self.load_model()
    
    def load_model(self):
        """Load trained model if exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ ML Model loaded from {self.model_path}")
            else:
                print("⚠️ No trained model found. Using rule-based fallback.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def predict(self, sensor_data):
        """
        Predict fire risk from sensor data
        
        Args:
            sensor_data: dict with temperature, humidity, etc.
        Returns:
            risk_score (0-1) and risk_level
        """
        try:
            if self.model is not None:
                # Use ML model with proper feature names (fixes the warning)
                # Create DataFrame with proper column names
                features_df = pd.DataFrame([[
                    float(sensor_data.get('temperature', 25)),
                    float(sensor_data.get('humidity', 50)),
                    float(sensor_data.get('wind_speed', 10)),
                    float(sensor_data.get('wind_direction', 0)),
                    float(sensor_data.get('pressure', 1013)),
                    self._calculate_vegetation_index(sensor_data)
                ]], columns=self.feature_columns)
                
                risk_score = float(self.model.predict(features_df)[0])
                # Ensure risk_score is between 0 and 1
                risk_score = max(0, min(1, risk_score))
            else:
                # Fallback to rule-based calculation
                risk_score = self._rule_based_risk(sensor_data)
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'HIGH'
            elif risk_score >= 0.3:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'risk_score': round(risk_score, 3),
                'risk_level': risk_level,
                'model_used': 'ml' if self.model else 'rule-based'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'risk_score': 0.5,
                'risk_level': 'MEDIUM',
                'error': str(e)
            }
    
    def _calculate_vegetation_index(self, sensor_data):
        """Calculate vegetation moisture index"""
        humidity = float(sensor_data.get('humidity', 50))
        soil_moisture = float(sensor_data.get('soil_moisture', 30))
        return (humidity * 0.3 + soil_moisture * 0.7) / 100
    
    def _rule_based_risk(self, sensor_data):
        """Simple rule-based risk calculation as fallback"""
        temp = float(sensor_data.get('temperature', 25))
        humidity = float(sensor_data.get('humidity', 50))
        wind = float(sensor_data.get('wind_speed', 10))
        
        # Simple rules
        temp_risk = min(1.0, max(0, (temp - 20) / 30))
        humidity_risk = 1 - min(1.0, humidity / 80)
        wind_risk = min(1.0, wind / 30)
        
        risk = (temp_risk * 0.5 + humidity_risk * 0.3 + wind_risk * 0.2)
        return max(0, min(1, risk))
    
    def train(self, training_data_path):
        """Train ML model with historical data"""
        try:
            # Check if file exists
            if not os.path.exists(training_data_path):
                return {
                    'success': False,
                    'error': f'Training data file not found: {training_data_path}'
                }
            
            print(f"📊 Loading training data from: {training_data_path}")
            
            # Load training data
            df = pd.read_csv(training_data_path)
            
            print(f"✅ Data loaded successfully!")
            print(f"📊 Data shape: {df.shape}")
            print(f"📋 Columns found: {list(df.columns)}")
            
            # Clean column names (remove any whitespace)
            df.columns = df.columns.str.strip()
            
            # Check if required columns exist
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                return {
                    'success': False,
                    'error': f'Missing columns in training data: {missing_cols}. Available columns: {list(df.columns)}'
                }
            
            if 'fire_risk' not in df.columns:
                return {
                    'success': False,
                    'error': f'Target column "fire_risk" not found in training data. Available columns: {list(df.columns)}'
                }
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = df['fire_risk']
            
            print(f"🎯 Training with {len(X)} samples, {len(X.columns)} features")
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            print(f"✅ Model trained and saved to {self.model_path}")
            
            # Evaluate
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            score = self.model.score(X_test, y_test)
            
            return {
                'success': True,
                'model_path': self.model_path,
                'r2_score': round(score, 4),
                'training_samples': len(X),
                'features_used': self.feature_columns
            }
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


# Initialize ML
ml_model = WildfireML()

# ==================== DynamoDB Setup ====================

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
    dynamodb = boto3.resource(
        'dynamodb',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

table = dynamodb.Table('WildfireSensorData')


def decimal_default(obj):
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def get_all_sensors():
    """Get all unique sensors with their latest data."""
    try:
        items = []
        last_evaluated_key = None
        
        while True:
            if last_evaluated_key:
                response = table.scan(ExclusiveStartKey=last_evaluated_key)
            else:
                response = table.scan()
            
            items.extend(response.get('Items', []))
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        sensors = {}
        for item in items:
            device_id = item['deviceId']
            timestamp = item['timestamp']
            
            if device_id not in sensors or timestamp > sensors[device_id]['timestamp']:
                # ADDED: location field to sensor_data
                sensor_data = {
                    'deviceId': device_id,
                    'location': item.get('location', 'Unknown'),  # ← LOCATION ADDED
                    'temperature': float(item.get('temperature', 0)) if isinstance(item.get('temperature'), Decimal) else item.get('temperature', 25),
                    'humidity': float(item.get('humidity', 0)) if isinstance(item.get('humidity'), Decimal) else item.get('humidity', 50),
                    'wind_speed': float(item.get('wind_speed', 0)) if isinstance(item.get('wind_speed'), Decimal) else item.get('wind_speed', 10),
                    'wind_direction': float(item.get('wind_direction', 0)) if isinstance(item.get('wind_direction'), Decimal) else item.get('wind_direction', 0),
                    'pressure': float(item.get('pressure', 0)) if isinstance(item.get('pressure'), Decimal) else item.get('pressure', 1013),
                    'lat': float(item.get('lat', 0)) if isinstance(item.get('lat'), Decimal) else item.get('lat'),
                    'lng': float(item.get('lng', 0)) if isinstance(item.get('lng'), Decimal) else item.get('lng'),
                    'nearestFireDistance': float(item.get('nearestFireDistance', -1)) if isinstance(item.get('nearestFireDistance'), Decimal) else item.get('nearestFireDistance'),
                    'timestamp': timestamp
                }
                
                prediction = ml_model.predict(sensor_data)
                sensor_data['riskScore'] = prediction['risk_score']
                sensor_data['riskLevel'] = prediction['risk_level']
                
                sensors[device_id] = sensor_data
        
        return list(sensors.values())
    except Exception as e:
        print(f"Error getting sensors: {e}")
        return []


def get_sensor_by_id(device_id: str):
    """Get latest data for a specific sensor."""
    try:
        response = table.query(
            KeyConditionExpression=Key('deviceId').eq(device_id),
            ScanIndexForward=False,
            Limit=1
        )
        
        items = response.get('Items', [])
        if items:
            item = items[0]
            result = {}
            for key, value in item.items():
                if isinstance(value, Decimal):
                    result[key] = float(value)
                else:
                    result[key] = value
            
            # ADDED: Ensure location is in the result
            if 'location' not in result:
                result['location'] = 'Unknown'
            
            prediction = ml_model.predict(result)
            result['riskScore'] = prediction['risk_score']
            result['riskLevel'] = prediction['risk_level']
            
            return result
        return None
    except Exception as e:
        print(f"Error getting sensor {device_id}: {e}")
        return None


def get_risk_map_data():
    """Get risk map data for visualization (last 24 hours)."""
    try:
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat() + 'Z'
        
        response = table.scan(
            FilterExpression=Key('timestamp').gte(cutoff_time)
        )
        
        items = response.get('Items', [])
        
        map_data = []
        for item in items:
            sensor_data = {}
            for key, value in item.items():
                if isinstance(value, Decimal):
                    sensor_data[key] = float(value)
                else:
                    sensor_data[key] = value
            
            prediction = ml_model.predict(sensor_data)
            
            # ADDED: location field to map data
            map_data.append({
                'deviceId': sensor_data.get('deviceId'),
                'location': sensor_data.get('location', 'Unknown'),  # ← LOCATION ADDED
                'lat': sensor_data.get('lat'),
                'lng': sensor_data.get('lng'),
                'riskScore': prediction['risk_score'],
                'riskLevel': prediction['risk_level'],
                'temperature': sensor_data.get('temperature'),
                'humidity': sensor_data.get('humidity'),
                'wind_speed': sensor_data.get('wind_speed'),
                'nearestFireDistance': sensor_data.get('nearestFireDistance'),
                'timestamp': sensor_data.get('timestamp')
            })
        
        return map_data
    except Exception as e:
        print(f"Error getting risk map data: {e}")
        return []


# ==================== ML API Handlers ====================

def ml_predict_handler(event):
    """Handle ML prediction requests"""
    try:
        body = json.loads(event.get('body', '{}'))
        prediction = ml_model.predict(body)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'prediction': prediction
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }


def ml_train_handler(event):
    """Handle ML training requests"""
    try:
        body = json.loads(event.get('body', '{}'))
        data_path = body.get('data_path', 'data/training_data.csv')
        
        # Handle Windows paths
        if isinstance(data_path, str):
            data_path = data_path.replace('\\', '/')
        
        # If it's a relative path, make it absolute
        if not os.path.isabs(data_path):
            # Try multiple possible locations
            possible_paths = [
                os.path.join(os.getcwd(), data_path),  # Current directory
                os.path.join(os.path.dirname(__file__), '..', data_path),  # Project root
                os.path.join(os.path.dirname(__file__), data_path),  # Same directory as script
                data_path  # As is
            ]
        else:
            possible_paths = [data_path]
        
        full_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            print(f"Checking: {abs_path}")
            if os.path.exists(abs_path):
                full_path = abs_path
                print(f"✅ Found at: {full_path}")
                break
        
        if full_path is None:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': False,
                    'error': f'Data file not found. Checked: {possible_paths}'
                })
            }
        
        result = ml_model.train(full_path)
        
        if result['success']:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps(result)
            }
    except Exception as e:
        print(f"Training handler error: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }


def ml_model_info_handler():
    """Get ML model information"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'model_loaded': ml_model.model is not None,
            'model_path': ml_model.model_path,
            'features': ml_model.feature_columns,
            'model_type': type(ml_model.model).__name__ if ml_model.model else None
        })
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
        
        if not path:
            path = '/'
        
        normalized_path = path
        if path and not path.startswith('/api'):
            normalized_path = f'/api{path}' if path.startswith('/') else f'/api/{path}'
        
        path_params = event.get('pathParameters') or {}
        
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
            
            elif path == '/api/ml/info' or normalized_path == '/api/ml/info':
                return ml_model_info_handler()
            
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
            if path == '/api/ml/predict' or normalized_path == '/api/ml/predict':
                return ml_predict_handler(event)
            
            elif path == '/api/ml/train' or normalized_path == '/api/ml/train':
                return ml_train_handler(event)
            
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