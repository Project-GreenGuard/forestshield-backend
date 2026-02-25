"""
Local Flask server for development.
Simulates API Gateway + Lambda for local frontend development.
Now with ML wildfire prediction capabilities!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import boto3
from boto3.dynamodb.conditions import Key
from nasa_firms_service import get_nasa_fires
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import API handler functions - these now include Vertex AI integration
from api_handler import get_all_sensors, get_sensor_by_id, get_risk_map_data

app = Flask(__name__)
CORS(app)

# Configure DynamoDB client for local development
dynamodb_endpoint = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:8000')
dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url=dynamodb_endpoint,
    aws_access_key_id='dummy', #os.getenv('AWS_ACCESS_KEY_ID', 'local'),
    aws_secret_access_key='dummy', #os.getenv('AWS_SECRET_ACCESS_KEY', 'local'),
    region_name='us-east-1'
)

# ==================== ML Components (for local training only) ====================

class WildfireDataProcessor:
    """Process sensor and weather data for wildfire prediction"""
    
    def __init__(self):
        # These are the columns we have in our CSV file
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 
            'wind_direction', 'pressure', 'vegetation_index'
        ]
        # These are additional features we might add later
        self.optional_features = [
            'historical_fires_30d', 'days_since_rain'
        ]
    
    def process_sensor_data(self, mqtt_data):
        """
        Process incoming MQTT sensor data
        
        Args:
            mqtt_data: dict with sensor readings
        Returns:
            processed features for prediction
        """
        features = {}
        
        # Basic sensor features
        features['temperature'] = float(mqtt_data.get('temperature', 25))
        features['humidity'] = float(mqtt_data.get('humidity', 50))
        features['wind_speed'] = float(mqtt_data.get('wind_speed', 10))
        features['wind_direction'] = float(mqtt_data.get('wind_direction', 0))
        features['pressure'] = float(mqtt_data.get('pressure', 1013))
        
        # Calculate derived features
        features['vegetation_index'] = self._calculate_vegetation_index(
            features['humidity'], 
            mqtt_data.get('soil_moisture', 30)
        )
        
        # Optional features (set defaults)
        features['historical_fires_30d'] = 0
        features['days_since_rain'] = 7
        
        return features
    
    def _calculate_vegetation_index(self, humidity, soil_moisture):
        """Calculate simple vegetation moisture index"""
        return (humidity * 0.3 + soil_moisture * 0.7) / 100
    
    def _get_historical_fire_count(self, lat, lon):
        """Get number of fires in last 30 days for this area"""
        return 0
    
    def _get_days_since_rain(self, lat, lon):
        """Get days since last rainfall for this area"""
        return 7
    
    def prepare_training_data(self, historical_data_path):
        """
        Load and prepare historical data for training
        
        Args:
            historical_data_path: path to CSV with historical data
        Returns:
            X_train, y_train, X_test, y_test
        """
        try:
            print(f"📊 Loading training data from: {historical_data_path}")
            df = pd.read_csv(historical_data_path)
            
            # Clean column names (remove any whitespace)
            df.columns = df.columns.str.strip()
            
            print(f"✅ Data loaded. Shape: {df.shape}")
            print(f"📋 Columns found: {list(df.columns)}")
            
            # Check if required columns exist (the ones we have in CSV)
            required_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'vegetation_index']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return None, None, None, None
            
            # Check for target column
            if 'fire_risk' not in df.columns:
                print(f"❌ Target column 'fire_risk' not found")
                return None, None, None, None
            
            # Features - use only the columns we have
            X = df[required_cols]
            y = df['fire_risk']
            
            print(f"📊 Features shape: {X.shape}")
            print(f"📊 Target shape: {y.shape}")
            print(f"📊 First few rows of features:")
            print(X.head(3))
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"✅ Training data prepared. Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ Error preparing training data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None


class WildfireModelTrainer:
    """Train and save wildfire prediction models"""
    
    def __init__(self, model_type='regressor'):
        """
        Args:
            model_type: 'regressor' for risk scores, 'classifier' for fire/no-fire
        """
        self.model_type = model_type
        self.model = None
        self.processor = WildfireDataProcessor()
        
    def create_model(self):
        """Create a new model with reasonable defaults"""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        if self.model_type == 'regressor':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        print(f"🎯 Training model with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        print(f"✅ Model training complete")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score
        
        predictions = self.model.predict(X_test)
        
        if self.model_type == 'regressor':
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'r2': float(r2)
            }
        else:
            accuracy = accuracy_score(y_test, predictions)
            return {
                'accuracy': float(accuracy)
            }
    
    def save_model(self, model_name=None):
        """Save model to disk"""
        if model_name is None:
            model_name = f"wildfire_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        model_path = os.path.join('ml_models', model_name)
        os.makedirs('ml_models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        print(f"✅ Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


class WildfirePredictor:
    """Make predictions using trained models"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.processor = WildfireDataProcessor()
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a specific model"""
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            print(f"✅ Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_latest_model(self, models_dir='ml_models'):
        """Load the most recently saved model"""
        if not os.path.exists(models_dir):
            print(f"⚠️ No models directory found at {models_dir}")
            return False
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        if not model_files:
            print("⚠️ No models found")
            return False
        
        # Get latest by name (assuming timestamp in filename)
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        return self.load_model(model_path)
    
    def predict_from_sensor_data(self, sensor_data):
        """
        Make prediction from live sensor data
        
        Args:
            sensor_data: dict from sensor or API
        Returns:
            prediction dict with risk score and level
        """
        if self.model is None:
            if not self.load_latest_model():
                return {
                    'error': 'No model loaded. Train a model first using POST /api/ml/train',
                    'risk_score': None,
                    'risk_level': 'UNKNOWN'
                }
        
        try:
            # Process sensor data into features
            features = self.processor.process_sensor_data(sensor_data)
            
            # Use only the features the model was trained on (first 6)
            feature_array = np.array([[
                features['temperature'],
                features['humidity'],
                features['wind_speed'],
                features['wind_direction'],
                features['pressure'],
                features['vegetation_index']
            ]])
            
            # Make prediction
            risk_score = float(self.model.predict(feature_array)[0])
            
            # Ensure risk_score is between 0 and 1
            risk_score = max(0, min(1, risk_score))
            
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
                'features': {k: v for k, v in features.items() if k in self.processor.feature_columns},
                'timestamp': sensor_data.get('timestamp', datetime.now().isoformat()),
                'sensor_id': sensor_data.get('sensor_id', sensor_data.get('device_id', 'unknown'))
            }
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'risk_score': None,
                'risk_level': 'ERROR'
            }
    
    def predict_batch(self, features_df):
        """Make predictions for multiple data points"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        predictions = self.model.predict(features_df)
        return predictions


# Initialize ML components (for local training only)
ml_processor = WildfireDataProcessor()
ml_trainer = WildfireModelTrainer()
ml_predictor = WildfirePredictor()

# Try to load latest model on startup
if ml_predictor.load_latest_model():
    print("✅ Local ML model loaded successfully")
else:
    print("⚠️ No local ML model found. Train one using POST /api/ml/train")

# ==================== ML API Endpoints ====================

@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """
    Predict fire risk from sensor data
    Expects JSON with sensor readings
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Make prediction using local predictor
        result = ml_predictor.predict_from_sensor_data(data)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 404
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """
    Train a new model with historical data
    Expects JSON with data_path or file upload
    """
    try:
        data = request.json or {}
        data_path = data.get('data_path', 'data/training_data.csv')
        
        print(f"\n{'='*60}")
        print(f"🚀 Training request received")
        print(f"{'='*60}")
        print(f"Requested path: {data_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(__file__)}")
        
        # Handle different path formats
        if not os.path.isabs(data_path):
            # Try multiple possible locations
            possible_paths = [
                data_path,
                os.path.join(os.getcwd(), data_path),
                os.path.join(os.path.dirname(__file__), '..', data_path),
                os.path.join(os.path.dirname(__file__), data_path),
                os.path.join(os.getcwd(), 'data', 'training_data.csv'),
                'C:/Users/Owner/Desktop/sample/forestshield-backend/data/training_data.csv'
            ]
            
            found_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                print(f"Checking: {abs_path}")
                if os.path.exists(abs_path):
                    found_path = abs_path
                    print(f"✅ Found at: {found_path}")
                    break
            
            if found_path:
                data_path = found_path
            else:
                return jsonify({
                    'success': False,
                    'error': f'Data file not found. Checked: {possible_paths}'
                }), 404
        
        # Check if file exists
        if not os.path.exists(data_path):
            return jsonify({
                'success': False,
                'error': f'Data file not found: {data_path}'
            }), 404
        
        print(f"📊 Using file: {data_path}")
        
        # Prepare training data
        X_train, X_test, y_train, y_test = ml_processor.prepare_training_data(data_path)
        
        if X_train is None:
            return jsonify({
                'success': False,
                'error': 'Failed to prepare training data. Check file format and column names.'
            }), 400
        
        # Train model
        ml_trainer.train(X_train, y_train)
        
        # Evaluate
        metrics = ml_trainer.evaluate(X_test, y_test)
        
        # Save model
        model_path = ml_trainer.save_model()
        
        # Reload predictor with new model
        ml_predictor.load_model(model_path)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics,
            'model_path': model_path
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ml/model/info', methods=['GET'])
def ml_model_info():
    """Get information about current loaded model"""
    if ml_predictor.model is None:
        return jsonify({
            'loaded': False,
            'message': 'No model loaded. Train one using POST /api/ml/train'
        })
    
    return jsonify({
        'loaded': True,
        'model_type': type(ml_predictor.model).__name__,
        'model_path': ml_predictor.model_path,
        'features': ml_processor.feature_columns,
        'feature_count': len(ml_processor.feature_columns)
    })

@app.route('/api/ml/health', methods=['GET'])
def ml_health():
    """ML component health check"""
    return jsonify({
        'model_loaded': ml_predictor.model is not None,
        'model_path': ml_predictor.model_path,
        'processor_ready': True
    })

# ==================== Existing API Endpoints ====================

@app.route('/api/sensors', methods=['GET'])
def api_sensors():
    """Get all sensors endpoint"""
    # Get sensors from api_handler - they already include ML predictions from Vertex AI or local model
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
    # Get sensor from api_handler - already includes ML predictions
    sensor = get_sensor_by_id(device_id)
    if sensor:
        return jsonify(sensor)
    else:
        return jsonify({'error': 'Sensor not found'}), 404

@app.route('/api/risk-map', methods=['GET'])
def api_risk_map():
    """Get risk map data endpoint with ML predictions"""
    # Get map data from api_handler - already includes ML predictions
    map_data = get_risk_map_data()
    return jsonify(map_data)

@app.route('/api/ml/predict/sensor/<device_id>', methods=['GET'])
def ml_predict_sensor(device_id):
    """Get ML prediction for a specific sensor"""
    sensor = get_sensor_by_id(device_id)
    if not sensor:
        return jsonify({'error': 'Sensor not found'}), 404
    
    result = ml_predictor.predict_from_sensor_data(sensor)
    
    if 'error' in result:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 404
    
    return jsonify({
        'success': True,
        'device_id': device_id,
        'prediction': result
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok', 
        'service': 'forestshield-api',
        'ml_status': 'loaded' if ml_predictor.model is not None else 'no_model'
    })

# ==================== Create sample data file if not exists ====================
def create_sample_data():
    """Create sample historical data file for testing"""
    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created data directory: {data_dir}")
    
    # Create training_data.csv (your actual training file)
    training_data_path = os.path.join(data_dir, 'training_data.csv')
    if not os.path.exists(training_data_path):
        training_data = """temperature,humidity,wind_speed,wind_direction,pressure,vegetation_index,fire_risk
32.5,45,12,180,1012,0.45,0.8
28.3,60,8,90,1015,0.65,0.4
35.1,30,20,270,1008,0.25,0.9
22.4,80,5,45,1020,0.85,0.1
30.2,55,15,135,1013,0.55,0.6
33.7,38,18,200,1010,0.35,0.85
26.8,70,7,80,1017,0.75,0.25
31.5,48,14,160,1011,0.5,0.7
29.4,58,10,110,1014,0.6,0.5
36.2,25,25,300,1005,0.2,0.95"""
        
        with open(training_data_path, 'w') as f:
            f.write(training_data)
        
        print(f"✅ Training data created at {training_data_path} with 10 samples")
    else:
        print(f"📁 Training data already exists at {training_data_path}")
    
    # Also create sample_historical_fires.csv for backward compatibility
    sample_data_path = os.path.join(data_dir, 'sample_historical_fires.csv')
    if not os.path.exists(sample_data_path):
        sample_data = """temperature,humidity,wind_speed,wind_direction,pressure,vegetation_index,historical_fires_30d,days_since_rain,fire_risk
32.5,45,12,180,1012,0.45,2,7,0.8
28.3,60,8,90,1015,0.65,1,3,0.4
35.1,30,20,270,1008,0.25,3,15,0.9
22.4,80,5,45,1020,0.85,0,1,0.1
30.2,55,15,135,1013,0.55,1,5,0.6"""
        
        with open(sample_data_path, 'w') as f:
            f.write(sample_data)
        
        print(f"✅ Sample data created at {sample_data_path}")

# Create sample data on startup
create_sample_data()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌲 ForestShield API Server Starting...")
    print("="*60)
    print("📡 API available at http://localhost:5001")
    print("\n📋 Existing Endpoints:")
    print("  GET  /api/sensors")
    print("  GET  /api/sensor/<id>")
    print("  GET  /api/risk-map")
    print("  GET  /api/nasa-fires")
    print("\n🤖 NEW ML Endpoints:")
    print("  POST /api/ml/predict           - Predict fire risk from sensor data")
    print("  POST /api/ml/train              - Train new model with historical data")
    print("  GET  /api/ml/model/info         - Get current model info")
    print("  GET  /api/ml/health              - ML component health check")
    print("  GET  /api/ml/predict/sensor/<id> - Predict for specific sensor")
    print("\n📊 Data Files:")
    print("  📁 data/training_data.csv - Your training data (10 samples)")
    print("  📁 data/sample_historical_fires.csv - Sample data with additional features")
    print("\n🚀 Quick Start:")
    print("  1. Train a model:  Invoke-RestMethod -Uri 'http://localhost:5001/api/ml/train' -Method POST -Headers @{'Content-Type'='application/json'} -Body '{\"data_path\": \"data/training_data.csv\"}'")
    print("  2. Make a prediction: Invoke-RestMethod -Uri 'http://localhost:5001/api/ml/predict' -Method POST -Headers @{'Content-Type'='application/json'} -Body '{\"temperature\": 35, \"humidity\": 40, \"wind_speed\": 20}'")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)