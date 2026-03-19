"""ForestShield API Server - ML Models with Sensor Data"""
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import boto3
from datetime import datetime
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
AI_DIR = Path(__file__).resolve().parents[2] / "forestshield-ai"
sys.path.insert(0, str(AI_DIR))

# DynamoDB
dynamodb_endpoint = os.getenv('AWS_ENDPOINT_URL', 'http://dynamodb:8000')
dynamodb = boto3.resource('dynamodb', endpoint_url=dynamodb_endpoint,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'local'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'local'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

# ===== LOAD MODELS =====
print("\n🔥 Loading ML models...")

# Fire Spread Model
try:
    from inference.fire_spread_model import FireSpreadModel
    fire_model = FireSpreadModel()
    HAS_FIRE_MODEL = fire_model.use_ml
    if HAS_FIRE_MODEL:
        print("✅ fire_spread_model (ML) loaded")
    else:
        print("⚠️  fire_spread_model (fallback formulas)")
except Exception as e:
    print(f"❌ fire_spread_model error: {e}")
    from inference.fire_spread_model import FireSpreadModel
    fire_model = FireSpreadModel()
    HAS_FIRE_MODEL = False

# Risk Model
try:
    model_dir = Path(__file__).resolve().parents[2] / "forestshield-ai" / "training" / "models"
    wildfire_risk_model = joblib.load(model_dir / "wildfire_risk_model.pkl")
    risk_scaler = joblib.load(model_dir / "wildfire_risk_scaler.pkl") if (model_dir / "wildfire_risk_scaler.pkl").exists() else None
    HAS_RISK_MODEL = True
    print("✅ wildfire_risk_model loaded")
except Exception as e:
    HAS_RISK_MODEL = False
    print(f"⚠️  wildfire_risk_model: {e}")

# Features Module
try:
    from utils.features import build_feature_vector
    HAS_FEATURES_MODULE = True
    print("✅ features module loaded")
except ImportError:
    HAS_FEATURES_MODULE = False
    print("⚠️  features module not available")

# API Handler
try:
    from api_handler import get_sensor_by_id, get_all_sensors
    print("✅ api_handler loaded\n")
except ImportError:
    def get_sensor_by_id(id): return None
    def get_all_sensors(): return []
    print("⚠️  api_handler not available\n")

# ===== FLASK APP =====
app = Flask(__name__)
CORS(app)

def predict_risk(features_vector):
    """Predict risk using trained model"""
    if not HAS_RISK_MODEL:
        return None
    try:
        if not isinstance(features_vector, np.ndarray):
            features_vector = np.array(features_vector).reshape(1, -1)
        else:
            features_vector = features_vector.reshape(1, -1)
        features_scaled = risk_scaler.transform(features_vector) if risk_scaler else features_vector
        risk_score = wildfire_risk_model.predict(features_scaled)[0]
        return max(0, min(100, risk_score))
    except Exception as e:
        print(f"❌ Risk prediction error: {e}")
        return None

# ===== ENDPOINTS =====
@app.route('/api/sensors', methods=['GET'])
def api_sensors():
    return jsonify(get_all_sensors() or [])

@app.route('/api/sensor/<sensor_id>', methods=['GET'])
def api_sensor(sensor_id):
    sensor = get_sensor_by_id(sensor_id)
    return jsonify(sensor) if sensor else jsonify({'error': 'Not found'}), 404

@app.route('/api/sensor/<sensor_id>/ml-risk', methods=['GET'])
def api_sensor_ml_risk(sensor_id):
    sensor_data = get_sensor_by_id(sensor_id)
    if not sensor_data:
        return jsonify({'error': 'Sensor not found'}), 404
    
    print(f"\n📊 /api/sensor/{sensor_id}/ml-risk")
    
    ml_score = None
    if HAS_RISK_MODEL and HAS_FEATURES_MODULE:
        try:
            feature_payload = {
                'temperature': sensor_data.get('temperature', 0),
                'humidity': sensor_data.get('humidity', 50),
                'lat': sensor_data.get('lat', 43.65),
                'lng': sensor_data.get('lng', -79.38),
                'nearestFireDistance': sensor_data.get('nearestFireDistance', 3),
                'timestamp': sensor_data.get('timestamp', datetime.now().isoformat())
            }
            features_vector = build_feature_vector(feature_payload)
            ml_score = predict_risk(features_vector)
            print(f"   ✅ ML Risk Score: {ml_score:.2f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            ml_score = None
    
    # If ML failed, use rule-based
    if ml_score is None:
        ml_score = sensor_data.get('riskScore', 50)
        print(f"   ⚠️  Using fallback rule-based: {ml_score:.2f}")
    
    # Get rule-based score for blending
    rule_score = sensor_data.get('riskScore', 50)
    
    # Blend: 70% ML + 30% Rule
    final_score = (ml_score * 0.7) + (rule_score * 0.3)
    final_score = max(0, min(100, final_score))
    
    # Determine level
    risk_level = 'HIGH' if final_score >= 61 else 'MEDIUM' if final_score >= 31 else 'LOW'
    
    print(f"   Rule Score: {rule_score:.2f}")
    print(f"   Blended (70% ML + 30% Rule): {final_score:.2f}")
    print(f"   Level: {risk_level}\n")
    
    return jsonify({
        "sensor_id": sensor_id,
        "timestamp": datetime.now().isoformat(),
        "risk_score": round(final_score, 1),
        "risk_level": risk_level,
        "confidence": 0.85,
        "ml_score": round(ml_score, 1),
        "rule_score": round(rule_score, 1),
        "blended_score": round(final_score, 1)
    })
@app.route('/api/sensor/<sensor_id>/fire-spread', methods=['GET'])
def api_sensor_fire_spread(sensor_id):
    hours = int(request.args.get('hours', 12))
    sensor_data = get_sensor_by_id(sensor_id)
    if not sensor_data:
        return jsonify({'error': 'Sensor not found'}), 404
    
    print(f"\n🔥 /api/sensor/{sensor_id}/fire-spread?hours={hours}")
    
    # GET RISK SCORE FROM RISK MODEL
    risk_score = sensor_data.get('riskScore', 50)
    print(f"   Risk Score: {risk_score}")
    
    fire_data = {
        'lat': sensor_data.get('lat', 43.65),
        'lng': sensor_data.get('lng', -79.38),
        'temperature': sensor_data.get('temperature', 25),
        'humidity': sensor_data.get('humidity', 50),
        'wind_speed': sensor_data.get('wind_speed', 12),
        'wind_direction': sensor_data.get('wind_direction', 180),
        'vegetation_density': sensor_data.get('vegetation_density', 0.7),
        'soil_moisture': sensor_data.get('soil_moisture', 0.3),
        'elevation': sensor_data.get('elevation', 300),
        'nearest_water': sensor_data.get('nearest_water', 5),
        'fire_history': sensor_data.get('fire_history', 3),
        'population_density': sensor_data.get('population_density', 20),
        'intensity': sensor_data.get('intensity', 50)
    }
    
    try:
        fire_result = fire_model.predict_spread(fire_data, hours)
        print(f"   ✅ Fire spread: {fire_result.get('estimated_speed', 2.3):.2f} km/h")
        print(f"   Spread Risk Level: {fire_result.get('spread_risk_level', 'UNKNOWN')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        fire_result = fire_model._predict_with_formulas(fire_data, hours)
    
    spread = fire_result.get('fire_spread', {})
    if not spread:
        spread = {
            'spread_rate_kmh': fire_result.get('estimated_speed', 2.3),
            'direction_degrees': fire_result.get('direction', 180),
            'area_hectares': fire_result.get('affected_radius', 15.5),
            'intensity_level': fire_result.get('predicted_intensity', 7)
        }
    
    return jsonify({
        "sensor_id": sensor_id,
        "timestamp": datetime.now().isoformat(),
        "risk_score": round(risk_score, 1),  # ADD FROM RISK MODEL
        "ignition_point": {"lat": fire_data['lat'], "lng": fire_data['lng']},
        "forecast_hours": hours,
        "fire_spread": {
            "spread_rate_kmh": round(spread.get('spread_rate_kmh', fire_result.get('estimated_speed', 2.3)), 1),
            "direction_degrees": round(spread.get('direction_degrees', fire_result.get('direction', 180)), 1),
            "area_hectares": round(spread.get('area_hectares', fire_result.get('affected_radius', 15.5)), 1),
            "intensity_level": round(spread.get('intensity_level', fire_result.get('predicted_intensity', 7)), 1)
        },
        "spread_risk_level": fire_result.get('spread_risk_level', 'UNKNOWN'),  # ADD THIS
        "danger_score": fire_result.get('danger_score', 0),  # ADD THIS
        "model_type": fire_result.get('model_type', 'UNKNOWN'),
        "model_confidence": fire_result.get('model_confidence', 0.65)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'fire_model': 'ML' if HAS_FIRE_MODEL else 'Formulas',
        'risk_model': 'loaded' if HAS_RISK_MODEL else 'not_loaded',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 ForestShield API Server")
    print("="*70)
    print(f"📍 http://localhost:5001")
    print(f"🔥 Fire Model: {'✅ ML Loaded' if HAS_FIRE_MODEL else '⚠️  Formulas'}")
    print(f"📊 Risk Model: {'✅ ML Loaded' if HAS_RISK_MODEL else '❌ Not loaded'}")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)