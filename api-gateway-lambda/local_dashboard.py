"""
Local Flask server for development.
Simulates API Gateway + Lambda for local frontend development.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from nasa_firms_service import get_nasa_fires
from api_handler import get_all_sensors, get_sensor_by_id, get_risk_map_data, ml_model

app = Flask(__name__)
CORS(app)

# ==================== ML API Endpoints ====================


@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    """Predict fire risk from sensor data."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        prediction = ml_model.predict(data)
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """Train a new model with historical data."""
    try:
        data = request.json or {}
        data_path = data.get('data_path', 'data/training_data.csv')

        if not os.path.isabs(data_path):
            possible_paths = [
                data_path,
                os.path.join(os.getcwd(), data_path),
                os.path.join(os.path.dirname(__file__), '..', data_path),
                os.path.join(os.path.dirname(__file__), data_path),
            ]
            found_path = next(
                (os.path.abspath(p) for p in possible_paths if os.path.exists(os.path.abspath(p))),
                None
            )
            if not found_path:
                return jsonify({'success': False, 'error': 'Data file not found'}), 404
            data_path = found_path

        result = ml_model.train(data_path)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/model/info', methods=['GET'])
def ml_model_info():
    """Get information about current loaded model."""
    return jsonify({
        'loaded': ml_model.model is not None,
        'model_type': type(ml_model.model).__name__ if ml_model.model else None,
        'model_path': ml_model.model_path,
        'features': ml_model.feature_columns,
        'use_vertex': ml_model.use_vertex
    })

@app.route('/api/ml/health', methods=['GET'])
def ml_health():
    """ML component health check."""
    return jsonify({
        'model_loaded': ml_model.model is not None,
        'model_path': ml_model.model_path,
        'use_vertex': ml_model.use_vertex
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
    """Get ML prediction for a specific sensor."""
    sensor = get_sensor_by_id(device_id)
    if not sensor:
        return jsonify({'error': 'Sensor not found'}), 404
    prediction = ml_model.predict(sensor)
    return jsonify({'success': True, 'device_id': device_id, 'prediction': prediction})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'forestshield-api',
        'ml_status': 'loaded' if ml_model.model is not None else 'no_model',
        'use_vertex': ml_model.use_vertex
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌲 ForestShield API Server Starting...")
    print("="*60)
    print("📡 http://localhost:5001")
    print("  GET  /api/sensors, /api/sensor/<id>, /api/risk-map, /api/nasa-fires")
    print("  POST /api/ml/predict  GET /api/ml/model/info  GET /api/ml/health")
    print("  POST /api/ml/train    GET /api/ml/predict/sensor/<id>")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)