"""API Handler - Local development with mock data"""
import boto3
from datetime import datetime
import os

dynamodb_endpoint = os.getenv('AWS_ENDPOINT_URL', 'http://dynamodb:8000')
dynamodb = boto3.resource('dynamodb', endpoint_url=dynamodb_endpoint, 
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'local'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'local'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

SENSORS_TABLE = 'WildfireSensorData'

def get_table(table_name):
    try:
        table = dynamodb.Table(table_name)
        table.load()
        return table
    except:
        return None

# ✅ MOCK SENSOR DATA
ONTARIO_CITIES = [
    ('SENSOR-001', 'Toronto Downtown', 43.6629, -79.3957, 72.5),
    ('SENSOR-002', 'Ottawa Parliament Hill', 45.4215, -75.6972, 38.0),
    ('SENSOR-003', 'Hamilton Downtown', 43.2557, -79.8711, 28.0),
    ('SENSOR-004', 'London Downtown', 42.9849, -81.2453, 52.0),
    ('SENSOR-005', 'Windsor Downtown', 42.3149, -83.0364, 65.0),
    ('SENSOR-006', 'Kitchener Downtown', 43.4516, -80.4925, 35.0),
    ('SENSOR-007', 'Niagara Falls Downtown', 43.0896, -79.0849, 25.0),
    ('SENSOR-008', 'Sudbury Downtown', 46.4917, -80.9930, 42.0),
    ('SENSOR-009', 'Timmins Downtown', 48.4758, -81.3304, 32.0),
    ('SENSOR-010', 'Thunder Bay Downtown', 48.3809, -89.2477, 28.0),
]

def calculate_fire_spread_direction(wind_direction):
    """Fire spreads opposite to wind direction"""
    return (wind_direction + 180) % 360

def create_sensor(device_id, name, lat, lng, risk_score):
    """Create sensor with mock data"""
    return {
        'id': device_id.lower().replace('-', '_'),
        'deviceId': device_id,
        'name': name,
        'lat': lat,
        'lng': lng,
        'temperature': 25.0,
        'humidity': 50.0,
        'wind_speed': 12.0,
        'wind_direction': 180.0,
        'fire_spread_direction': calculate_fire_spread_direction(180.0),
        'vegetation_density': 0.70,
        'soil_moisture': 0.35,
        'elevation': 300,
        'nearest_water': 5,
        'fire_history': 3,
        'population_density': 50,
        'riskScore': risk_score,
        'timestamp': datetime.utcnow().isoformat(),
        'nearestFireDistance': 15.0,
    }

def get_fallback_sensors():
    """Get mock sensors"""
    return [create_sensor(did, name, lat, lng, risk) 
            for did, name, lat, lng, risk in ONTARIO_CITIES]

def get_all_sensors():
    """Get all sensors from DynamoDB or fallback to mock"""
    table = get_table(SENSORS_TABLE)
    if table:
        try:
            response = table.scan(Limit=100)
            return response.get('Items', get_fallback_sensors())
        except:
            pass
    return get_fallback_sensors()

def get_sensor_by_id(sensor_id):
    """Get a single sensor by ID - handle both formats"""
    table = get_table(SENSORS_TABLE)
    if table:
        try:
            response = table.get_item(Key={'sensorId': sensor_id})
            if 'Item' in response:
                return response['Item']
        except:
            pass
    
    # Fallback to mock sensors - handle BOTH formats!
    sensor_id_lower = sensor_id.lower().replace('-', '_')  # ✅ Normalize
    
    for sensor in get_fallback_sensors():
        if (sensor['id'] == sensor_id_lower or 
            sensor['deviceId'] == sensor_id or
            sensor['deviceId'].lower() == sensor_id_lower):  # ✅ Match either way
            return sensor
    return None

def get_risk_map_data():
    """Get data for risk map visualization"""
    return {
        'sensors': get_all_sensors(),
        'timestamp': datetime.utcnow().isoformat(),
        'bounds': {'north': 49.0, 'south': 42.0, 'east': -74.0, 'west': -90.0}
    }

def get_detailed_risk(payload):
    """Get detailed risk info for a sensor"""
    sensor_id = payload.get('sensor_id')
    sensor = get_sensor_by_id(sensor_id)
    if not sensor:
        return {'error': f'Sensor {sensor_id} not found'}
    return {
        'sensor_id': sensor_id,
        'timestamp': datetime.utcnow().isoformat(),
        'risk_score': sensor.get('riskScore', 50),
    }

def get_model_health():
    """Check if ML model is loaded"""
    try:
        from inference.predict import model
        return {'status': 'healthy', 'model_loaded': True}
    except:
        return {'status': 'degraded', 'model_loaded': False}

def get_nasa_fires():
    """Get NASA FIRMS fire data"""
    try:
        from nasa_firms_service import get_nasa_fires as get_nasa
        return get_nasa()
    except:
        return []