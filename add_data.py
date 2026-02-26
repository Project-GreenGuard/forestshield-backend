import boto3
from datetime import datetime
import random
from decimal import Decimal

# Connect to DynamoDB Local
dynamodb = boto3.resource(
    'dynamodb',
    endpoint_url='http://localhost:8000',
    region_name='us-east-1',
    aws_access_key_id='dummy',
    aws_secret_access_key='dummy'
)

table = dynamodb.Table('WildfireSensorData')

# Ontario cities from your dashboard - EXPANDED with new locations
sensors = [
    # Original sensors
    {"deviceId": "S001", "location": "Brampton", "lat": 43.7315, "lng": -79.7624},
    {"deviceId": "S002", "location": "Hamilton", "lat": 43.2557, "lng": -79.8711},
    {"deviceId": "S003", "location": "Kitchener", "lat": 43.4516, "lng": -80.4925},
    {"deviceId": "S004", "location": "London", "lat": 42.9849, "lng": -81.2453},
    {"deviceId": "S005", "location": "Barrie", "lat": 44.3894, "lng": -79.6903},
    {"deviceId": "S006", "location": "Guelph", "lat": 43.5448, "lng": -80.2482},
    {"deviceId": "S007", "location": "Cambridge", "lat": 43.3616, "lng": -80.3144},
    {"deviceId": "S008", "location": "Stratford", "lat": 43.3708, "lng": -80.9822},
    {"deviceId": "S009", "location": "Woodstock", "lat": 43.1300, "lng": -80.7467},
    {"deviceId": "S010", "location": "St. Thomas", "lat": 42.7775, "lng": -81.1928},
    
    # NEW CITIES - Oakville, Toronto, Brantford, St. Catharines, Vaughan, Markham, Newmarket
    {"deviceId": "S011", "location": "Oakville", "lat": 43.4675, "lng": -79.6877},
    {"deviceId": "S012", "location": "Toronto", "lat": 43.6532, "lng": -79.3832},
    {"deviceId": "S013", "location": "Brantford", "lat": 43.1394, "lng": -80.2644},
    {"deviceId": "S014", "location": "St. Catharines", "lat": 43.1594, "lng": -79.2469},
    {"deviceId": "S015", "location": "Vaughan", "lat": 43.8083, "lng": -79.4722},
    {"deviceId": "S016", "location": "Markham", "lat": 43.8563, "lng": -79.3370},
    {"deviceId": "S017", "location": "Newmarket", "lat": 44.0593, "lng": -79.4618},
]

print("🚀 Adding sensor data to DynamoDB...\n")
print(f"📍 Total locations: {len(sensors)} cities\n")

for sensor in sensors:
    # SPECIAL HANDLING FOR BRAMPTON (S001) - TEST DIFFERENT RISK LEVELS
    if sensor['deviceId'] == 'S001':
        # ===== ADJUST THESE VALUES TO TEST BRAMPTON =====
        temp = 38.5        # ← Higher = more risk (try 40 for CRITICAL)
        humidity = 15.2     # ← Lower = more risk (try 10 for CRITICAL)
        wind = 34.8         # ← Higher = more risk (try 40 for CRITICAL)
        # ===============================================
        print(f"🔥 BRAMPTON TEST - Temp: {temp}°C, Humidity: {humidity}%, Wind: {wind} km/h")
    else:
        # Different conditions for different regions
        if sensor['location'] in ['Toronto', 'Mississauga', 'Vaughan', 'Markham']:
            # Urban areas - slightly warmer
            temp = float(round(random.uniform(20, 35), 1))
            humidity = float(round(random.uniform(25, 70), 1))
            wind = float(round(random.uniform(5, 25), 1))
        elif sensor['location'] in ['Oakville', 'Burlington', 'St. Catharines']:
            # Near Lake Ontario - cooler, more humid
            temp = float(round(random.uniform(18, 30), 1))
            humidity = float(round(random.uniform(40, 85), 1))
            wind = float(round(random.uniform(10, 30), 1))
        elif sensor['location'] in ['Brantford', 'Woodstock', 'Stratford']:
            # Inland - more variable
            temp = float(round(random.uniform(16, 32), 1))
            humidity = float(round(random.uniform(30, 80), 1))
            wind = float(round(random.uniform(5, 28), 1))
        else:
            # Default random for others
            temp = float(round(random.uniform(18, 32), 1))
            humidity = float(round(random.uniform(30, 75), 1))
            wind = float(round(random.uniform(2, 20), 1))
    
    # Calculate risk score (as Decimal)
    risk = Decimal(str(round(
        (float(temp)/40) * 0.5 + 
        (1 - float(humidity)/100) * 0.3 + 
        (float(wind)/30) * 0.2, 
    2)))
    
    # Determine risk level
    if risk >= Decimal('0.7'):
        risk_level = "HIGH"
    elif risk >= Decimal('0.3'):
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    item = {
        'deviceId': sensor['deviceId'],
        'timestamp': datetime.now().isoformat(),
        'temperature': Decimal(str(temp)),
        'humidity': Decimal(str(humidity)),
        'wind_speed': Decimal(str(wind)),
        'wind_direction': random.randint(0, 359),
        'pressure': Decimal(str(round(random.uniform(1005, 1025), 1))),
        'lat': Decimal(str(sensor['lat'])),
        'lng': Decimal(str(sensor['lng'])),
        'location': sensor['location'],
        'riskScore': risk
    }
    
    table.put_item(Item=item)
    print(f"✅ Added {sensor['location']} ({risk_level}): {float(risk)*100:.1f}/100")

print("\n✨ Sample data added successfully!")
print("\n📊 Check your data at: http://localhost:5001/api/sensors")

# Verify data was added
response = table.scan()
print(f"\n📈 Total items in table: {len(response.get('Items', []))}")

# Show Brampton specifically
print("\n🔥 BRAMPTON CURRENT SETTINGS:")
print(f"   Temperature: {temp if 'temp' in locals() else 'N/A'}°C")
print(f"   Humidity: {humidity if 'humidity' in locals() else 'N/A'}%")
print(f"   Wind Speed: {wind if 'wind' in locals() else 'N/A'} km/h")
print(f"   Risk Score: {float(risk)*100:.1f}/100 ({risk_level})")

# Show count by region
print("\n📍 CITIES ADDED BY REGION:")
gta = [s for s in sensors if s['location'] in ['Toronto', 'Mississauga', 'Brampton', 'Vaughan', 'Markham', 'Oakville', 'Newmarket']]
niagara = [s for s in sensors if s['location'] in ['Hamilton', 'St. Catharines']]
western = [s for s in sensors if s['location'] in ['London', 'Woodstock', 'Stratford', 'Brantford', 'Cambridge', 'Kitchener', 'Guelph']]
northern = [s for s in sensors if s['location'] in ['Barrie']]

print(f"   🏙️ GTA: {len(gta)} cities")
print(f"   🌊 Niagara: {len(niagara)} cities")
print(f"   🌾 Western: {len(western)} cities")
print(f"   ⛰️ Northern: {len(northern)} cities")

print("\n💡 To test different scenarios, edit the temp, humidity, wind values for Brampton (S001)!")