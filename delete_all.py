import boto3
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

# Scan to get all items
print("Scanning for items to delete...")
response = table.scan()
items = response.get('Items', [])

print(f"Found {len(items)} items to delete")

# Delete each item
for item in items:
    try:
        table.delete_item(
            Key={
                'deviceId': item['deviceId'],
                'timestamp': item['timestamp']
            }
        )
        print(f"✅ Deleted {item['deviceId']} - {item['timestamp']}")
    except Exception as e:
        print(f"❌ Error deleting {item['deviceId']}: {e}")

print(f"\n✨ Deleted {len(items)} items from DynamoDB")