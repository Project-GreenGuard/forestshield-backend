# ForestShield Backend

AWS Lambda functions and backend services for processing wildfire sensor data.

## Overview

This repository contains:

- Lambda function for processing IoT sensor data
- Lambda function for API Gateway endpoints
- Local development environment with Docker

## Architecture

```
ESP32 → AWS IoT Core → Lambda (process_sensor_data) → DynamoDB
                                                              ↓
API Gateway → Lambda (api_handler) ← Frontend Dashboard
```

## Components

### Lambda Functions

#### 1. Process Sensor Data (`lambda-processing/`)

- Receives MQTT messages from AWS IoT Core
- Fetches NASA FIRMS wildfire data
- Calculates risk score
- Stores enriched data in DynamoDB

#### 2. API Handler (`api-gateway-lambda/`)

- Handles API Gateway requests
- Queries DynamoDB
- Returns sensor and risk map data

**Endpoints:**

- `GET /api/sensors` - List all sensors
- `GET /api/sensor/{id}` - Get sensor by ID
- `GET /api/risk-map` - Get risk map data

## Local Development

### Prerequisites

- Docker & Docker Compose
- Python 3.11+

### Environment Configuration

Copy the environment template:

```bash
cp .env.example .env
```

The default values work for local development. No changes needed unless customizing ports or region.

### Start Local Services

```bash
docker-compose up -d
```

This starts:

- **API Server** (port 5001) - Local Flask server
- **DynamoDB Local** (port 8000)
- **Mosquitto MQTT** (ports 1883, 9001)

### Access Services

- API: http://localhost:5001
- DynamoDB Local: http://localhost:8000
- MQTT: mqtt://localhost:1883

### View Logs

```bash
docker-compose logs -f
```

### Stop Services

```bash
docker-compose down
```

## Testing

### Test API Endpoints

```bash
# Get all sensors
curl http://localhost:5001/api/sensors

# Get specific sensor
curl http://localhost:5001/api/sensor/esp32-01

# Get risk map data
curl http://localhost:5001/api/risk-map
```

## Deployment

### Package Lambda Functions

```bash
# Process sensor data Lambda
cd lambda-processing
zip -r ../lambda-processing.zip . -x "*.pyc" "__pycache__/*" "*.zip"

# API handler Lambda
cd ../api-gateway-lambda
zip -r ../api-gateway-lambda.zip . -x "*.pyc" "__pycache__/*" "*.zip"
```

Then move zip files to `forestshield-infrastructure/` before running Terraform.

## Environment Variables

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

**Local development variables:**

- `AWS_ENDPOINT_URL=http://dynamodb:8000` (DynamoDB Local)
- `AWS_ACCESS_KEY_ID=local`
- `AWS_SECRET_ACCESS_KEY=local`
- `AWS_REGION=us-east-1`

**For AWS deployment:** Environment variables are set via Terraform in `forestshield-infrastructure`. Lambda functions use IAM roles for authentication.

See `TEAM_SETUP_GUIDE.md` in the `/docs` folder for detailed configuration instructions.

## Dependencies

See `requirements.txt` in each Lambda directory:

- `boto3` - AWS SDK
- `requests` - HTTP client (for NASA FIRMS API)

## Related Repositories

- **forestshield-iot-firmware** - ESP32 devices that send data
- **forestshield-infrastructure** - AWS infrastructure setup
- **forestshield-frontend** - Dashboard that consumes these APIs

## License

See LICENSE file
