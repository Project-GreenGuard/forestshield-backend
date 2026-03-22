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
                         │              ↑
                         └── optional: GCP Cloud Run /predict (HTTPS)
                                                              ↓
API Gateway → Lambda (api_handler) ← Frontend Dashboard (never calls Cloud Run directly)
```

- **`CLOUD_RUN_PREDICT_URL`**: full URL for `POST` JSON predict (e.g. `https://….run.app/predict`). If unset, processing uses **rule-based** risk + local spread heuristic only.
- **Dashboard** uses only **`REACT_APP_API_URL`** (API Gateway / local Flask). It reads **`riskLevel`** and **`spreadRateKmh`** from DynamoDB (written by the processing Lambda).

## Components

### Lambda Functions

#### 1. Process Sensor Data (`lambda-processing/`)

- Receives MQTT messages from AWS IoT Core
- Fetches NASA FIRMS data (area API if `NASA_MAP_KEY` is set, else country CSV)
- Risk + spread: **Cloud Run** when `CLOUD_RUN_PREDICT_URL` is set; otherwise rule-based fallback
- Writes `riskScore`, **`riskLevel`**, **`spreadRateKmh`**, and related fields to DynamoDB

#### 2. API Handler (`api-gateway-lambda/`)

- Handles API Gateway requests
- Queries DynamoDB
- Returns sensor and risk map data

**Endpoints:**

- `GET /api/sensors` - List all sensors
- `GET /api/sensor/{id}` - Get sensor by ID
- `GET /api/risk-map` - Get risk map data
- `POST /api/ingest` - Local only: simulate IoT payload through `process_sensor_data` (creates DynamoDB rows with ML fields)

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

### Unit tests (enrichment / thresholds)

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

### Test on AWS (real Lambda + DynamoDB, not Docker)

Prerequisites: latest **`lambda-processing.zip`** / **`api-gateway-lambda.zip`** deployed (Terraform or console), processing Lambda env includes **`DYNAMODB_TABLE`**, optional **`CLOUD_RUN_PREDICT_URL`** and **`NASA_MAP_KEY`**.

**A — Invoke processing Lambda directly** (fastest sanity check; same handler code as IoT):

```bash
export AWS_PROFILE=GreenGuard   # or your profile
export AWS_REGION=us-east-1
export PROCESS_LAMBDA_NAME=wildfire-process-sensor-data   # add -staging if needed
chmod +x scripts/aws_invoke_process_lambda.sh
./scripts/aws_invoke_process_lambda.sh
```

Then open **CloudWatch → Log group** for that function and confirm no errors; check **DynamoDB** for a new item for `TEST_DEVICE_ID`.

**B — Publish over IoT Core** (full pipeline: MQTT → rule → Lambda):

```bash
export AWS_PROFILE=GreenGuard
export AWS_REGION=us-east-1
chmod +x scripts/aws_publish_iot_topic.sh
./scripts/aws_publish_iot_topic.sh
```

Your IAM user/role must be allowed **`iot:Publish`** on the topic. Real devices use X.509 instead.

**C — Confirm dashboard API** (API Gateway URL from Terraform output):

```bash
curl -s "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/api/sensors"
```

### Test API Endpoints (local Docker)

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
