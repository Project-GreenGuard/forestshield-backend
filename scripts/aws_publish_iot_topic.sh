#!/usr/bin/env bash
# Full path: MQTT publish -> IoT Rule -> Lambda -> DynamoDB (same as a real device).
#
# Prereqs: credentials allowed for iot:Publish on topic wildfire/sensors/*
#
# Usage:
#   export AWS_PROFILE=GreenGuard
#   export AWS_REGION=us-east-1
#   # optional: export IOT_ENDPOINT=xxxxxx-ats.iot.us-east-1.amazonaws.com
#   ./scripts/aws_publish_iot_topic.sh
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ENDPOINT="${IOT_ENDPOINT:-}"

if [[ -z "$ENDPOINT" ]]; then
  echo "Resolving IoT data endpoint (ATS)..."
  ENDPOINT="$(aws iot describe-endpoint --endpoint-type iot:Data-ATS --region "$REGION" --query endpointAddress --output text)"
  echo "Using IOT_ENDPOINT=$ENDPOINT"
fi

TOPIC="${IOT_TOPIC:-wildfire/sensors/esp32-wildfire-sensor-1}"
DEVICE_ID="${TEST_DEVICE_ID:-esp32-wildfire-sensor-1}"
TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

TMPDIR="$(mktemp -d)"
PAYLOAD_FILE="$TMPDIR/payload.json"
printf '{"deviceId":"%s","temperature":36.0,"humidity":24.0,"lat":43.55,"lng":-79.65,"timestamp":"%s"}\n' "$DEVICE_ID" "$TS" >"$PAYLOAD_FILE"

echo "Publishing to topic: $TOPIC"
aws iot-data publish \
  --region "$REGION" \
  --endpoint-url "https://${ENDPOINT}" \
  --topic "$TOPIC" \
  --cli-binary-format raw-in-base64-out \
  --payload "fileb://$PAYLOAD_FILE"

rm -rf "$TMPDIR"
echo "Published. Rule should invoke process_sensor_data within a few seconds."
echo "Check CloudWatch Logs for the Lambda and DynamoDB table WildfireSensorData*."
