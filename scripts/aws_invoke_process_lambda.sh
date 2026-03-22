
#!/usr/bin/env bash
# Invoke the deployed process_sensor_data Lambda with a payload matching MQTT JSON.
# Same Python handler as IoT Core; event shape matches what the rule passes for a JSON body.
#
# Usage:
#   export AWS_PROFILE=GreenGuard   # or rely on default credentials
#   export AWS_REGION=us-east-1
#   export PROCESS_LAMBDA_NAME=wildfire-process-sensor-data   # or wildfire-process-sensor-data-staging
#   ./scripts/aws_invoke_process_lambda.sh
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
FUNC="${PROCESS_LAMBDA_NAME:-}"

if [[ -z "$FUNC" ]]; then
  echo "Set PROCESS_LAMBDA_NAME (e.g. wildfire-process-sensor-data or wildfire-process-sensor-data-staging)" >&2
  exit 1
fi

TMPDIR="$(mktemp -d)"
PAYLOAD="$TMPDIR/payload.json"
OUT="$TMPDIR/out.json"

DEVICE_ID="${TEST_DEVICE_ID:-esp32-wildfire-sensor-1}"
TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat >"$PAYLOAD" <<EOF
{
  "deviceId": "${DEVICE_ID}",
  "temperature": 37.5,
  "humidity": 22.0,
  "lat": 43.55,
  "lng": -79.65,
  "timestamp": "${TS}"
}
EOF

echo "Invoking ${FUNC} (${REGION}) with deviceId=${DEVICE_ID} ..."
aws lambda invoke \
  --function-name "$FUNC" \
  --region "$REGION" \
  --cli-binary-format raw-in-base64-out \
  --payload "file://$PAYLOAD" \
  "$OUT"

echo "Response file:"
cat "$OUT"
echo ""
rm -rf "$TMPDIR"

echo ""
echo "Next: confirm DynamoDB item and dashboard API, e.g."
echo "  curl -s \"\${API_BASE_URL}/sensors\" | jq ."
