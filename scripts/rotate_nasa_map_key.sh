#!/usr/bin/env bash
# Rotate NASA FIRMS MAP key on the processing Lambda without printing the new key.
#
# 1) Request a new key at https://firms.modaps.eosdis.nasa.gov/api/area/ (NASA account).
# 2) Export it (do not commit):  export NEW_NASA_MAP_KEY='...'
# 3) Run:
#      export AWS_PROFILE=GreenGuard AWS_REGION=us-east-1
#      export PROCESS_LAMBDA_NAME=wildfire-process-sensor-data
#      ./scripts/rotate_nasa_map_key.sh
#
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
FUNC="${PROCESS_LAMBDA_NAME:-wildfire-process-sensor-data}"
NEW_KEY="${NEW_NASA_MAP_KEY:-}"

if [[ -z "$NEW_KEY" ]]; then
  echo "Set NEW_NASA_MAP_KEY to your new NASA FIRMS MAP API key (from NASA), then re-run." >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

aws lambda get-function-configuration \
  --function-name "$FUNC" \
  --region "$REGION" \
  --output json |
  jq --arg k "$NEW_KEY" '{Variables: ((.Environment.Variables // {}) + {NASA_MAP_KEY: $k})}' >"$TMP"

aws lambda update-function-configuration \
  --function-name "$FUNC" \
  --region "$REGION" \
  --environment "file://$TMP" >/dev/null

echo "Updated NASA_MAP_KEY on $FUNC ($REGION). Revoke the old key in NASA FIRMS if supported."
