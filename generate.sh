#!/usr/bin/env bash
set -euo pipefail

PAYLOAD="${1:-generate_payload.json}"
MODEL_PATH="${MODEL_PATH:-ckpts}"

cd /opt/hunyuan/HunyuanVideo-1.5
if [[ ! -f "$PAYLOAD" ]]; then
  echo "Payload file not found: $PAYLOAD" >&2
  exit 1
fi

curl -s -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d @"$PAYLOAD" \
  -d "{\"model_path\": \"${MODEL_PATH}\"}"
