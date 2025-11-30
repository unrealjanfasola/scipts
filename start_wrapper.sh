#!/usr/bin/env bash
set -euo pipefail

# Stop any existing wrapper
pkill -f hunyuan_wrapper_service.py || true

cd /opt/hunyuan/HunyuanVideo-1.5
source /opt/hunyuan/venv-fa27/bin/activate

export WRAPPER_PORT="${WRAPPER_PORT:-8000}"
export HUNYUAN_MODEL_PATH="${HUNYUAN_MODEL_PATH:-.}"
export HUNYUAN_OUTPUT_DIR="${HUNYUAN_OUTPUT_DIR:-/opt/hunyuan/HunyuanVideo-1.5/outputs}"

echo "Starting wrapper on port ${WRAPPER_PORT} with model_path=${HUNYUAN_MODEL_PATH}"
python hunyuan_wrapper_service.py
