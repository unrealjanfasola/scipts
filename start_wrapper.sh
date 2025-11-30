#!/usr/bin/env bash
set -euo pipefail

# Stop any existing wrapper
pkill -f hunyuan_wrapper_service.py || true

cd "${REPO_DIR:-/opt/hunyuan/HunyuanVideo-1.5}"
source "${VENV_DIR:-/opt/hunyuan/venv-fa27}/bin/activate"

export WRAPPER_PORT="${WRAPPER_PORT:-8000}"
# Default to ckpts dir so transformer/vae paths resolve
export HUNYUAN_MODEL_PATH="${HUNYUAN_MODEL_PATH:-ckpts}"
export HUNYUAN_OUTPUT_DIR="${HUNYUAN_OUTPUT_DIR:-/opt/hunyuan/HunyuanVideo-1.5/outputs}"

echo "Starting wrapper on port ${WRAPPER_PORT} with model_path=${HUNYUAN_MODEL_PATH}"
if [[ "${START_BACKGROUND:-0}" == "1" ]]; then
  nohup python hunyuan_wrapper_service.py >/tmp/hy_wrapper.log 2>&1 &
  echo "Wrapper started in background (log: /tmp/hy_wrapper.log)"
else
  python hunyuan_wrapper_service.py
fi
