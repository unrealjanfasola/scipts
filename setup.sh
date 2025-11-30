#!/usr/bin/env bash
set -euo pipefail

ONLY_WRAPPER_START=0
for arg in "$@"; do
  if [[ "$arg" == "--only-wrapper-start" ]]; then
    ONLY_WRAPPER_START=1
  fi
done

# Defaults
REPO_DIR="${REPO_DIR:-/opt/hunyuan/HunyuanVideo-1.5}"
VENV_DIR="${VENV_DIR:-/opt/hunyuan/venv-fa27}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-$(pwd)/flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$(pwd)/constraints-cu128.txt}"

if [[ "$ONLY_WRAPPER_START" -eq 0 && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required (unless --only-wrapper-start)" >&2
  exit 1
fi

if [[ "$ONLY_WRAPPER_START" -eq 0 ]]; then
  echo "== Stage 1: prerequisites =="
  chmod +x install_prereqs.sh download_ckpts.sh start_wrapper.sh generate.sh
  ./install_prereqs.sh
  python3 -m pip install --upgrade pip >/dev/null
  python3 -m pip install --upgrade "huggingface_hub[cli]" hf-transfer modelscope requests >/dev/null

  echo "== Stage 2: download ckpts =="
  REPO_DIR="$REPO_DIR" HF_TOKEN="$HF_TOKEN" ./download_ckpts.sh

  echo "== Stage 3: clone upstream repo and restore ckpts =="
  if [[ -d "$REPO_DIR" ]]; then
    echo "Repo dir exists; backing up ckpts and recloning."
    mv "$REPO_DIR/ckpts" /opt/hunyuan/ckpts
    rm -rf "$REPO_DIR"
  fi
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 "$REPO_DIR"
  mv /opt/hunyuan/ckpts "$REPO_DIR/ckpts"

  echo "== Stage 4: copy wrapper files =="
  cp hunyuan_wrapper_service.py start_wrapper.sh generate.sh generate_payload.json instr.md "$REPO_DIR/"

  echo "== Stage 5: create venv and install deps =="
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel --quiet
  pip install --index-url "$CUDA_INDEX_URL" \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 --quiet
  pip install "$FLASH_ATTN_WHL" --quiet
  sed -i 's/^torchaudio.*/torchaudio==2.7.0+cu128/' "$REPO_DIR/requirements.txt"
  pip install -c "$CONSTRAINTS_FILE" -r "$REPO_DIR/requirements.txt" --quiet
else
  echo "Skipping setup; starting wrapper only."
fi

echo "== Stage 6: start wrapper =="
REPO_DIR="$REPO_DIR" VENV_DIR="$VENV_DIR" START_BACKGROUND=1 ./start_wrapper.sh

echo "== Stage 7: health check =="
HEALTH_URL="http://localhost:${WRAPPER_PORT:-8000}/health"
health_resp=""
for attempt in {1..10}; do
  health_resp=$(curl -s --max-time 5 "$HEALTH_URL" || true)
  if echo "$health_resp" | grep -q '\"ready\": true'; then
    echo "Wrapper healthy on $HEALTH_URL"
    break
  fi
  echo "Health not ready yet (attempt $attempt); retrying..."
  sleep 3
done

if ! echo "$health_resp" | grep -q '\"ready\": true'; then
  echo "Health check failed; last response: ${health_resp:-<empty>} (log: /tmp/hy_wrapper.log)" >&2
  exit 1
fi

echo "== Done =="
echo "Repo: $REPO_DIR"
echo "Venv: $VENV_DIR"
