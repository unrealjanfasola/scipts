#!/usr/bin/env bash
set -euo pipefail

# Defaults
REPO_DIR="${REPO_DIR:-/opt/hunyuan/HunyuanVideo-1.5}"
VENV_DIR="${VENV_DIR:-/opt/hunyuan/venv-fa27}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-$(pwd)/flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$(pwd)/constraints-cu128.txt}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

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

echo "== Done =="
echo "Repo: $REPO_DIR"
echo "Venv: $VENV_DIR"
echo "Start wrapper (foreground):"
echo "  cd $REPO_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  WRAPPER_PORT=8000 HUNYUAN_MODEL_PATH=. HUNYUAN_OUTPUT_DIR=$REPO_DIR/outputs python hunyuan_wrapper_service.py"
echo "Or use start_wrapper.sh if you want a wrapper script."
