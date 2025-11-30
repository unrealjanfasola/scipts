#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

REPO_URL="${REPO_URL:-https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5}"
REPO_DIR="${REPO_DIR:-/opt/hunyuan/HunyuanVideo-1.5}"
VENV_DIR="${VENV_DIR:-/opt/hunyuan/venv-fa27}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-$REPO_ROOT/DOCS/batches/batch401script/constraints-cu128.txt}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-$REPO_ROOT/vendor/wheels/flash_attn-2.7.4.post1-0rtx5090torch270cu128cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"
HF_TOKEN="${HF_TOKEN:-}"
FLASH_ATTN_URL="${FLASH_ATTN_URL:-https://github.com/Zarrac/flashattention-blackwell-wheels-whl-ONLY-5090-5080-5070-5060-flash-attention-/releases/download/FlashAttention/flash_attn-2.7.4.post1-rtx5090-torch2.7.0cu128cxx11abiTRUE-cp310-linux_x86_64.whl}"

if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is required for checkpoint downloads." >&2
  exit 1
fi

# Reduce noise from pip/hf downloads
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_WARN_SCRIPT_LOCATION=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

for bin in python3 git; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "Missing dependency: $bin" >&2
    exit 1
  fi
done

# Optional: install docker + nvidia-container-toolkit if requested (for containerized runtime)
if [[ -n "${INSTALL_DOCKER:-}" ]]; then
  echo "Installing docker.io and nvidia-container-toolkit..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get -qq update >/dev/null
  apt-get -qq install -y docker.io nvidia-container-toolkit >/dev/null
fi

if [[ ! -f "$FLASH_ATTN_WHL" ]]; then
  echo "FlashAttention wheel not found at $FLASH_ATTN_WHL; attempting download from $FLASH_ATTN_URL"
  mkdir -p "$(dirname "$FLASH_ATTN_WHL")"
  curl -L -o "$FLASH_ATTN_WHL" "$FLASH_ATTN_URL"
fi

if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "Constraints file not found at $CONSTRAINTS_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$REPO_DIR")"
# Optional code checkout; required if you need local access to scripts/paths
if [[ -z "${CKPTS_ONLY:-}" ]]; then
  if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" pull --ff-only
  else
    git clone "$REPO_URL" "$REPO_DIR" --quiet
  fi

  echo "Aligning torch/torchvision/torchaudio pins in requirements.txt to constraints to avoid pip conflicts..."
  python3 - "$REPO_DIR/requirements.txt" "$CONSTRAINTS_FILE" <<'PY'
import re
import sys
from pathlib import Path

req_path = Path(sys.argv[1])
con_path = Path(sys.argv[2])

# Load pins from the constraints file
pins = {}
for line in con_path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if "==" in line:
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()

target_names = {"torch", "torchvision", "torchaudio"}
target_pins = {k: v for k, v in pins.items() if k in target_names}

if not target_pins:
    sys.exit(0)

pattern = re.compile(r"^(torch(?:vision|audio)?)([<>=!~].*)?$", re.IGNORECASE)
out_lines = []
changed = False
for raw_line in req_path.read_text().splitlines():
    line = raw_line.strip()
    m = pattern.match(line)
    if m:
        name = m.group(1)
        key = name.lower()
        if key in target_pins:
            new_line = f"{name}=={target_pins[key]}"
            if line != new_line:
                changed = True
            out_lines.append(new_line)
            continue
    out_lines.append(raw_line)

if changed:
    req_path.write_text("\n".join(out_lines) + "\n")
PY

  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip setuptools wheel --quiet
  pip install --index-url "$CUDA_INDEX_URL" \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 --quiet
  pip install "$FLASH_ATTN_WHL" --quiet
  pip install -c "$CONSTRAINTS_FILE" -r "$REPO_DIR/requirements.txt" --quiet

  cd "$REPO_DIR"
fi

mkdir -p "$REPO_DIR"
cd "$REPO_DIR"
mkdir -p ckpts
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" HF_TOKEN="$HF_TOKEN" \
  hf download tencent/HunyuanVideo-1.5 --repo-type model --local-dir ./ckpts --include \
    "transformer/480p_i2v/*" \
    "vae/*" \
    "scheduler/scheduler_config.json" \
    "text_encoder/**" \
    "text_encoder_2/**"

if [[ ! -d ckpts/vision_encoder/siglip ]]; then
  echo "Downloading SigLIP vision encoder from black-forest-labs/FLUX.1-Redux-dev..."
  hf_env=(HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" HF_TOKEN="$HF_TOKEN")
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    hf_env+=(HF_ENDPOINT="$HF_ENDPOINT")
  fi
  env "${hf_env[@]}" hf download black-forest-labs/FLUX.1-Redux-dev --repo-type model --local-dir ./ckpts/vision_encoder/siglip
fi

if [[ ! -d ckpts/vision_encoder/siglip ]]; then
  echo "Failed to fetch SigLIP vision encoder (ckpts/vision_encoder/siglip). Ensure HF_TOKEN has access to black-forest-labs/FLUX.1-Redux-dev and rerun." >&2
  exit 1
fi

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" HF_TOKEN="$HF_TOKEN" \
  hf download google/byt5-small --repo-type model --local-dir ./ckpts/text_encoder/byt5-small

HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}" HF_TOKEN="$HF_TOKEN" \
  hf download Qwen/Qwen2.5-VL-7B-Instruct --repo-type model --local-dir ./ckpts/text_encoder/llm

if [[ ! -d ckpts/text_encoder_2/Glyph-SDXL-v2 ]]; then
  if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder_2/Glyph-SDXL-v2
  else
    echo "modelscope CLI not found; Glyph-SDXL-v2 not downloaded." >&2
  fi
fi

if [[ -d ckpts/text_encoder_2/Glyph-SDXL-v2 && ! -e ckpts/text_encoder/Glyph-SDXL-v2 ]]; then
  ln -s ../text_encoder_2/Glyph-SDXL-v2 ckpts/text_encoder/Glyph-SDXL-v2
fi

echo "Repo synced to $REPO_DIR"
echo "Venv ready at $VENV_DIR"
echo "Checkpoints downloaded under $REPO_DIR/ckpts"
