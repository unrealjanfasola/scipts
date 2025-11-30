#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required" >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-/opt/hunyuan/HunyuanVideo-1.5}"
mkdir -p "$REPO_DIR"
LOG=/tmp/hf_download.log
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS=1

echo "Downloading core ckpts into $REPO_DIR/ckpts ..."
HF_TOKEN="$HF_TOKEN" hf download tencent/HunyuanVideo-1.5 --repo-type model \
  --local-dir "$REPO_DIR/ckpts" --include "transformer/480p_i2v/*" \
  --include "vae/*" --include "scheduler/scheduler_config.json" \
  --include "text_encoder/**" >"$LOG" 2>&1

echo "Ensuring Glyph and Qwen LLM ckpts ..."
# Glyph lives on ModelScope; skip if repo not found
HF_TOKEN="$HF_TOKEN" hf download AI-ModelScope/Glyph-SDXL-v2 --repo-type model \
  --local-dir "$REPO_DIR/ckpts/text_encoder/Glyph-SDXL-v2" >>"$LOG" 2>&1 || \
  echo "Glyph-SDXL-v2 download skipped (repo not found); populate manually if needed" >>"$LOG"
# Qwen on HF (requires access)
HF_TOKEN="$HF_TOKEN" hf download Qwen/Qwen2.5-VL-7B-Instruct --repo-type model \
  --local-dir "$REPO_DIR/ckpts/text_encoder/llm" >>"$LOG" 2>&1 || \
  echo "Qwen2.5-VL-7B-Instruct download skipped (repo not accessible); populate manually if needed" >>"$LOG"

if [[ ! -d "$REPO_DIR/ckpts/vision_encoder/siglip" ]]; then
  echo "Downloading SigLIP ..."
  HF_TOKEN="$HF_TOKEN" hf download black-forest-labs/FLUX.1-Redux-dev --repo-type model \
    --local-dir "$REPO_DIR/ckpts/vision_encoder/siglip" >>"$LOG" 2>&1
fi

if [[ ! -d "$REPO_DIR/ckpts/text_encoder/byt5" ]]; then
  echo "Downloading byt5 ..."
  HF_TOKEN="$HF_TOKEN" hf download google/byt5-small --repo-type model \
    --local-dir "$REPO_DIR/ckpts/text_encoder/byt5" >>"$LOG" 2>&1
fi

echo "Download log: $LOG"

echo "Done. Logs: $LOG"
