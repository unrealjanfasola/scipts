# Hunyuan Wrapper Smoke (Local)

Quick steps to run the warm wrapper service and exercise a generate call.

## Prereqs
- Checkpoints mounted/available at `hunyuanvideo-1.5/ckpts` (vision encoder/SigLIP, Glyph, Qwen, byt5, vae, transformer, scheduler).
- Python env from the image (or host venv) with torch 2.7.0+cu128 and flash-attn installed.
- If `ckpts/transformer` is missing, fetch it before starting:
  ```
  HF_TOKEN="$HF_TOKEN" HF_HUB_ENABLE_HF_TRANSFER=0 \
  hf download tencent/HunyuanVideo-1.5 --repo-type model \
    --local-dir hunyuanvideo-1.5/ckpts \
    --include "transformer/480p_i2v/*"
  ```
- Make sure the wrapper files exist on the host: `hunyuan_wrapper_service.py` (and this `instr.md`) must be in the repo root. If you cloned upstream fresh, copy them over from your local workspace before starting.
- If health reports missing `ckpts/text_encoder/byt5`, pull it explicitly:
  ```
  HF_TOKEN="$HF_TOKEN" HF_HUB_ENABLE_HF_TRANSFER=0 \
  hf download google/byt5-small --repo-type model \
    --local-dir hunyuanvideo-1.5/ckpts/text_encoder/byt5
  ```
- To reduce noisy download logs, you can set `HF_HUB_DISABLE_PROGRESS_BARS=1 HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS=1` and redirect downloads to a log:
  ```
  HF_TOKEN="$HF_TOKEN" HF_HUB_ENABLE_HF_TRANSFER=0 \
  HF_HUB_DISABLE_PROGRESS_BARS=1 HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS=1 \
  hf download ... > /tmp/hf_download.log 2>&1
  ```
- Bundle repo: `https://github.com/unrealjanfasola/scipts.git`
  ```
  git clone https://github.com/unrealjanfasola/scipts.git /root/hunyuan-setup
  cd /root/hunyuan-setup
  HF_TOKEN=... ./setup.sh        # full setup + start wrapper
  ./setup.sh --only-wrapper-start  # just restart wrapper/health after setup
  ```
  Scripts inside:
  - `download_ckpts.sh` – pulls transformer/SigLIP/byt5 (uses `HF_TOKEN`, logs to `/tmp/hf_download.log`)
  - `start_wrapper.sh` – stops old wrapper, activates venv, starts service (set `HUNYUAN_MODEL_PATH=.`)
  - `generate.sh` – posts a payload file (default `generate_payload.json`) to `localhost:8000/generate`
- If you downloaded ckpts before cloning, preserve them during reclone:
  ```
  mv /opt/hunyuan/HunyuanVideo-1.5/ckpts /opt/hunyuan/ckpts && \
  rm -rf /opt/hunyuan/HunyuanVideo-1.5 && \
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 /opt/hunyuan/HunyuanVideo-1.5 && \
  mv /opt/hunyuan/ckpts /opt/hunyuan/HunyuanVideo-1.5/ckpts
  ```

## Start the service
```bash
cd hunyuanvideo-1.5
WRAPPER_PORT=8000 \
HUNYUAN_MODEL_PATH=. \
HUNYUAN_OUTPUT_DIR=outputs \
python hunyuan_wrapper_service.py
```
Service listens on `http://0.0.0.0:8000`.
Tip: start in a clean shell (no long command chains) to avoid silent failures; check `/tmp/hy_wrapper.log` if health is false. Ensure you run from the repo root so relative ckpt paths resolve.
Avoid nohup/long chains here; start in the foreground and use a separate shell/tab to issue requests. Kill any old wrapper before restarting: `pkill -f hunyuan_wrapper_service.py`.
If you must background it, explicitly `&` the command (or run `bash /tmp/start_wrapper.sh &`) so your SSH session doesn’t hang. Foreground start is preferred when debugging. Some hosts kill background jobs or block curl while the process exits; if health fails, start in the foreground to debug.
Common hiccups observed:
- Health showing missing ckpts even though they exist: set `HUNYUAN_MODEL_PATH=.` (repo root), not `ckpts`. Ensure you’re in the repo root when starting.
- Invalid JSON via inline curl over SSH: create a payload file on host and `-d @generate_payload.json` instead of embedding JSON in the command line.
- Long commands causing SSH to hang: avoid chaining; start service in one shell, curl from another. If you use a start script, background it explicitly (`&`) to avoid blocking the SSH session.

## Health check
```bash
curl -s http://localhost:8000/health | jq
```
`ready` should be true when ckpts are present.

## Generate (i2v example)
```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "render_request_id": "rr-test-001",
    "prompt": "a quick smoke test clip",
    "image_path": "assets/assetr.png",
    "resolution": "480p",
    "num_inference_steps": 10,
    "video_length": 8,
    "sr": false,
    "rewrite": false,
    "offloading": true,
    "group_offloading": true,
    "overlap_group_offloading": true
  }'
```
Result includes `output_path`, `checksum_sha256`, `duration_seconds`, `warnings`. Output lands at `outputs/rr-test-001.mp4`. Omit `image_path` for t2v. Increase steps/length/SR for quality if VRAM allows.
HF token: set HF_TOKEN in env (do not commit real tokens)
