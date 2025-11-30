import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from loguru import logger

from generate import save_video
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

# Keep CUDA selection consistent with the CLI script
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))

DEFAULT_MODEL_PATH = os.getenv("HUNYUAN_MODEL_PATH", "ckpts")
DEFAULT_OUTPUT_DIR = Path(os.getenv("HUNYUAN_OUTPUT_DIR", "outputs"))
DEFAULT_ASPECT = "16:9"
ALLOWED_RESOLUTIONS = {"480p", "720p"}
ALLOWED_DTYPES = {"bf16", "fp32"}

REQUIRED_CKPTS = [
    Path("vision_encoder/siglip"),
    Path("text_encoder/Glyph-SDXL-v2"),
    Path("text_encoder/llm"),
    Path("text_encoder/byt5"),
    Path("vae"),
    Path("transformer"),
    Path("scheduler/scheduler_config.json"),
]


@dataclass
class GenerateRequest:
    prompt: str
    render_request_id: str
    negative_prompt: str = ""
    resolution: str = "480p"
    aspect_ratio: str = DEFAULT_ASPECT
    num_inference_steps: int = 50
    video_length: int = 121
    seed: int = 123
    image_path: Optional[str] = None
    output_path: Optional[str] = None
    sr: bool = True
    save_pre_sr_video: bool = False
    rewrite: bool = True
    cfg_distilled: bool = False
    sparse_attn: bool = False
    offloading: bool = False
    group_offloading: Optional[bool] = None
    overlap_group_offloading: bool = False
    dtype: str = "bf16"
    use_sageattn: bool = False
    sage_blocks_range: str = "0-53"
    enable_torch_compile: bool = False
    enable_cache: bool = False
    cache_type: str = "deepcache"
    no_cache_block_id: str = "53"
    cache_start_step: int = 11
    cache_end_step: int = 45
    total_steps: int = 50
    cache_step_interval: int = 4
    model_path: str = DEFAULT_MODEL_PATH


@dataclass
class GenerateResponse:
    output_path: str
    checksum_sha256: str
    duration_seconds: float
    transformer_version: str
    resolution: str
    task: str
    sr_enabled: bool
    warnings: List[str]
    pre_sr_output_path: Optional[str] = None


@dataclass(frozen=True)
class PipelineKey:
    resolution: str
    task: str
    cfg_distilled: bool
    sparse_attn: bool
    sr: bool
    dtype: str
    offloading: bool
    group_offloading: Optional[bool]
    overlap_group_offloading: bool


class ValidationError(Exception):
    pass


class ServiceError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


PIPELINE_CACHE: Dict[PipelineKey, HunyuanVideo_1_5_Pipeline] = {}
PIPELINE_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_ckpts(model_path: str) -> List[str]:
    missing = []
    base = Path(model_path)
    for rel in REQUIRED_CKPTS:
        candidate = base / rel
        if not candidate.exists():
            missing.append(str(rel))
    return missing


def _normalize_request(data: Dict) -> GenerateRequest:
    try:
        prompt = data["prompt"]
        render_request_id = data["render_request_id"]
    except KeyError as exc:
        raise ValidationError(f"Missing required field: {exc.args[0]}") from exc

    return GenerateRequest(
        prompt=str(prompt),
        render_request_id=str(render_request_id),
        negative_prompt=data.get("negative_prompt", ""),
        resolution=data.get("resolution", "480p"),
        aspect_ratio=data.get("aspect_ratio", DEFAULT_ASPECT),
        num_inference_steps=int(data.get("num_inference_steps", 50)),
        video_length=int(data.get("video_length", 121)),
        seed=int(data.get("seed", 123)),
        image_path=data.get("image_path"),
        output_path=data.get("output_path"),
        sr=bool(data.get("sr", True)),
        save_pre_sr_video=bool(data.get("save_pre_sr_video", False)),
        rewrite=bool(data.get("rewrite", True)),
        cfg_distilled=bool(data.get("cfg_distilled", False)),
        sparse_attn=bool(data.get("sparse_attn", False)),
        offloading=bool(data.get("offloading", False)),
        group_offloading=data.get("group_offloading"),
        overlap_group_offloading=bool(data.get("overlap_group_offloading", False)),
        dtype=data.get("dtype", "bf16"),
        use_sageattn=bool(data.get("use_sageattn", False)),
        sage_blocks_range=str(data.get("sage_blocks_range", "0-53")),
        enable_torch_compile=bool(data.get("enable_torch_compile", False)),
        enable_cache=bool(data.get("enable_cache", False)),
        cache_type=str(data.get("cache_type", "deepcache")),
        no_cache_block_id=str(data.get("no_cache_block_id", "53")),
        cache_start_step=int(data.get("cache_start_step", 11)),
        cache_end_step=int(data.get("cache_end_step", 45)),
        total_steps=int(data.get("total_steps", 50)),
        cache_step_interval=int(data.get("cache_step_interval", 4)),
        model_path=str(data.get("model_path", DEFAULT_MODEL_PATH)),
    )


def _validate_request(req: GenerateRequest) -> None:
    if not req.prompt.strip():
        raise ValidationError("prompt is required")
    if req.resolution not in ALLOWED_RESOLUTIONS:
        raise ValidationError(f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
    if req.dtype not in ALLOWED_DTYPES:
        raise ValidationError(f"dtype must be one of {sorted(ALLOWED_DTYPES)}")
    if req.sparse_attn and req.resolution != "720p":
        raise ValidationError("sparse_attn is only supported with 720p resolution")
    if req.video_length <= 0:
        raise ValidationError("video_length must be positive")
    if req.num_inference_steps <= 0:
        raise ValidationError("num_inference_steps must be positive")
    if req.image_path:
        if not Path(req.image_path).exists():
            raise ValidationError(f"image_path not found: {req.image_path}")
    missing = _check_ckpts(req.model_path)
    if missing:
        raise ServiceError(f"Missing required ckpts: {', '.join(missing)}")


def _infer_task(req: GenerateRequest) -> str:
    return "i2v" if req.image_path else "t2v"


def _build_output_path(req: GenerateRequest) -> Path:
    if req.output_path:
        return Path(req.output_path)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{req.render_request_id}.mp4"


def _get_transformer_version(req: GenerateRequest, task: str) -> str:
    return HunyuanVideo_1_5_Pipeline.get_transformer_version(
        req.resolution, task, req.cfg_distilled, False, req.sparse_attn
    )


def _get_pipeline(req: GenerateRequest, task: str) -> HunyuanVideo_1_5_Pipeline:
    key = PipelineKey(
        resolution=req.resolution,
        task=task,
        cfg_distilled=req.cfg_distilled,
        sparse_attn=req.sparse_attn,
        sr=req.sr,
        dtype=req.dtype,
        offloading=req.offloading,
        group_offloading=req.group_offloading,
        overlap_group_offloading=req.overlap_group_offloading,
    )

    with PIPELINE_LOCK:
        cached = PIPELINE_CACHE.get(key)
        if cached:
            return cached

        transformer_dtype = torch.bfloat16 if req.dtype == "bf16" else torch.float32
        transformer_version = _get_transformer_version(req, task)
        logger.info(
            f"Loading pipeline variant={transformer_version} sr={req.sr} "
            f"offload={req.offloading} group_offload={req.group_offloading} "
            f"overlap_group_offload={req.overlap_group_offloading} dtype={req.dtype}"
        )
        pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=req.model_path,
            transformer_version=transformer_version,
            create_sr_pipeline=req.sr,
            force_sparse_attn=req.sparse_attn,
            transformer_dtype=transformer_dtype,
            enable_offloading=req.offloading,
            enable_group_offloading=req.group_offloading,
            overlap_group_offloading=req.overlap_group_offloading,
        )
        PIPELINE_CACHE[key] = pipe
        return pipe


def _initialize_infer_state(req: GenerateRequest) -> None:
    initialize_infer_state(
        type(
            "Args",
            (),
            {
                "enable_cache": req.enable_cache,
                "cache_type": req.cache_type,
                "no_cache_block_id": req.no_cache_block_id,
                "cache_start_step": req.cache_start_step,
                "cache_end_step": req.cache_end_step,
                "total_steps": req.total_steps,
                "cache_step_interval": req.cache_step_interval,
                "enable_torch_compile": req.enable_torch_compile,
                "sage_blocks_range": req.sage_blocks_range,
                "use_sageattn": req.use_sageattn,
            },
        )
    )


def generate(
    req: GenerateRequest, progress_callback: Optional[Callable[[int, str], None]] = None
) -> GenerateResponse:
    _validate_request(req)
    task = _infer_task(req)
    _initialize_infer_state(req)
    pipe = _get_pipeline(req, task)
    output_path = _build_output_path(req)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0, "started")

    warnings: List[str] = []
    if req.video_length != 121:
        warnings.append("video_length != 121; best quality is 121 frames")
    if not req.rewrite:
        warnings.append("prompt rewrite disabled; quality may degrade")

    transformer_version = _get_transformer_version(req, task)
    enable_sr = req.sr
    extra_kwargs = {}
    if task == "i2v":
        extra_kwargs["reference_image"] = req.image_path

    start = time.monotonic()
    try:
        out = pipe(
            enable_sr=enable_sr,
            prompt=req.prompt,
            aspect_ratio=req.aspect_ratio,
            num_inference_steps=req.num_inference_steps,
            sr_num_inference_steps=None,
            video_length=req.video_length,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            output_type="pt",
            prompt_rewrite=req.rewrite,
            return_pre_sr_video=req.save_pre_sr_video,
            **extra_kwargs,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise ServiceError("CUDA OOM during generation", retryable=True) from exc
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Pipeline execution failed: {exc}", retryable=False) from exc

    pre_sr_path: Optional[Path] = None
    try:
        if enable_sr and hasattr(out, "sr_videos"):
            save_video(out.sr_videos, str(output_path))
            if req.save_pre_sr_video:
                base, ext = os.path.splitext(output_path)
                pre_sr_path = Path(f"{base}_before_sr{ext}")
                save_video(out.videos, str(pre_sr_path))
        else:
            save_video(out.videos, str(output_path))
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Failed to save video: {exc}", retryable=False) from exc

    duration = time.monotonic() - start
    checksum = _sha256(output_path)

    if progress_callback:
        progress_callback(100, "completed")

    return GenerateResponse(
        output_path=str(output_path),
        pre_sr_output_path=str(pre_sr_path) if pre_sr_path else None,
        checksum_sha256=checksum,
        duration_seconds=duration,
        transformer_version=transformer_version,
        resolution=req.resolution,
        task=task,
        sr_enabled=enable_sr,
        warnings=warnings,
    )


def health() -> Dict:
    missing = _check_ckpts(DEFAULT_MODEL_PATH)
    ready = not missing
    return {
        "ready": ready,
        "missing_ckpts": missing,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "cache_keys": [asdict(key) for key in PIPELINE_CACHE.keys()],
    }


class WrapperHandler(BaseHTTPRequestHandler):
    server_version = "HunyuanWrapper/0.1"

    def _json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            info = health()
            status = 200 if info.get("ready") else 503
            self._json(status, info)
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/generate"):
            self._json(404, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(content_length) or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"})
            return

        try:
            req = _normalize_request(payload)
            result = generate(req)
            self._json(200, {"result": asdict(result)})
        except ValidationError as exc:
            self._json(400, {"error": "validation", "message": str(exc)})
        except ServiceError as exc:
            self._json(
                500,
                {
                    "error": "service",
                    "message": str(exc),
                    "retryable": getattr(exc, "retryable", False),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error")
            self._json(500, {"error": "internal", "message": str(exc)})

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), WrapperHandler)
    logger.info(f"Starting wrapper service on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down wrapper service")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server(
        host=os.getenv("WRAPPER_HOST", "0.0.0.0"),
        port=int(os.getenv("WRAPPER_PORT", "8000")),
    )
