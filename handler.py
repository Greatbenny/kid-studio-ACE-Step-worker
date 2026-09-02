import base64
import gc
import hashlib
import os
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import runpod
import torch

SERVICE = "kid-studio-ace-step-worker"
WORKER_BUILD = "ace-step-1.5-v1"
UPSTREAM_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
MODEL_LICENSE = "MIT"

PROJECT_ROOT = Path(
    os.getenv("ACESTEP_PROJECT_ROOT", "/opt/ace-step")
)
CHECKPOINTS = Path(
    os.getenv(
        "ACESTEP_CHECKPOINTS_DIR",
        "/runpod-volume/ace-step/checkpoints",
    )
)
TMP_ROOT = Path(os.getenv("TMPDIR", "/runpod-volume/tmp"))

QUALITY_MODELS = {
    "cheap": ("acestep-v15-turbo", 8),
    "balanced": ("acestep-v15-turbo", 12),
    "high": ("acestep-v15-xl-turbo", 8),
}

_dit_handler: Any = None
_loaded_config: str | None = None
_init_status: str | None = None


def _ensure_storage() -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_bytes": props.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def _storage() -> dict[str, Any]:
    _ensure_storage()
    try:
        stat = os.statvfs("/runpod-volume")
        return {
            "root": "/runpod-volume",
            "checkpoints": str(CHECKPOINTS),
            "free_bytes": stat.f_bavail * stat.f_frsize,
            "total_bytes": stat.f_blocks * stat.f_frsize,
        }
    except OSError:
        return {
            "root": None,
            "checkpoints": str(CHECKPOINTS),
            "free_bytes": None,
            "total_bytes": None,
        }


def _health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": SERVICE,
        "worker_build": WORKER_BUILD,
        "upstream_commit": UPSTREAM_COMMIT,
        "model_license": MODEL_LICENSE,
        "loaded_model": _loaded_config,
        "init_status": _init_status,
        "quality_models": {
            quality: model
            for quality, (model, _) in QUALITY_MODELS.items()
        },
        "gpu": _gpu(),
        "storage": _storage(),
    }


def _unload() -> None:
    global _dit_handler, _loaded_config, _init_status
    _dit_handler = None
    _loaded_config = None
    _init_status = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_model(config_path: str) -> Any:
    global _dit_handler, _loaded_config, _init_status

    if _dit_handler is not None and _loaded_config == config_path:
        return _dit_handler
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    _unload()
    _ensure_storage()

    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    status, success = handler.initialize_service(
        project_root=str(PROJECT_ROOT),
        config_path=config_path,
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        prefer_source="huggingface",
    )
    if not success:
        raise RuntimeError(status)

    _dit_handler = handler
    _loaded_config = config_path
    _init_status = status
    return handler


def _integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
    default: int,
) -> int:
    if value is None:
        return default
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if result < minimum or result > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return result


def _optional_bpm(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _integer(value, "bpm", 30, 300, 120)


def _seed(value: Any) -> int:
    if value is None:
        return secrets.randbelow(2**31)
    return _integer(value, "seed", 0, 2**31 - 1, 0)


def _clean_text(
    value: Any,
    name: str,
    maximum: int,
    required: bool = False,
) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name} is required.")
    if len(result) > maximum:
        raise ValueError(
            f"{name} cannot exceed {maximum} characters."
        )
    return result


def _generate(data: dict[str, Any]) -> dict[str, Any]:
    quality = str(data.get("quality") or "cheap").strip().lower()
    if quality not in QUALITY_MODELS:
        raise ValueError(
            "quality must be cheap, balanced, or high."
        )
    config_path, default_steps = QUALITY_MODELS[quality]

    caption = _clean_text(
        data.get("caption") or data.get("prompt"),
        "caption",
        512,
        required=True,
    )
    instrumental = bool(data.get("instrumental", False))
    lyrics = _clean_text(data.get("lyrics"), "lyrics", 4096)
    if instrumental:
        lyrics = "[Instrumental]"
    elif not lyrics:
        raise ValueError(
            "lyrics are required unless instrumental=true."
        )

    duration = _integer(
        data.get("duration_seconds") or data.get("duration"),
        "duration_seconds",
        10,
        240,
        30,
    )
    bpm = _optional_bpm(data.get("bpm"))
    seed = _seed(data.get("seed"))
    vocal_language = _clean_text(
        data.get("vocal_language") or "unknown",
        "vocal_language",
        32,
        required=True,
    )
    keyscale = _clean_text(data.get("keyscale"), "keyscale", 32)
    timesignature = _clean_text(
        data.get("timesignature"), "timesignature", 8
    )
    steps = _integer(
        data.get("inference_steps"),
        "inference_steps",
        1,
        20,
        default_steps,
    )

    from acestep.inference import (
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    dit_handler = _load_model(config_path)
    job_dir = TMP_ROOT / (
        "ace-step-" + secrets.token_hex(12)
    )
    job_dir.mkdir(parents=True, exist_ok=False)

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=instrumental,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        duration=float(duration),
        inference_steps=steps,
        seed=seed,
        guidance_scale=1.0,
        shift=3.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_lyrics=False,
        use_cot_language=False,
        use_constrained_decoding=False,
    )
    config = GenerationConfig(
        batch_size=1,
        allow_lm_batch=False,
        use_random_seed=False,
        seeds=[seed],
        audio_format="mp3",
    )

    try:
        started = time.perf_counter()
        generated = generate_music(
            dit_handler,
            None,
            params,
            config,
            save_dir=str(job_dir),
        )
        inference_ms = round(
            (time.perf_counter() - started) * 1000
        )
        if not generated.success or not generated.audios:
            raise RuntimeError(
                generated.error
                or generated.status_message
                or "ACE-Step returned no audio."
            )

        item = generated.audios[0]
        path = Path(str(item.get("path") or ""))
        if not path.is_file():
            candidates = sorted(job_dir.glob("**/*.mp3"))
            if not candidates:
                raise RuntimeError(
                    "ACE-Step output file was not found."
                )
            path = candidates[0]

        raw = path.read_bytes()
        if not raw:
            raise RuntimeError("ACE-Step returned an empty audio file.")
        if len(raw) > 9 * 1024 * 1024:
            raise RuntimeError(
                "Generated audio exceeds the inline response limit. "
                "Reduce duration below 240 seconds."
            )

        resolved_seed = item.get("params", {}).get("seed", seed)
        return {
            "ok": True,
            "service": SERVICE,
            "worker_build": WORKER_BUILD,
            "upstream_commit": UPSTREAM_COMMIT,
            "model": config_path,
            "model_license": MODEL_LICENSE,
            "quality": quality,
            "caption": caption,
            "instrumental": instrumental,
            "vocal_language": vocal_language,
            "duration_requested_seconds": duration,
            "seed": resolved_seed,
            "inference_steps": steps,
            "inference_ms": inference_ms,
            "mime_type": "audio/mpeg",
            "audio_sha256": hashlib.sha256(raw).hexdigest(),
            "audio_base64": base64.b64encode(raw).decode("ascii"),
        }
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input")
    if not isinstance(data, dict):
        return {
            "ok": False,
            "worker_build": WORKER_BUILD,
            "error": "input must be a JSON object.",
        }

    operation = str(
        data.get("operation") or "generate"
    ).strip().lower()

    try:
        if operation in {"health", "preflight"}:
            return _health()
        if operation == "unload":
            _unload()
            return {**_health(), "unloaded": True}
        if operation == "warmup":
            quality = str(
                data.get("quality") or "cheap"
            ).strip().lower()
            if quality not in QUALITY_MODELS:
                raise ValueError(
                    "quality must be cheap, balanced, or high."
                )
            config_path, _ = QUALITY_MODELS[quality]
            started = time.perf_counter()
            _load_model(config_path)
            return {
                **_health(),
                "warmed_quality": quality,
                "load_ms": round(
                    (time.perf_counter() - started) * 1000
                ),
            }
        if operation != "generate":
            raise ValueError(
                "operation must be health, preflight, warmup, "
                "unload, or generate."
            )
        return _generate(data)
    except Exception as exc:
        return {
            "ok": False,
            "service": SERVICE,
            "worker_build": WORKER_BUILD,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


if __name__ == "__main__":
    _ensure_storage()
    runpod.serverless.start({"handler": handler})
