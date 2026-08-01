from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .inference import DEFAULT_MODEL_PATH, InferenceConfig, PanoVLNPredictor


def _log_stage(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [realworld.server] {message}", flush=True)


class PredictJsonRequest(BaseModel):
    instruction: str
    images: list[str] = Field(
        ...,
        description="Base64-encoded JPEG/PNG images ordered from older to newer.",
    )
    interframe_actions: list[list[str]] | None = Field(
        default=None,
        description="Executed action spans between consecutive uploaded images.",
    )


def create_app(settings: InferenceConfig, predictor: PanoVLNPredictor) -> FastAPI:
    app = FastAPI(title="PanoVLN Real-world Server")
    app.state.settings = settings
    app.state.predictor = predictor
    app.state.model_lock = threading.RLock()

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_loaded": True,
            "model_path": app.state.settings.model_path,
        }

    @app.get("/ready")
    def ready():
        _log_stage("/ready requested")
        return {
            "ok": True,
            "model_loaded": True,
            "model_path": app.state.settings.model_path,
        }

    @app.post("/predict")
    async def predict_multipart(request: Request):
        request_start = time.perf_counter()
        form = await request.form()
        instruction = str(form.get("instruction", "")).strip()
        if not instruction:
            raise HTTPException(status_code=400, detail="Missing form field: instruction")

        uploads = []
        for field_name in ("images", "image", "files", "file"):
            uploads.extend(form.getlist(field_name))
        image_bytes = []
        for upload in uploads:
            if hasattr(upload, "read"):
                image_bytes.append(await upload.read())
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Upload at least one image file")
        interframe_actions = None
        raw_interframe_actions = form.get("interframe_actions")
        if raw_interframe_actions is not None:
            try:
                interframe_actions = json.loads(str(raw_interframe_actions))
            except (TypeError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="interframe_actions must be a JSON array of action arrays",
                ) from exc
        _log_stage(
            "/predict received "
            f"images={len(image_bytes)} instruction_chars={len(instruction)}"
        )

        try:
            with app.state.model_lock:
                result = app.state.predictor.predict(
                    instruction=instruction,
                    images=image_bytes,
                    interframe_actions=interframe_actions,
                )
        except Exception as exc:
            _log_stage(
                "/predict failed "
                f"after {time.perf_counter() - request_start:.3f}s: {type(exc).__name__}: {exc}"
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _log_stage(
            "/predict finished "
            f"total_s={time.perf_counter() - request_start:.3f} "
            f"model_latency_s={result.latency_s:.3f} "
            f"actions={result.actions}"
        )

        return {
            "actions": result.actions,
            "executable_actions": result.executable_actions,
            "raw_text": result.raw_text,
            "prompt_images": result.prompt_images,
            "latency_s": result.latency_s,
        }

    @app.post("/predict_json")
    def predict_json(payload: PredictJsonRequest):
        request_start = time.perf_counter()
        if not payload.instruction.strip():
            raise HTTPException(status_code=400, detail="instruction must be non-empty")
        if not payload.images:
            raise HTTPException(status_code=400, detail="images must be non-empty")
        _log_stage(
            "/predict_json received "
            f"images={len(payload.images)} instruction_chars={len(payload.instruction.strip())}"
        )

        try:
            image_bytes = [base64.b64decode(image) for image in payload.images]
            with app.state.model_lock:
                result = app.state.predictor.predict(
                    instruction=payload.instruction,
                    images=image_bytes,
                    interframe_actions=payload.interframe_actions,
                )
        except Exception as exc:
            _log_stage(
                "/predict_json failed "
                f"after {time.perf_counter() - request_start:.3f}s: {type(exc).__name__}: {exc}"
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _log_stage(
            "/predict_json finished "
            f"total_s={time.perf_counter() - request_start:.3f} "
            f"model_latency_s={result.latency_s:.3f} "
            f"actions={result.actions}"
        )

        return {
            "actions": result.actions,
            "executable_actions": result.executable_actions,
            "raw_text": result.raw_text,
            "prompt_images": result.prompt_images,
            "latency_s": result.latency_s,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PanoVLN over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--panovggt-checkpoint", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--max-memory-images", type=int, default=10)
    parser.add_argument("--memory-pool-window-frames", type=int, default=100)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = InferenceConfig(
        model_path=args.model_path,
        panovggt_checkpoint_path=args.panovggt_checkpoint,
        attn_implementation=args.attn_implementation,
        max_memory_images=args.max_memory_images,
        memory_pool_window_frames=args.memory_pool_window_frames,
    )
    _log_stage(f"server_settings={asdict(settings)}")
    start = time.perf_counter()
    _log_stage("loading model before starting HTTP server")
    predictor = PanoVLNPredictor(settings)
    _log_stage(f"model ready in {time.perf_counter() - start:.2f}s; starting HTTP server")
    server_app = create_app(settings, predictor)
    uvicorn.run(server_app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
