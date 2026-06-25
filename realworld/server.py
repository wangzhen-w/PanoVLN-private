from __future__ import annotations

import argparse
import base64
import os
import threading
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .inference import DEFAULT_MODEL_PATH, InferenceConfig, PanoVLNPredictor


class PredictJsonRequest(BaseModel):
    instruction: str
    images: list[str] = Field(
        ...,
        description="Base64-encoded JPEG/PNG images ordered from older to newer.",
    )


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def settings_from_env() -> InferenceConfig:
    return InferenceConfig(
        model_path=_env("VLN_MODEL_PATH", DEFAULT_MODEL_PATH),
        panovggt_checkpoint_path=_env("VLN_PANOVGGT_CHECKPOINT"),
        attn_implementation=_env("VLN_ATTN_IMPLEMENTATION", "flash_attention_2"),
        max_memory_images=_env_int("VLN_MAX_MEMORY_IMAGES", 10),
        memory_pool_window_frames=_env_int("VLN_MEMORY_POOL_WINDOW_FRAMES", 100),
    )


def create_app(settings: InferenceConfig | None = None) -> FastAPI:
    app = FastAPI(title="PanoVLN Real-world Server")
    app.state.settings = settings or settings_from_env()
    app.state.predictor = None
    app.state.model_lock = threading.RLock()

    def predictor() -> PanoVLNPredictor:
        with app.state.model_lock:
            if app.state.predictor is None:
                app.state.predictor = PanoVLNPredictor(app.state.settings)
        return app.state.predictor

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_loaded": app.state.predictor is not None,
            "model_path": app.state.settings.model_path,
        }

    @app.get("/ready")
    def ready():
        predictor()
        return {
            "ok": True,
            "model_loaded": True,
            "model_path": app.state.settings.model_path,
        }

    @app.post("/predict")
    async def predict_multipart(request: Request):
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

        try:
            with app.state.model_lock:
                result = predictor().predict(instruction=instruction, images=image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "actions": result.actions,
            "executable_actions": result.executable_actions,
            "raw_text": result.raw_text,
            "prompt_images": result.prompt_images,
            "latency_s": result.latency_s,
        }

    @app.post("/predict_json")
    def predict_json(payload: PredictJsonRequest):
        if not payload.instruction.strip():
            raise HTTPException(status_code=400, detail="instruction must be non-empty")
        if not payload.images:
            raise HTTPException(status_code=400, detail="images must be non-empty")

        try:
            image_bytes = [base64.b64decode(image) for image in payload.images]
            with app.state.model_lock:
                result = predictor().predict(instruction=payload.instruction, images=image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "actions": result.actions,
            "executable_actions": result.executable_actions,
            "raw_text": result.raw_text,
            "prompt_images": result.prompt_images,
            "latency_s": result.latency_s,
        }

    return app


app = create_app()


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
    server_app = create_app(settings)
    print({"server_settings": asdict(settings)}, flush=True)
    uvicorn.run(server_app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
