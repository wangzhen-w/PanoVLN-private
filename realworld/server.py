from __future__ import annotations

import argparse
import base64
import threading
import time
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from .inference import DEFAULT_MODEL_PATH, InferenceConfig, PanoVLNPredictor
from src.eval.action_policy import DEFAULT_REPLAN_ACTION_RANGE


def _log_stage(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [realworld.server] {message}", flush=True)


class PredictionOptions(BaseModel):
    include_uncertainty: bool = False
    uncertainty_max_actions: int = Field(default=DEFAULT_REPLAN_ACTION_RANGE[1], gt=0)


class PredictJsonRequest(PredictionOptions):
    instruction: str
    images: list[str] = Field(
        ...,
        description="Base64-encoded JPEG/PNG images ordered from older to newer.",
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
            "action_sequence_length": app.state.predictor.action_sequence_length,
            "view_mode": app.state.predictor.view_mode,
            "supports_action_uncertainty": True,
        }

    @app.get("/ready")
    def ready():
        _log_stage("/ready requested")
        return {
            "ok": True,
            "model_loaded": True,
            "model_path": app.state.settings.model_path,
            "action_sequence_length": app.state.predictor.action_sequence_length,
            "view_mode": app.state.predictor.view_mode,
            "supports_action_uncertainty": True,
        }

    @app.post("/predict")
    async def predict_multipart(request: Request):
        request_start = time.perf_counter()
        async with request.form() as form:
            try:
                options = PredictionOptions(
                    include_uncertainty=form.get("include_uncertainty", False),
                    uncertainty_max_actions=form.get("uncertainty_max_actions", DEFAULT_REPLAN_ACTION_RANGE[1]),
                )
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        _log_stage(
            "/predict received "
            f"images={len(image_bytes)} instruction_chars={len(instruction)}"
        )

        try:
            with app.state.model_lock:
                result = app.state.predictor.predict(
                    instruction=instruction,
                    images=image_bytes,
                    include_uncertainty=options.include_uncertainty,
                    uncertainty_max_actions=options.uncertainty_max_actions,
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
            "uncertainty_actions": result.uncertainty_actions,
            "action_uncertainties": result.action_uncertainties,
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
                    include_uncertainty=payload.include_uncertainty,
                    uncertainty_max_actions=payload.uncertainty_max_actions,
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
            "uncertainty_actions": result.uncertainty_actions,
            "action_uncertainties": result.action_uncertainties,
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
