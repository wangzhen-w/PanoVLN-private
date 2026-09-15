"""Small vLLM HTTP client with resumable, content-addressed request auditing."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from dataset_create.trajectory.io_utils import atomic_json_dump, load_json


class ModelError(RuntimeError):
    pass


class ModelOutputError(ModelError):
    """The model failed to return a complete, usable JSON object."""


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def text_item(text):
    return {"type": "text", "text": text}


def media_item(path, kind="image"):
    # Keep file paths until dispatch: audit logs contain hashes, never base64 blobs.
    return {"type": kind, "path": str(path)}


def video_items(video, frames, mode, description):
    result = [text_item(description)]
    if mode == "video":
        result.append(media_item(video, "video"))
    elif mode == "frames":
        for i, frame in enumerate(frames):
            result.extend([text_item(f"Chronological frame {i}:"), media_item(frame)])
    else:
        raise ValueError(f"Unknown media mode: {mode}")
    return result


class QwenClient:
    def __init__(self, settings, cache_dir):
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        if urlparse(settings["base_url"]).hostname in {"localhost", "127.0.0.1", "::1"}:
            self.session.trust_env = False
        self.session.headers["Authorization"] = "Bearer " + os.environ.get(settings["api_key_env"], "test")
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def healthcheck(self):
        response = self.session.get(self.settings["base_url"].rstrip("/") + "/models", timeout=15)
        response.raise_for_status()
        names = [model["id"] for model in response.json().get("data", [])]
        if self.settings["name"] not in names:
            raise ModelError(f"Model {self.settings['name']} unavailable; served: {names}")
        return response.json()

    def discard_response(self, key):
        """Preserve a rejected reply for inspection without reusing it."""
        cache = self.cache_dir / f"{key}.json"
        if cache.is_file():
            cache.replace(self.cache_dir / f"{key}.invalid.json")

    def complete(self, stage, system, content, *, temperature=None, attempt=0):
        audit_content = []
        for item in content:
            if item["type"] == "text":
                audit_content.append(item)
            else:
                path = Path(item["path"])
                audit_content.append({**item, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        parameters = {"model": self.settings["name"],
                      "temperature": self.settings["temperature"] if temperature is None else temperature,
                      "max_tokens": self.settings["max_tokens"],
                      "chat_template_kwargs": {"enable_thinking": self.settings["enable_thinking"]},
                      "response_format": {"type": "json_object"}}
        for name in ("top_p", "top_k"):
            if name in self.settings:
                parameters[name] = self.settings[name]
        if any(item["type"] == "video" for item in content):
            # Our videos are already sampled along the trajectory. Explicitly
            # use the uniform decoder: Qwen's automatic fps sampler interprets
            # fps=-1 as its minimum (4 frames), not "keep every frame".
            parameters["media_io_kwargs"] = {"video": {"video_backend": "opencv", "num_frames": -1, "fps": -1}}
            parameters["mm_processor_kwargs"] = {"do_sample_frames": False}
        audit = {"stage": stage, "system": system, "content": audit_content,
                 "parameters": parameters, "attempt": attempt, "base_url": self.settings["base_url"]}
        key = digest(audit)
        cache = self.cache_dir / f"{key}.json"
        if cache.is_file():
            stored = load_json(cache)
            self.cache_hits += 1
            return stored["parsed"], key
        payload_content = []
        for item in content:
            if item["type"] == "text":
                payload_content.append(item)
            else:
                mime = mimetypes.guess_type(item["path"])[0] or "application/octet-stream"
                encoded = base64.b64encode(Path(item["path"]).read_bytes()).decode("ascii")
                kind = item["type"] + "_url"
                payload_content.append({"type": kind, kind: {"url": f"data:{mime};base64,{encoded}"}})
        payload = {**parameters, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": payload_content}]}
        started = time.monotonic()
        error = None
        for retry in range(self.settings["retries"]):
            try:
                response = self.session.post(self.settings["base_url"].rstrip("/") + "/chat/completions",
                                             json=payload, timeout=(15, self.settings["timeout_seconds"]))
                if response.status_code in (408, 429) or response.status_code >= 500:
                    raise requests.RequestException(f"HTTP {response.status_code}: {response.text[:1000]}")
                if not response.ok:
                    raise ModelError(f"HTTP {response.status_code}: {response.text[:1800]}")
                raw = response.json()
                choice = raw["choices"][0]
                if choice.get("finish_reason") != "stop":
                    atomic_json_dump({"request": audit, "response": raw, "retry": retry},
                                     self.cache_dir / f"{key}.incomplete.json")
                    raise ModelOutputError(f"Incomplete model response: {choice.get('finish_reason')}")
                value = choice["message"].get("content") or ""
                try:
                    if value.strip().startswith("```"):
                        value = value.strip().split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(value)
                except (json.JSONDecodeError, IndexError) as exc:
                    raise ModelOutputError(f"Invalid model JSON: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ModelOutputError("Model JSON must be an object")
                usage = raw.get("usage") or {}
                self.calls += 1
                for name in self.usage:
                    self.usage[name] += int(usage.get(name, 0))
                atomic_json_dump({"request": audit, "response": raw, "parsed": parsed,
                                  "elapsed_seconds": time.monotonic()-started}, cache)
                return parsed, key
            except (requests.RequestException, ValueError, KeyError, IndexError, ModelOutputError) as exc:
                error = exc
                atomic_json_dump({"request": audit, "error": str(exc), "retry": retry},
                                 self.cache_dir / f"{key}.error.json")
                if retry + 1 < self.settings["retries"]:
                    time.sleep(min(2**retry, 8))
        failure = ModelOutputError if isinstance(error, ModelOutputError) else ModelError
        raise failure(f"{stage} failed after retries: {error}")
