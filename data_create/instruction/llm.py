"""OpenAI-compatible Qwen client and JSON parsing."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests


def data_url(jpeg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                try:
                    value = json.loads(chunk)
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found: Optional[Tuple[int, int, Dict[str, Any]]] = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            span = end
            if found is None or span > found[1]:
                found = (index, span, value)
    if found is not None:
        return found[2]
    raise ValueError(f"Model response did not contain a JSON object: {text[:500]}")


class QwenClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        retries: int,
        disable_thinking: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.disable_thinking = bool(disable_thinking)
        self.session = requests.Session()

    def preflight(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
        if self.model not in model_ids:
            raise RuntimeError(f"Model {self.model!r} not listed by /models: {sorted(model_ids)}")
        return payload

    def chat_json(
        self,
        *,
        system: str,
        prompt: str,
        images: Sequence[Tuple[str, bytes]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, image_bytes in images:
            if str(label).strip():
                content.append({"type": "text", "text": f"Image: {label}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(image_bytes), "detail": "high"},
                }
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.retries)):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
                return extract_json_object(raw), raw
            except Exception as error:  # noqa: BLE001 - preserve API body context upstream
                last_error = error
                if attempt + 1 >= max(1, self.retries):
                    break
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
        raise RuntimeError(f"Qwen request failed after {self.retries} retries: {last_error}") from last_error
