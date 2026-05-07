"""Thin wrapper around an OpenAI-compatible chat/completions endpoint.

All VDE-Bench data-generation scripts call this helper so that credentials
and endpoints are specified **once** via environment variables (or CLI flags)
instead of being hard-coded in individual scripts.

Environment variables
---------------------
LLM_API_BASE : full URL of the endpoint (up to and including ``/v1`` or
               ``/llmproxy``).  The ``/chat/completions`` suffix is appended
               automatically.
LLM_API_KEY  : bearer token / API key.

Example
-------
>>> from llm_client import LLMClient, image_to_base64
>>> client = LLMClient()                      # picks up env vars
>>> img_b64 = image_to_base64("page.png")
>>> img, text = client.edit_image(
...     image_b64=img_b64,
...     prompt="Delete the sub-header.",
...     model="gemini-3-pro-image",
... )
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
from io import BytesIO
from typing import Optional, Tuple

import requests
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_base64(path: str) -> str:
    """Read an image from disk and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _as_data_url(image_b64: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{image_b64}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal OpenAI-compatible chat-completions client.

    Supports both text-only and image-input requests.  Additionally exposes
    an ``edit_image`` helper that understands the vendor extension used by
    image-editing models (returns both a textual response and a generated
    image embedded as a data-URL).
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    DEFAULT_IMAGE_EDIT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Your task is to generate a new image "
        "that satisfies the user's editing request."
    )

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        api_base = api_base or os.environ.get("LLM_API_BASE")
        api_key = api_key or os.environ.get("LLM_API_KEY")
        if not api_base or not api_key:
            raise RuntimeError(
                "LLM_API_BASE and LLM_API_KEY must be set (via environment "
                "variables or constructor arguments)."
            )

        # Normalise endpoint URL.
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        self.url = api_base
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Low-level primitive
    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> Optional[dict]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            self.url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            print(f"[LLMClient] HTTP {response.status_code}: {response.text[:300]}")
            return None
        return response.json()

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 10,
        retry_delay: float = 2.0,
        **extra_payload,
    ) -> Optional[str]:
        """Send a text-only prompt and return the assistant's reply content."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            "temperature": temperature,
            **extra_payload,
        }

        for attempt in range(1, max_retries + 1):
            try:
                result = self._post(payload)
                if result is not None:
                    return result["choices"][0]["message"]["content"]
            except Exception:
                traceback.print_exc()
            if attempt < max_retries:
                time.sleep(retry_delay)
        return None

    # ------------------------------------------------------------------
    # Image + text -> text
    # ------------------------------------------------------------------
    def chat_with_images(
        self,
        prompt: str,
        image_b64_list,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 10,
        retry_delay: float = 2.0,
    ) -> Optional[str]:
        """Ask the model a question about one or more images."""
        content = [{"type": "text", "text": prompt}]
        for b64 in image_b64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": _as_data_url(b64)},
            })

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
        }

        for attempt in range(1, max_retries + 1):
            try:
                result = self._post(payload)
                if result is not None:
                    return result["choices"][0]["message"]["content"]
            except Exception:
                traceback.print_exc()
            if attempt < max_retries:
                time.sleep(retry_delay)
        return None

    # ------------------------------------------------------------------
    # Image + text -> (image, text)   (image-editing models)
    # ------------------------------------------------------------------
    def edit_image(
        self,
        image_b64: str,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Call an image-editing model.

        Returns ``(edited_image, text_response)``.  Either may be ``None`` if
        the request failed / the model didn't return that modality.

        This follows a common multimodal-response schema where each content
        entry carries a ``type`` of either ``"text"`` or a vendor-specific
        image field.  We look for any of the known image-field names.
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or self.DEFAULT_IMAGE_EDIT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _as_data_url(image_b64)},
                        },
                    ],
                },
            ],
        }
        try:
            result = self._post(payload)
            if result is None:
                return None, None

            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, list):
                return None, content if isinstance(content, str) else None

            text_parts = []
            image_data_url = None
            for entity in content:
                etype = entity.get("type", "")
                if etype == "text":
                    text_parts.append(entity.get("text", ""))
                else:
                    # Try common multimodal-image field names
                    payload_field = entity.get(etype)
                    if isinstance(payload_field, dict) and "url" in payload_field:
                        image_data_url = payload_field["url"]

            text_response = "\n".join(p.strip() for p in text_parts if p).strip() or None

            if image_data_url is None:
                return None, text_response

            _, encoded = image_data_url.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
            return img, text_response

        except Exception:
            traceback.print_exc()
            return None, None


__all__ = ["LLMClient", "image_to_base64"]
