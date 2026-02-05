from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, timeout_s: int = 60) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
        except requests.RequestException as e:
            raise RuntimeError(
                "Could not reach Ollama. Make sure Ollama is running (ollama serve) and OLLAMA_URL is correct."
            ) from e

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        return (data.get("response") or "").strip()
