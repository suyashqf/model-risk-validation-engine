from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class OllamaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaClient:
    model: str = "phi3:latest"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    timeout_seconds: float = 90.0

    def generate(self, system_prompt: str, user_payload: dict[str, Any]) -> str:
        prompt = (
            f"[SYSTEM]\n{system_prompt.strip()}\n\n"
            f"[STRUCTURED VALIDATION CONTEXT JSON]\n{json.dumps(user_payload, indent=2, sort_keys=True)}"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "num_ctx": 4096,
                "num_predict": 240,
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise OllamaUnavailableError(
                "Local Ollama LLM is required for report generation but is not reachable at "
                f"{self.endpoint}. Start Ollama and make sure model '{self.model}' is available."
            ) from exc

        text = str(parsed.get("response", "")).strip()
        if not text:
            raise OllamaUnavailableError("Local Ollama returned an empty report section.")
        return text
