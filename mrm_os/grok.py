from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class GrokUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GrokClient:
    model: str = "grok-beta"
    endpoint: str = "https://api.x.ai/v1/chat/completions"
    timeout_seconds: float = 90.0

    def generate(self, system_prompt: str, user_payload: dict[str, Any]) -> str:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise GrokUnavailableError("GROK_API_KEY environment variable is not set. Please provide the API key.")

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"[STRUCTURED VALIDATION CONTEXT JSON]\n{json.dumps(user_payload, indent=2, sort_keys=True)}"}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 800,
            "stream": False
        }
        
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_msg = exc.read().decode("utf-8")
            raise GrokUnavailableError(f"Grok API returned HTTP {exc.code}: {err_msg}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise GrokUnavailableError(
                f"Grok API is unreachable at {self.endpoint}. Error: {exc}"
            ) from exc

        try:
            text = parsed["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise GrokUnavailableError(f"Unexpected response format from Grok API: {parsed}") from exc

        if not text:
            raise GrokUnavailableError("Grok returned an empty report section.")
        return text
