from __future__ import annotations

import os
import json
import re
import httpx


def parse_json_object(text: str) -> dict:
    """Parse strict JSON, with a small fallback for fenced model output."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"LLM did not return a JSON object: {text[:500]}")


class OpenAICompatibleChatClient:
    """
    Minimal client for an OpenAI-compatible /v1/chat/completions endpoint.

    Environment variables:
      LLM_BASE_URL  e.g. http://localhost:11434/v1
      LLM_API_KEY   optional for local servers
      LLM_MODEL     model name exposed by your endpoint
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    @classmethod
    def from_config(cls, cfg: dict):
        llm_cfg = cfg["llm"]

        base_url = os.getenv(llm_cfg["base_url_env"])
        model = os.getenv(llm_cfg["model_env"])
        api_key = os.getenv(llm_cfg["api_key_env"], "")

        if not base_url or not model:
            raise RuntimeError(
                "Set LLM_BASE_URL and LLM_MODEL before running the full RAG pipeline. "
                "LLM_API_KEY is optional for local servers."
            )

        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=llm_cfg.get("timeout_seconds", 120),
            temperature=llm_cfg.get("temperature", 0.0),
        )

    def _post(self, payload: dict, headers: dict) -> httpx.Response:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._post(payload, headers)

        # Some OpenAI-compatible local servers do not implement response_format.
        # Retry once without that optional field while retaining the JSON-only prompt.
        if response.status_code >= 400 and json_mode:
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            response = self._post(fallback_payload, headers)

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
