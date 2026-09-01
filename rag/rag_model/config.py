from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDQ_CANCERS_DIR = PROJECT_ROOT / "data" / "nci_pdq" / "cancers"
PDQ_GENERAL_TOPICS_DIR = PROJECT_ROOT / "data" / "nci_pdq" / "general_topics"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


@dataclass(frozen=True)
class LLMSettings:
    """Connection settings for an OpenAI-compatible chat-completions API."""

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 120.0

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            model=os.getenv("LLM_MODEL", DEFAULT_OPENAI_MODEL),
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        )

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("LLM base URL is required.")
        if not self.model.strip():
            raise ValueError("LLM model is required.")
