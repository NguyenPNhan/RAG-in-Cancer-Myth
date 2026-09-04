from __future__ import annotations

from dataclasses import replace

from rag.rag_model.config import LLMSettings


TERRA_MODEL = "gpt-5.6-terra"


def terra_settings() -> LLMSettings:
    """Load connection settings while keeping the experiment model pinned."""
    return replace(LLMSettings.from_environment(), model=TERRA_MODEL)
