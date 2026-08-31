from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import LLMSettings
from .llm import OpenAICompatibleBooleanClient
from .prompts import render_prompt
from .retriever import BM25Retriever, SearchResult, format_evidence


@dataclass(frozen=True)
class ClassificationResult:
    value: bool
    evidence: tuple[SearchResult, ...]

    @property
    def text(self) -> str:
        return "true" if self.value else "false"


class CancerMythRAG:
    def __init__(self, retriever: BM25Retriever, top_k: int = 6):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.retriever = retriever
        self.top_k = top_k

    def classify(
        self,
        question: str,
        prompt_key: str,
        llm_settings: LLMSettings,
    ) -> ClassificationResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        prompt = render_prompt(prompt_key, question)
        evidence: Sequence[SearchResult] = self.retriever.search(
            question,
            top_k=self.top_k,
        )
        evidence_text = format_evidence(evidence)
        client = OpenAICompatibleBooleanClient(llm_settings)
        value = client.classify(prompt, evidence_text)
        return ClassificationResult(value=value, evidence=tuple(evidence))
