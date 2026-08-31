from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .corpus import PDQChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_/+][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class SearchResult:
    chunk: PDQChunk
    score: float
    rank: int


class BM25Retriever:
    """A lightweight local retriever over the NCI PDQ chunk corpus."""

    def __init__(self, chunks: Sequence[PDQChunk]):
        if not chunks:
            raise ValueError("At least one PDQ chunk is required.")
        self.chunks = tuple(chunks)
        tokenized_corpus = [tokenize(chunk.search_text) for chunk in self.chunks]
        self._token_sets = tuple(set(tokens) for tokens in tokenized_corpus)
        self.index = BM25Okapi(tokenized_corpus)

    def search(self, question: str, top_k: int = 6) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)

        scores = np.asarray(self.index.get_scores(query_tokens), dtype=float)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for rank, index in enumerate(ranked_indices, start=1):
            score = float(scores[int(index)])
            # BM25 may produce a negative score for a matching term in a very
            # small corpus. Token overlap is therefore a safer relevance gate
            # than requiring a positive score.
            if not query_token_set.intersection(self._token_sets[int(index)]):
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[int(index)],
                    score=score,
                    rank=rank,
                )
            )
        return results


def format_evidence(
    results: Sequence[SearchResult],
    *,
    max_chars_per_chunk: int = 1_800,
) -> str:
    if not results:
        return "No matching NCI PDQ passage was retrieved."

    passages: list[str] = []
    for result in results:
        chunk = result.chunk
        cancer = chunk.cancer_type or "General cancer topic"
        text = chunk.text[:max_chars_per_chunk]
        passages.append(
            "\n".join(
                (
                    f"[Evidence {result.rank}]",
                    f"Title: {chunk.title}",
                    f"Cancer/topic: {cancer} / {chunk.topic}",
                    f"Section: {' > '.join(chunk.section_path)}",
                    f"Passage: {text}",
                )
            )
        )
    return "\n\n".join(passages)
