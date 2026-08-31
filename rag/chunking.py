from __future__ import annotations

from dataclasses import dataclass
import re


def approx_tokens(text: str) -> int:
    # Intentionally tokenizer-independent for corpus preparation.
    # For English medical prose, word-count * 1.3 is a useful rough estimate.
    words = re.findall(r"\S+", text)
    return int(len(words) * 1.3)


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n+", text)
    return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]


def chunk_section(
    text: str,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[str]:
    """
    Chunk only inside one semantic section. Never cross section boundaries.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        return "\n\n".join(current).strip()

    for paragraph in paragraphs:
        candidate = (current_text() + "\n\n" + paragraph).strip() if current else paragraph

        if current and approx_tokens(candidate) > max_tokens:
            finished = current_text()
            if finished:
                chunks.append(finished)

            # Keep a small paragraph-level overlap.
            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                p_tokens = approx_tokens(p)
                if overlap and overlap_count + p_tokens > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_count += p_tokens

            current = overlap + [paragraph]
        else:
            current.append(paragraph)

        # Flush around target size when the next paragraph would likely overshoot.
        if approx_tokens(current_text()) >= target_tokens:
            chunks.append(current_text())

            overlap: list[str] = []
            overlap_count = 0
            for p in reversed(current):
                p_tokens = approx_tokens(p)
                if overlap and overlap_count + p_tokens > overlap_tokens:
                    break
                overlap.insert(0, p)
                overlap_count += p_tokens
            current = overlap

    tail = current_text()
    if tail and (not chunks or tail != chunks[-1]):
        chunks.append(tail)

    # Remove exact duplicates created by overlap edge cases.
    deduped = []
    seen = set()
    for c in chunks:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped
