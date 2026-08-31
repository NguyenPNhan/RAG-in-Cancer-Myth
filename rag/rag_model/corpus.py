from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Sequence


_WHITESPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class PDQChunk:
    chunk_id: str
    title: str
    cancer_type: str | None
    topic: str
    audience: str
    section: str
    section_path: tuple[str, ...]
    url: str
    source_file: str
    text: str

    @property
    def search_text(self) -> str:
        """Repeat high-value metadata so BM25 can reward exact disease matches."""
        metadata = " ".join(
            value
            for value in (
                self.title,
                self.title,
                self.cancer_type or "",
                self.cancer_type or "",
                self.topic.replace("_", " "),
                self.audience.replace("_", " "),
                self.section,
                " ".join(self.section_path),
            )
            if value
        )
        return f"{metadata}\n{self.text}"


def _clean_text(text: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _section_path(value: object, fallback: str) -> tuple[str, ...]:
    if isinstance(value, list):
        path = tuple(str(part).strip() for part in value if str(part).strip())
        return path or (fallback,)
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return (fallback,)


def _split_words(
    text: str,
    target_words: int = 350,
    overlap_words: int = 50,
) -> list[str]:
    """Split long sections without crossing their semantic section boundary."""
    words = text.split()
    if not words:
        return []
    if len(words) <= target_words:
        return [text]
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words.")

    chunks: list[str] = []
    step = target_words - overlap_words
    for start in range(0, len(words), step):
        part = words[start : start + target_words]
        if not part:
            break
        chunks.append(" ".join(part))
        if start + target_words >= len(words):
            break
    return chunks


def _chunk_id(url: str, section_path: Sequence[str], index: int, text: str) -> str:
    key = "\n".join((url, " > ".join(section_path), str(index), text))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"nci-pdq-{digest}"


def _iter_pages(raw: object) -> Iterator[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("pages"), list):
        yield from (page for page in raw["pages"] if isinstance(page, dict))
    elif isinstance(raw, list):
        yield from (page for page in raw if isinstance(page, dict))
    elif isinstance(raw, dict):
        yield raw


def load_pdq_chunks(
    source_dirs: Iterable[str | Path],
    *,
    target_words: int = 350,
    overlap_words: int = 50,
) -> list[PDQChunk]:
    """Load NCI PDQ cancer and general-topic JSON files into retrieval chunks."""
    chunks: list[PDQChunk] = []
    seen_ids: set[str] = set()

    json_paths: list[Path] = []
    for source_dir in source_dirs:
        directory = Path(source_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"PDQ data directory does not exist: {directory}")
        json_paths.extend(sorted(directory.glob("*.json")))

    if not json_paths:
        raise FileNotFoundError("No NCI PDQ JSON files were found.")

    for path in sorted(json_paths):
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        wrapper_cancer = raw.get("cancer_type") if isinstance(raw, dict) else None
        for page in _iter_pages(raw):
            title = str(page.get("title") or page.get("summary_name") or path.stem)
            cancer_type = page.get("cancer_type") or wrapper_cancer
            topic = str(page.get("topic") or "general")
            audience = str(page.get("audience") or "unspecified")
            url = str(page.get("url") or "")
            if not url:
                continue

            sections = page.get("sections")
            if not isinstance(sections, list):
                continue

            for section in sections:
                if not isinstance(section, dict) or section.get("is_boilerplate", False):
                    continue

                text = _clean_text(str(section.get("text") or ""))
                if not text:
                    continue

                section_name = str(section.get("section") or title)
                path_parts = _section_path(section.get("section_path"), section_name)
                for index, chunk_text in enumerate(
                    _split_words(
                        text,
                        target_words=target_words,
                        overlap_words=overlap_words,
                    )
                ):
                    chunk_id = _chunk_id(url, path_parts, index, chunk_text)
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    chunks.append(
                        PDQChunk(
                            chunk_id=chunk_id,
                            title=title,
                            cancer_type=str(cancer_type) if cancer_type else None,
                            topic=topic,
                            audience=audience,
                            section=section_name,
                            section_path=path_parts,
                            url=url,
                            source_file=str(path),
                            text=chunk_text,
                        )
                    )

    if not chunks:
        raise RuntimeError("The NCI PDQ files did not produce any retrieval chunks.")
    return chunks
