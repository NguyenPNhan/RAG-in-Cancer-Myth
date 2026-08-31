from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import hashlib

from .cancer_mapping import canonical_cancer_family
from .chunking import chunk_section


TEXT_KEYS = ("text", "content", "body", "paragraphs", "description")
TITLE_KEYS = ("section", "heading", "title", "name")


def _stringify_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_stringify_text(v) for v in value]
        return "\n\n".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if k.lower() in TEXT_KEYS:
                s = _stringify_text(v)
                if s:
                    parts.append(s)
        return "\n\n".join(parts)
    return ""


def _get_first(d: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return default


def iter_sections(node: Any, path: list[str] | None = None) -> Iterable[tuple[list[str], str]]:
    """
    Recursively extracts semantic sections from many common crawler JSON shapes.

    Preferred input shape:
      {
        "title": "...",
        "sections": [
          {"heading": "...", "text": "...", "subsections": [...]}
        ]
      }

    If your crawler uses different field names, adapt TEXT_KEYS/TITLE_KEYS above.
    """
    path = path or []

    if isinstance(node, str):
        if node.strip():
            yield path, node.strip()
        return

    if isinstance(node, list):
        for item in node:
            yield from iter_sections(item, path)
        return

    if not isinstance(node, dict):
        return

    heading = str(_get_first(node, TITLE_KEYS, "") or "").strip()
    new_path = path + ([heading] if heading else [])

    # Emit local text once.
    local_text_parts = []
    for key in TEXT_KEYS:
        if key in node:
            text = _stringify_text(node[key])
            if text:
                local_text_parts.append(text)
    local_text = "\n\n".join(local_text_parts).strip()
    if local_text:
        yield new_path, local_text

    # Recurse through likely structural containers first.
    structural_keys = (
        "sections", "subsections", "children", "items", "content_sections",
        "summary_sections", "data"
    )
    visited = set(TEXT_KEYS) | set(TITLE_KEYS)
    for key in structural_keys:
        if key in node:
            visited.add(key)
            yield from iter_sections(node[key], new_path)

    # Generic fallback for nested dict/list values not already handled.
    for key, value in node.items():
        if key in visited:
            continue
        if isinstance(value, (dict, list)):
            yield from iter_sections(value, new_path)


def normalize_document(raw: dict, source_path: Path) -> dict:
    title = raw.get("title") or raw.get("name") or source_path.stem
    cancer_type = (
        raw.get("cancer_type")
        or raw.get("cancer")
        or raw.get("cancer_name")
        or raw.get("site")
    )
    topic = raw.get("topic") or raw.get("category") or "Unknown"
    audience = raw.get("audience") or raw.get("version") or "unknown"
    url = raw.get("url") or raw.get("source_url") or raw.get("canonical_url")
    last_updated = raw.get("last_updated") or raw.get("updated_at") or raw.get("date_updated")

    doc_key = f"{source_path.resolve()}::{title}::{audience}"
    document_id = hashlib.sha1(doc_key.encode("utf-8")).hexdigest()[:20]

    return {
        "document_id": document_id,
        "source": raw.get("source") or "NCI",
        "collection": raw.get("collection") or "PDQ",
        "title": str(title),
        "cancer_type": cancer_type,
        "cancer_family": canonical_cancer_family(cancer_type),
        "topic": str(topic),
        "audience": str(audience),
        "url": url,
        "last_updated": last_updated,
        "source_file": str(source_path),
    }


def build_chunks(
    raw_dir: str | Path,
    target_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[dict]:
    raw_dir = Path(raw_dir)
    chunks: list[dict] = []

    for path in sorted(raw_dir.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Could not parse {path}: {e}")
            continue

        # A file may contain one document or a list of documents.
        docs = raw if isinstance(raw, list) else [raw]

        for raw_doc in docs:
            if not isinstance(raw_doc, dict):
                continue

            meta = normalize_document(raw_doc, path)

            # Prefer structured sections if present; otherwise recurse over the doc.
            root = raw_doc.get("sections", raw_doc)
            emitted = 0

            for section_path, section_text in iter_sections(root):
                if not section_text.strip():
                    continue

                section_title = section_path[-1] if section_path else meta["title"]
                section_chunks = chunk_section(
                    section_text,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )

                for i, text in enumerate(section_chunks):
                    chunk_key = (
                        f'{meta["document_id"]}::{" > ".join(section_path)}::{i}::{text[:120]}'
                    )
                    chunk_id = hashlib.sha1(chunk_key.encode("utf-8")).hexdigest()[:24]

                    chunks.append({
                        **meta,
                        "chunk_id": chunk_id,
                        "section": section_title,
                        "section_path": section_path,
                        "chunk_index": i,
                        "text": text,
                    })
                    emitted += 1

            if emitted == 0:
                print(f"[WARN] No text extracted from {path}")

    return chunks
