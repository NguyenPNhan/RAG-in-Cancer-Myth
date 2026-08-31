from __future__ import annotations

import json
from pathlib import Path

from rag.config import load_config
from rag.normalize_pdq import build_chunks


def main():
    cfg = load_config("config.yaml")
    c = cfg["chunking"]

    chunks = build_chunks(
        raw_dir=cfg["paths"]["pdq_raw_dir"],
        target_tokens=c["target_tokens"],
        max_tokens=c["max_tokens"],
        overlap_tokens=c["overlap_tokens"],
    )

    output = Path(cfg["paths"]["chunks_file"])
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {output}")

    if chunks:
        print("\nExample chunk:")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
