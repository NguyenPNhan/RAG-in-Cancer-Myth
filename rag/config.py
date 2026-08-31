from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = path.parent.resolve()

    for key in ("pdq_raw_dir", "chunks_file", "qdrant_dir", "benchmark_file", "benchmark_output"):
        value = cfg["paths"].get(key)
        if value:
            cfg["paths"][key] = str((base / value).resolve())

    return cfg
