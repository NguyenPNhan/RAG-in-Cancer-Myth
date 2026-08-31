from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from cancer_rag.config import load_config
from cancer_rag.pipeline import CancerMythRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark_path = Path(cfg["paths"]["benchmark_file"])
    output_path = Path(cfg["paths"]["benchmark_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    subset = data[args.start:]
    if args.limit is not None:
        subset = subset[:args.limit]

    rag = CancerMythRAG(args.config)

    with output_path.open("a", encoding="utf-8") as f:
        for item in tqdm(subset, desc="Benchmark"):
            try:
                prediction = rag.answer(
                    item["question"],
                    cancer_hint=item.get("cancer"),
                )
                row = {
                    "id": item.get("id"),
                    "split": item.get("split"),
                    "cancer": item.get("cancer"),
                    "question": item.get("question"),
                    "gold_correct_answer": item.get("correct_answer"),
                    "prediction": prediction,
                }
            except Exception as e:
                row = {
                    "id": item.get("id"),
                    "error": repr(e),
                }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
