"""
Template for retrieval evaluation.

Create a small manually annotated file such as:

data/evaluation/retrieval_gold.jsonl

{"id": 1, "question": "...", "cancer": "Lymphoma",
 "relevant_chunk_ids": ["abc", "def"]}

Then compute Recall@K / MRR independently from final generation quality.
"""

from __future__ import annotations

import json
from pathlib import Path

from cancer_rag.pipeline import CancerMythRAG


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def main(gold_path="data/evaluation/retrieval_gold.jsonl"):
    rag = CancerMythRAG("config.yaml")

    rows = [
        json.loads(line)
        for line in Path(gold_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    metrics = {"recall@5": [], "recall@10": [], "mrr": []}

    for row in rows:
        _, evidence = rag.retrieve(row["question"], row.get("cancer"))
        ids = [e["chunk_id"] for e in evidence]
        relevant = set(row["relevant_chunk_ids"])

        metrics["recall@5"].append(recall_at_k(ids, relevant, 5))
        metrics["recall@10"].append(recall_at_k(ids, relevant, 10))
        metrics["mrr"].append(reciprocal_rank(ids, relevant))

    for name, values in metrics.items():
        print(name, sum(values) / max(len(values), 1))


if __name__ == "__main__":
    main()
