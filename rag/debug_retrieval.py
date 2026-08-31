from __future__ import annotations

import argparse
import json

from rag.pipeline import CancerMythRAG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    rag = CancerMythRAG(args.config)
    analysis, evidence = rag.retrieve(args.question, cancer_hint=args.cancer)

    print("QUERY ANALYSIS")
    print(json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2))

    print("\nTOP EVIDENCE")
    for i, e in enumerate(evidence, start=1):
        print("=" * 80)
        print(f"{i}. {e.get('chunk_id')}  score={e.get('_rerank_score')}")
        print(f"{e.get('title')} > {e.get('section')}")
        print(f"audience={e.get('audience')} cancer={e.get('cancer_type')}")
        print(e.get("text", "")[:1800])


if __name__ == "__main__":
    main()
