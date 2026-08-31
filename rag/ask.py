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
    result = rag.answer(args.question, cancer_hint=args.cancer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
