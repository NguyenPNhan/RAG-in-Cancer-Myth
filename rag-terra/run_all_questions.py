from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
for import_root in (PROJECT_ROOT, EXPERIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from rag.rag_model.config import PDQ_CANCERS_DIR, PDQ_GENERAL_TOPICS_DIR  # noqa: E402
from rag.rag_model.corpus import load_pdq_chunks  # noqa: E402
from rag.rag_model.prompts import PROMPT_OPTIONS  # noqa: E402
from rag.rag_model.retriever import BM25Retriever  # noqa: E402
from rag.rag_model.service import CancerMythRAG  # noqa: E402
from rag_terra import TERRA_MODEL, terra_settings  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "data" / "cancermyth_screening_dataset.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "results"
PROMPT_KEYS = tuple(option.key for option in PROMPT_OPTIONS)
CSV_COLUMNS = ["question_id", "answer"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-5.6 Terra over the cancer-myth RAG experiment.",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        help="Process only the first N questions (default: the complete dataset).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 4),
        help="Number of concurrent API requests.",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Retrieved passages per question.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per API request.")
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=PROMPT_KEYS,
        default=list(PROMPT_KEYS),
        help="Prompt variants to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prompt-specific answer CSVs.",
    )
    return parser.parse_args(argv)


def load_questions(limit: int | None) -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        questions = json.load(dataset_file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("The dataset must be a non-empty JSON array.")
    if any(
        not isinstance(row, dict)
        or "id" not in row
        or not str(row.get("question", "")).strip()
        for row in questions
    ):
        raise ValueError("Every dataset row must contain an ID and a non-empty question.")

    question_ids = [str(row["id"]) for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question IDs must be unique.")

    selected_count = len(questions) if limit is None else limit
    if isinstance(selected_count, bool) or not 1 <= selected_count <= len(questions):
        raise ValueError(f"--n-questions must be between 1 and {len(questions)}.")
    return questions[:selected_count]


def load_completed_ids(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists() or output_path.stat().st_size == 0:
        return completed

    with output_path.open(newline="", encoding="utf-8") as existing_file:
        reader = csv.DictReader(existing_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f"{output_path.name} must have columns {CSV_COLUMNS}.")
        for row in reader:
            question_id = row["question_id"].strip()
            if question_id in completed:
                raise ValueError(f"Duplicate question_id in {output_path.name}: {question_id}")
            if row["answer"] not in {"true", "false"}:
                raise ValueError(f"Invalid answer for question_id={question_id} in {output_path.name}.")
            completed.add(question_id)
    return completed


def classify_with_retry(
    rag: CancerMythRAG,
    question: str,
    prompt_key: str,
    settings,
    max_attempts: int,
    question_id: object,
) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            return rag.classify(question, prompt_key, settings).text
        except Exception:
            if attempt == max_attempts:
                raise
            delay_seconds = 2 ** (attempt - 1)
            print(
                f"[{prompt_key}] question_id={question_id} failed; "
                f"retrying in {delay_seconds}s."
            )
            time.sleep(delay_seconds)
    raise AssertionError("Unreachable")


def run_prompt(
    rag: CancerMythRAG,
    questions: list[dict],
    prompt_key: str,
    settings,
    output_dir: Path,
    workers: int,
    max_attempts: int,
) -> Path:
    output_path = output_dir / f"answers_{prompt_key}.csv"
    completed_ids = load_completed_ids(output_path)
    pending = [row for row in questions if str(row["id"]) not in completed_ids]
    worker_count = min(workers, max(1, len(pending)))
    print(
        f"\n[{prompt_key}] selected={len(questions)}, "
        f"completed={len(questions) - len(pending)}, pending={len(pending)}, "
        f"workers={worker_count}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            results_file.flush()
            os.fsync(results_file.fileno())

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_record = {
                executor.submit(
                    classify_with_retry,
                    rag,
                    str(record["question"]),
                    prompt_key,
                    settings,
                    max_attempts,
                    record["id"],
                ): record
                for record in pending
            }
            for finished, future in enumerate(as_completed(future_to_record), start=1):
                record = future_to_record[future]
                writer.writerow({"question_id": record["id"], "answer": future.result()})
                results_file.flush()
                os.fsync(results_file.fileno())
                if finished == 1 or finished % 10 == 0 or finished == len(pending):
                    print(f"[{prompt_key}] saved {finished}/{len(pending)} pending answers.")

    print(f"[{prompt_key}] complete: {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    questions = load_questions(args.n_questions)
    settings = terra_settings()
    settings.validate()
    if settings.model != TERRA_MODEL:
        raise AssertionError("The Terra experiment model was not pinned correctly.")
    if settings.base_url.rstrip("/") == "https://api.openai.com/v1" and not settings.api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the batch.")

    print(f"Model: {settings.model}")
    print(f"Selected questions: {len(questions)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    chunks = load_pdq_chunks((PDQ_CANCERS_DIR, PDQ_GENERAL_TOPICS_DIR))
    rag = CancerMythRAG(BM25Retriever(chunks), top_k=args.top_k)
    print(f"Indexed {len(chunks)} NCI PDQ chunks; retrieving top {args.top_k}.")

    for prompt_key in args.prompts:
        run_prompt(
            rag=rag,
            questions=questions,
            prompt_key=prompt_key,
            settings=settings,
            output_dir=args.output_dir,
            workers=args.workers,
            max_attempts=args.max_attempts,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
