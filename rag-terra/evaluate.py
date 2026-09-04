from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DATASET_PATH = PROJECT_ROOT / "data" / "cancermyth_screening_dataset.json"
DEFAULT_RESULTS_DIR = EXPERIMENT_ROOT / "results"
PROMPT_KEYS = ("basic", "oncology_expert", "patient_education")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.6 Terra RAG predictions.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def parse_boolean(value: str, question_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid answer for question_id={question_id}: {value!r}")


def load_dataset() -> tuple[list[dict], dict[str, dict]]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    dataset_by_id = {str(row["id"]): row for row in dataset}
    if len(dataset_by_id) != len(dataset):
        raise ValueError("Dataset question IDs must be unique.")
    if any(not isinstance(row.get("correct_answer"), bool) for row in dataset):
        raise ValueError("Every correct_answer must be Boolean.")
    return dataset, dataset_by_id


def load_predictions(path: Path, dataset_by_id: dict[str, dict]) -> dict[str, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing answer file: {path}")
    predictions: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as answer_file:
        reader = csv.DictReader(answer_file)
        if reader.fieldnames != ["question_id", "answer"]:
            raise ValueError(f"{path.name} must contain exactly question_id,answer.")
        for row in reader:
            question_id = row["question_id"].strip()
            if question_id not in dataset_by_id:
                raise ValueError(f"Unknown question_id in {path.name}: {question_id}")
            if question_id in predictions:
                raise ValueError(f"Duplicate question_id in {path.name}: {question_id}")
            predictions[question_id] = parse_boolean(row["answer"], question_id)
    return predictions


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def calculate_metrics(
    predictions: dict[str, bool],
    question_ids: set[str],
    dataset_by_id: dict[str, dict],
) -> dict[str, float | int]:
    pairs = [(predictions[qid], dataset_by_id[qid]["correct_answer"]) for qid in question_ids]
    tp = sum(prediction is True and expected is True for prediction, expected in pairs)
    tn = sum(prediction is False and expected is False for prediction, expected in pairs)
    fp = sum(prediction is True and expected is False for prediction, expected in pairs)
    fn = sum(prediction is False and expected is True for prediction, expected in pairs)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "n": len(pairs),
        "coverage": divide(len(pairs), len(dataset_by_id)),
        "accuracy": divide(tp + tn, len(pairs)),
        "precision": divide(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if not math.isnan(recall) and not math.isnan(specificity)
            else math.nan
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def display_metric(value: float | int) -> str:
    if isinstance(value, float):
        return "N/A" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset, dataset_by_id = load_dataset()
    predictions_by_prompt = {
        key: load_predictions(args.results_dir / f"answers_{key}.csv", dataset_by_id)
        for key in PROMPT_KEYS
    }

    metric_names = (
        "n",
        "coverage",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
    )
    print(f"GPT-5.6 Terra RAG evaluation ({len(dataset)} reference questions)\n")
    print("prompt," + ",".join(metric_names))
    for prompt_key, predictions in predictions_by_prompt.items():
        metrics = calculate_metrics(predictions, set(predictions), dataset_by_id)
        values = ",".join(display_metric(metrics[name]) for name in metric_names)
        print(f"{prompt_key},{values}")

    common_ids = set.intersection(*(set(values) for values in predictions_by_prompt.values()))
    print(f"\nCommon question IDs across all prompts: {len(common_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
