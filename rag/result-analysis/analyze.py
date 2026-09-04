#!/usr/bin/env python3
"""Compare the RAG and non-RAG cancer-myth classification results.

The script intentionally uses only Python's standard library. It validates all
inputs, joins by question ID, calculates classification and paired-comparison
statistics, and writes reproducible CSV, JSON, Markdown, and SVG artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROMPTS = ("basic", "oncology_expert", "patient_education")
CONDITIONS = ("rag", "non_rag")
DISPLAY_PROMPT = {
    "basic": "Basic",
    "oncology_expert": "Oncology expert",
    "patient_education": "Patient education",
}
COLORS = {"rag": "#2563eb", "non_rag": "#f97316"}
METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Analyze matched RAG and non-RAG cancer-myth predictions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "cancermyth_screening_dataset.json",
    )
    parser.add_argument(
        "--rag-dir", type=Path, default=root / "rag" / "rag_run_all"
    )
    parser.add_argument(
        "--non-rag-dir",
        type=Path,
        default=root / "non_rag" / "gpt-5.6-luna",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "rag" / "result-analysis" / "output"
    )
    return parser.parse_args(argv)


def id_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_dataset(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Dataset root must be a JSON array.")

    by_id: dict[str, dict] = {}
    required = {"id", "split", "cancer", "question", "correct_answer"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset row {index} must be an object.")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Dataset row {index} is missing: {sorted(missing)}")
        question_id = str(record["id"]).strip()
        if not question_id:
            raise ValueError(f"Dataset row {index} has an empty ID.")
        if question_id in by_id:
            raise ValueError(f"Duplicate dataset question ID: {question_id}")
        if not isinstance(record["correct_answer"], bool):
            raise ValueError(f"correct_answer for ID {question_id} must be Boolean.")
        by_id[question_id] = record
    if not by_id:
        raise ValueError("Dataset must not be empty.")
    return by_id


def parse_boolean(value: str, *, path: Path, question_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Invalid Boolean in {path.name} for question_id={question_id}: {value!r}"
    )


def load_predictions(path: Path, dataset: Mapping[str, dict]) -> dict[str, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    predictions: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["question_id", "answer"]:
            raise ValueError(f"{path} must contain exactly question_id,answer.")
        for line_number, row in enumerate(reader, start=2):
            question_id = (row.get("question_id") or "").strip()
            if question_id not in dataset:
                raise ValueError(
                    f"Unknown question_id {question_id!r} in {path} line {line_number}."
                )
            if question_id in predictions:
                raise ValueError(f"Duplicate question_id {question_id} in {path}.")
            predictions[question_id] = parse_boolean(
                row.get("answer") or "", path=path, question_id=question_id
            )
    if not predictions:
        raise ValueError(f"Prediction file is empty: {path}")
    return predictions


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def wilson_interval(successes: int, total: int, z: float = 1.95996398454) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def calculate_metrics(
    predictions: Mapping[str, bool],
    dataset: Mapping[str, dict],
    question_ids: Iterable[str],
) -> dict[str, int | float]:
    ids = list(question_ids)
    pairs = [(predictions[qid], dataset[qid]["correct_answer"]) for qid in ids]
    tp = sum(prediction is True and truth is True for prediction, truth in pairs)
    tn = sum(prediction is False and truth is False for prediction, truth in pairs)
    fp = sum(prediction is True and truth is False for prediction, truth in pairs)
    fn = sum(prediction is False and truth is True for prediction, truth in pairs)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, len(pairs))
    ci_low, ci_high = wilson_interval(tp + tn, len(pairs))
    return {
        "n": len(pairs),
        "accuracy": accuracy,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "precision": safe_divide(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
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


def exact_mcnemar_p(rag_only_correct: int, non_rag_only_correct: int) -> float:
    """Two-sided exact McNemar test using the discordant pairs."""
    discordant = rag_only_correct + non_rag_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(rag_only_correct, non_rag_only_correct)
    lower_tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * lower_tail)


def fmt_number(value: object, digits: int = 6) -> object:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.{digits}f}"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt_number(row.get(field, "")) for field in fields})


def svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 13,
    anchor: str = "middle",
    weight: str = "normal",
    fill: str = "#172033",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{html.escape(str(value))}</text>'
    )


def svg_document(width: int, height: int, body: Sequence[str], title: str) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{html.escape(title)}</title>",
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            *body,
            "</svg>",
        ]
    )


def write_overall_metrics_plot(path: Path, paired_metrics: Mapping[tuple[str, str], dict]) -> None:
    width, height = 1200, 770
    body = [svg_text(width / 2, 34, "RAG vs non-RAG metrics on matched questions", size=22, weight="bold")]
    body.extend(
        [
            f'<rect x="470" y="51" width="16" height="16" rx="2" fill="{COLORS["rag"]}"/>',
            svg_text(493, 64, "RAG", anchor="start"),
            f'<rect x="565" y="51" width="16" height="16" rx="2" fill="{COLORS["non_rag"]}"/>',
            svg_text(588, 64, "Non-RAG", anchor="start"),
        ]
    )
    panel_w, panel_h = 370, 300
    for index, metric in enumerate(METRIC_NAMES):
        col, row = index % 3, index // 3
        left, top = 55 + col * 390, 95 + row * 325
        chart_left, chart_top = left + 48, top + 35
        chart_w, chart_h = panel_w - 65, panel_h - 92
        body.append(svg_text(left + panel_w / 2, top + 18, metric.replace("_", " ").title(), size=16, weight="bold"))
        for tick in range(0, 6):
            value = tick / 5
            y = chart_top + chart_h * (1 - value)
            body.append(f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_left + chart_w}" y2="{y:.1f}" stroke="#dbe2ea"/>')
            body.append(svg_text(chart_left - 8, y + 4, f"{value:.1f}", anchor="end", size=10, fill="#526176"))
        group_w = chart_w / len(PROMPTS)
        bar_w = 31
        for prompt_index, prompt in enumerate(PROMPTS):
            center = chart_left + group_w * (prompt_index + 0.5)
            for condition_index, condition in enumerate(CONDITIONS):
                value = float(paired_metrics[(condition, prompt)][metric])
                value = 0 if math.isnan(value) else value
                x = center + (-bar_w - 2 if condition_index == 0 else 2)
                bar_h = value * chart_h
                y = chart_top + chart_h - bar_h
                body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="2" fill="{COLORS[condition]}"/>')
                body.append(svg_text(x + bar_w / 2, max(chart_top + 10, y - 4), f"{value:.3f}", size=9, fill="#334155"))
            label = {"basic": "Basic", "oncology_expert": "Oncology", "patient_education": "Patient ed."}[prompt]
            body.append(svg_text(center, chart_top + chart_h + 20, label, size=10))
        body.append(f'<line x1="{chart_left}" y1="{chart_top + chart_h}" x2="{chart_left + chart_w}" y2="{chart_top + chart_h}" stroke="#64748b"/>')
    path.write_text(svg_document(width, height, body, "Overall classification metrics"), encoding="utf-8")


def write_delta_plot(path: Path, comparisons: Sequence[Mapping[str, object]]) -> None:
    width, height = 900, 460
    left, right, top, bottom = 180, 60, 75, 75
    chart_w, chart_h = width - left - right, height - top - bottom
    max_abs = max(0.01, max(abs(float(row["accuracy_delta"])) for row in comparisons)) * 1.25
    body = [svg_text(width / 2, 33, "Accuracy change from retrieval", size=22, weight="bold")]
    body.append(svg_text(width / 2, 57, "Positive values favor RAG; matched question IDs", size=12, fill="#526176"))
    zero_x = left + chart_w / 2
    body.append(f'<line x1="{zero_x}" y1="{top - 10}" x2="{zero_x}" y2="{top + chart_h}" stroke="#172033" stroke-width="1.5"/>')
    for tick in range(-2, 3):
        value = max_abs * tick / 2
        x = left + chart_w * ((value + max_abs) / (2 * max_abs))
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="#e2e8f0"/>')
        body.append(svg_text(x, top + chart_h + 25, f"{value * 100:+.1f} pp", size=11, fill="#526176"))
    row_h = chart_h / len(comparisons)
    for index, row in enumerate(comparisons):
        y = top + row_h * (index + 0.5)
        delta = float(row["accuracy_delta"])
        end_x = left + chart_w * ((delta + max_abs) / (2 * max_abs))
        x = min(zero_x, end_x)
        bar_w = max(1, abs(end_x - zero_x))
        color = COLORS["rag"] if delta >= 0 else COLORS["non_rag"]
        body.append(svg_text(left - 18, y + 5, DISPLAY_PROMPT[str(row["prompt"])], anchor="end", size=13))
        body.append(f'<rect x="{x:.1f}" y="{y - 20:.1f}" width="{bar_w:.1f}" height="40" rx="3" fill="{color}"/>')
        anchor = "start" if delta >= 0 else "end"
        offset = 8 if delta >= 0 else -8
        body.append(svg_text(end_x + offset, y + 5, f"{delta * 100:+.2f} pp", anchor=anchor, weight="bold", fill=color))
        body.append(svg_text(width - 12, y + 5, f"p={float(row['mcnemar_exact_p']):.4f}", anchor="end", size=11, fill="#526176"))
    path.write_text(svg_document(width, height, body, "RAG accuracy delta"), encoding="utf-8")


def write_outcome_plot(path: Path, comparisons: Sequence[Mapping[str, object]]) -> None:
    width, height = 1000, 480
    left, right, top, bottom = 185, 55, 85, 60
    chart_w, chart_h = width - left - right, height - top - bottom
    categories = (
        ("both_correct", "Both correct", "#16a34a"),
        ("rag_only_correct", "RAG only correct", COLORS["rag"]),
        ("non_rag_only_correct", "Non-RAG only correct", COLORS["non_rag"]),
        ("both_wrong", "Both wrong", "#64748b"),
    )
    body = [svg_text(width / 2, 33, "Paired prediction outcomes", size=22, weight="bold")]
    legend_x = 205
    for key, label, color in categories:
        body.append(f'<rect x="{legend_x}" y="52" width="14" height="14" rx="2" fill="{color}"/>')
        body.append(svg_text(legend_x + 20, 64, label, anchor="start", size=11))
        legend_x += 185
    row_h = chart_h / len(comparisons)
    for index, row in enumerate(comparisons):
        y = top + row_h * (index + 0.5)
        n = int(row["n"])
        body.append(svg_text(left - 14, y + 5, DISPLAY_PROMPT[str(row["prompt"])], anchor="end", size=13))
        cursor = left
        for key, _, color in categories:
            count = int(row[key])
            segment_w = chart_w * count / n
            body.append(f'<rect x="{cursor:.1f}" y="{y - 23:.1f}" width="{segment_w:.1f}" height="46" fill="{color}"/>')
            if segment_w > 30:
                body.append(svg_text(cursor + segment_w / 2, y + 5, count, size=11, weight="bold", fill="#ffffff"))
            cursor += segment_w
    path.write_text(svg_document(width, height, body, "Paired outcome overlap"), encoding="utf-8")


def write_confusion_plot(path: Path, metrics: Mapping[tuple[str, str], dict]) -> None:
    width, height = 1000, 660
    body = [svg_text(width / 2, 34, "Confusion matrices on matched questions", size=22, weight="bold")]
    cell = 75
    for index, condition in enumerate(CONDITIONS):
        for prompt_index, prompt in enumerate(PROMPTS):
            col, row = prompt_index, index
            left, top = 95 + col * 305, 105 + row * 275
            values = metrics[(condition, prompt)]
            title = f"{'RAG' if condition == 'rag' else 'Non-RAG'} · {DISPLAY_PROMPT[prompt]}"
            body.append(svg_text(left + cell, top - 35, title, size=14, weight="bold"))
            body.append(svg_text(left + cell, top - 12, "Predicted", size=11, fill="#526176"))
            body.append(svg_text(left - 42, top + cell, "Actual", size=11, fill="#526176", rotate=-90))
            body.append(svg_text(left + cell / 2, top + 16, "False", size=10))
            body.append(svg_text(left + 1.5 * cell, top + 16, "True", size=10))
            matrix = (("tn", "True negative"), ("fp", "False positive"), ("fn", "False negative"), ("tp", "True positive"))
            max_value = max(int(values[key]) for key, _ in matrix) or 1
            for cell_index, (key, label) in enumerate(matrix):
                matrix_row, matrix_col = divmod(cell_index, 2)
                x, y = left + matrix_col * cell, top + 25 + matrix_row * cell
                value = int(values[key])
                opacity = 0.18 + 0.72 * value / max_value
                base = "37, 99, 235" if key in {"tn", "tp"} else "220, 38, 38"
                body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({base})" fill-opacity="{opacity:.3f}" stroke="#ffffff"/>')
                body.append(svg_text(x + cell / 2, y + 34, value, size=20, weight="bold", fill="#ffffff" if opacity > 0.55 else "#172033"))
                body.append(svg_text(x + cell / 2, y + 54, label.replace(" ", "\u00a0"), size=8, fill="#ffffff" if opacity > 0.55 else "#172033"))
            body.append(svg_text(left - 10, top + 25 + cell / 2 + 4, "False", anchor="end", size=10))
            body.append(svg_text(left - 10, top + 25 + 1.5 * cell + 4, "True", anchor="end", size=10))
    path.write_text(svg_document(width, height, body, "Confusion matrices"), encoding="utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer repository-relative paths while allowing explicit external inputs."""
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def pct(value: float) -> str:
    return "N/A" if math.isnan(value) else f"{value:.2%}"


def build_report(
    dataset: Mapping[str, dict],
    predictions: Mapping[tuple[str, str], Mapping[str, bool]],
    paired_metrics: Mapping[tuple[str, str], dict],
    comparisons: Sequence[Mapping[str, object]],
    all_common_ids: set[str],
) -> str:
    positive_count = sum(row["correct_answer"] is True for row in dataset.values())
    negative_count = len(dataset) - positive_count
    accuracy_deltas = [float(row["accuracy_delta"]) for row in comparisons]
    p_values = [float(row["mcnemar_exact_p"]) for row in comparisons]
    recall_deltas = [
        float(paired_metrics[("rag", prompt)]["recall"])
        - float(paired_metrics[("non_rag", prompt)]["recall"])
        for prompt in PROMPTS
    ]
    specificity_deltas = [
        float(paired_metrics[("rag", prompt)]["specificity"])
        - float(paired_metrics[("non_rag", prompt)]["specificity"])
        for prompt in PROMPTS
    ]
    balanced_deltas = [
        float(paired_metrics[("rag", prompt)]["balanced_accuracy"])
        - float(paired_metrics[("non_rag", prompt)]["balanced_accuracy"])
        for prompt in PROMPTS
    ]
    best_accuracy_key = max(
        paired_metrics, key=lambda key: float(paired_metrics[key]["accuracy"])
    )
    best_balanced_key = max(
        paired_metrics,
        key=lambda key: float(paired_metrics[key]["balanced_accuracy"]),
    )
    lines = [
        "# RAG vs non-RAG result analysis",
        "",
        "This report is generated by `rag/result-analysis/analyze.py`. Re-run the script after any prediction CSV changes.",
        "",
        "## Data completeness",
        "",
        "| Condition | Prompt | Predictions | Coverage |",
        "| --- | --- | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        for prompt in PROMPTS:
            count = len(predictions[(condition, prompt)])
            lines.append(f"| {condition.replace('_', '-').upper()} | {DISPLAY_PROMPT[prompt]} | {count:,} | {count / len(dataset):.2%} |")
    lines.extend(
        [
            "",
            f"The six-way intersection contains **{len(all_common_ids):,}** question IDs. Pairwise comparisons below use each prompt's RAG/non-RAG intersection.",
            f"The reference labels contain **{positive_count:,} positive** and **{negative_count:,} negative** cases ({positive_count / len(dataset):.2%} positive), so accuracy must be read alongside specificity and balanced accuracy.",
            "",
            "## Key findings",
            "",
            f"- RAG accuracy is higher for all three prompts by **{min(accuracy_deltas) * 100:.2f} to {max(accuracy_deltas) * 100:.2f} percentage points**. The paired exact McNemar p-values range from {min(p_values):.6f} to {max(p_values):.6f}.",
            f"- RAG recall is higher by **{min(recall_deltas) * 100:.2f} to {max(recall_deltas) * 100:.2f} points**, but specificity is lower by **{abs(max(specificity_deltas)) * 100:.2f} to {abs(min(specificity_deltas)) * 100:.2f} points**.",
            f"- Consequently, RAG balanced accuracy is lower by **{abs(max(balanced_deltas)) * 100:.2f} to {abs(min(balanced_deltas)) * 100:.2f} points**. Retrieval shifts the classifier toward the majority positive class rather than improving both classes uniformly.",
            f"- The highest raw accuracy is **{pct(float(paired_metrics[best_accuracy_key]['accuracy']))}** ({best_accuracy_key[0].replace('_', '-').upper()}, {DISPLAY_PROMPT[best_accuracy_key[1]]}); the highest balanced accuracy is **{pct(float(paired_metrics[best_balanced_key]['balanced_accuracy']))}** ({best_balanced_key[0].replace('_', '-').upper()}, {DISPLAY_PROMPT[best_balanced_key[1]]}).",
            "",
            "## Overall matched results",
            "",
            "| Prompt | Condition | n | Accuracy (95% CI) | Precision | Recall | Specificity | F1 | Balanced accuracy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for prompt in PROMPTS:
        for condition in CONDITIONS:
            row = paired_metrics[(condition, prompt)]
            ci = f"{pct(float(row['accuracy']))} ({pct(float(row['accuracy_ci_low']))} to {pct(float(row['accuracy_ci_high']))})"
            lines.append(
                f"| {DISPLAY_PROMPT[prompt]} | {condition.replace('_', '-').upper()} | {row['n']} | {ci} | "
                f"{pct(float(row['precision']))} | {pct(float(row['recall']))} | {pct(float(row['specificity']))} | "
                f"{pct(float(row['f1']))} | {pct(float(row['balanced_accuracy']))} |"
            )
    lines.extend(
        [
            "",
            "![Overall metrics](plots/overall_metrics.svg)",
            "",
            "## Retrieval effect",
            "",
            "| Prompt | Matched n | RAG accuracy | Non-RAG accuracy | Difference | RAG-only correct | Non-RAG-only correct | Exact McNemar p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {DISPLAY_PROMPT[str(row['prompt'])]} | {row['n']} | {pct(float(row['rag_accuracy']))} | "
            f"{pct(float(row['non_rag_accuracy']))} | {float(row['accuracy_delta']) * 100:+.2f} pp | "
            f"{row['rag_only_correct']} | {row['non_rag_only_correct']} | {float(row['mcnemar_exact_p']):.6f} |"
        )
    lines.extend(
        [
            "",
            "![Accuracy delta](plots/accuracy_delta.svg)",
            "",
            "![Paired outcomes](plots/paired_outcomes.svg)",
            "",
            "![Confusion matrices](plots/confusion_matrices.svg)",
            "",
            "## Interpretation guardrails",
            "",
            "- A positive difference means RAG was more accurate for that prompt on the same questions; it does not by itself establish that retrieval caused the change.",
            "- The exact McNemar test uses only discordant matched predictions. Its p-values are exploratory and are not adjusted for the three prompt comparisons.",
            "- Wilson intervals describe each accuracy estimate separately; they are not confidence intervals for the paired accuracy difference.",
            "- `true` is treated as the positive class: the question contains a false or materially misleading assumption.",
            "- Subgroup rows, especially individual cancer types, can be small. Use `subgroup_metrics.csv` for error exploration rather than definitive ranking.",
            "- The prediction files do not include latency, token use, retrieval relevance, or cost, so this report cannot compare those dimensions.",
            "",
            "## Machine-readable artifacts",
            "",
            "- `metrics.csv`: available-run and pairwise-matched classification metrics.",
            "- `paired_comparison.csv`: matched accuracy differences, error overlap, and exact McNemar tests.",
            "- `subgroup_metrics.csv`: six-way-matched metrics by dataset split and cancer type.",
            "- `predictions.csv`: reference labels and all six predictions, joined by question ID.",
            "- `disagreements.csv`: questions where corresponding RAG and non-RAG predictions differ.",
            "- `errors.csv`: every incorrect prediction with question text.",
            "- `manifest.json`: paths, hashes, counts, and analysis assumptions.",
        ]
    )
    if any(len(predictions[key]) != len(dataset) for key in predictions):
        lines.extend(
            [
                "",
                "> Warning: at least one run is incomplete. `available` metrics use all rows in each file, while comparative plots and tests use matched IDs only.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = args.dataset.resolve()
    source_paths = {
        ("rag", prompt): (args.rag_dir / f"answers_{prompt}.csv").resolve()
        for prompt in PROMPTS
    }
    source_paths.update(
        {
            ("non_rag", prompt): (args.non_rag_dir / f"answers_{prompt}.csv").resolve()
            for prompt in PROMPTS
        }
    )
    dataset = load_dataset(dataset_path)
    predictions = {
        key: load_predictions(path, dataset) for key, path in source_paths.items()
    }
    output = args.output.resolve()
    plots_dir = output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_common_ids = set.intersection(*(set(values) for values in predictions.values()))
    if not all_common_ids:
        raise ValueError("The six result files have no question IDs in common.")

    metric_fields = [
        "scope", "condition", "prompt", "n", "coverage", "accuracy",
        "accuracy_ci_low", "accuracy_ci_high", "precision", "recall",
        "specificity", "f1", "balanced_accuracy", "tp", "tn", "fp", "fn",
    ]
    metric_rows: list[dict[str, object]] = []
    available_metrics: dict[tuple[str, str], dict] = {}
    paired_metrics: dict[tuple[str, str], dict] = {}
    pair_ids_by_prompt: dict[str, set[str]] = {}
    for prompt in PROMPTS:
        pair_ids = set(predictions[("rag", prompt)]) & set(predictions[("non_rag", prompt)])
        if not pair_ids:
            raise ValueError(f"RAG and non-RAG have no shared IDs for prompt {prompt}.")
        pair_ids_by_prompt[prompt] = pair_ids
        for condition in CONDITIONS:
            key = (condition, prompt)
            available = calculate_metrics(predictions[key], dataset, predictions[key])
            paired = calculate_metrics(predictions[key], dataset, pair_ids)
            available_metrics[key] = available
            paired_metrics[key] = paired
            for scope, values in (("available", available), ("pairwise_matched", paired)):
                metric_rows.append(
                    {
                        "scope": scope,
                        "condition": condition,
                        "prompt": prompt,
                        "coverage": values["n"] / len(dataset),
                        **values,
                    }
                )
    write_csv(output / "metrics.csv", metric_rows, metric_fields)

    comparisons: list[dict[str, object]] = []
    for prompt in PROMPTS:
        ids = pair_ids_by_prompt[prompt]
        rag_prediction = predictions[("rag", prompt)]
        non_rag_prediction = predictions[("non_rag", prompt)]
        outcome_counts: Counter[str] = Counter()
        for qid in ids:
            truth = dataset[qid]["correct_answer"]
            rag_correct = rag_prediction[qid] is truth
            non_rag_correct = non_rag_prediction[qid] is truth
            if rag_correct and non_rag_correct:
                outcome_counts["both_correct"] += 1
            elif rag_correct:
                outcome_counts["rag_only_correct"] += 1
            elif non_rag_correct:
                outcome_counts["non_rag_only_correct"] += 1
            else:
                outcome_counts["both_wrong"] += 1
        rag_accuracy = float(paired_metrics[("rag", prompt)]["accuracy"])
        non_rag_accuracy = float(paired_metrics[("non_rag", prompt)]["accuracy"])
        comparisons.append(
            {
                "prompt": prompt,
                "n": len(ids),
                "rag_accuracy": rag_accuracy,
                "non_rag_accuracy": non_rag_accuracy,
                "accuracy_delta": rag_accuracy - non_rag_accuracy,
                **{key: outcome_counts[key] for key in ("both_correct", "rag_only_correct", "non_rag_only_correct", "both_wrong")},
                "discordant": outcome_counts["rag_only_correct"] + outcome_counts["non_rag_only_correct"],
                "mcnemar_exact_p": exact_mcnemar_p(outcome_counts["rag_only_correct"], outcome_counts["non_rag_only_correct"]),
            }
        )
    comparison_fields = [
        "prompt", "n", "rag_accuracy", "non_rag_accuracy", "accuracy_delta",
        "both_correct", "rag_only_correct", "non_rag_only_correct", "both_wrong",
        "discordant", "mcnemar_exact_p",
    ]
    write_csv(output / "paired_comparison.csv", comparisons, comparison_fields)

    subgroup_rows: list[dict[str, object]] = []
    for group_type in ("split", "cancer"):
        groups: dict[str, set[str]] = {}
        for qid in all_common_ids:
            group = str(dataset[qid][group_type]).strip() or "Unknown"
            groups.setdefault(group, set()).add(qid)
        for group, ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
            positives = sum(dataset[qid]["correct_answer"] is True for qid in ids)
            for condition in CONDITIONS:
                for prompt in PROMPTS:
                    values = calculate_metrics(predictions[(condition, prompt)], dataset, ids)
                    subgroup_rows.append(
                        {
                            "group_type": group_type,
                            "group": group,
                            "condition": condition,
                            "prompt": prompt,
                            "prevalence": positives / len(ids),
                            **values,
                        }
                    )
    subgroup_fields = [
        "group_type", "group", "condition", "prompt", "n", "prevalence",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high", "precision", "recall",
        "specificity", "f1", "balanced_accuracy", "tp", "tn", "fp", "fn",
    ]
    write_csv(output / "subgroup_metrics.csv", subgroup_rows, subgroup_fields)

    prediction_rows: list[dict[str, object]] = []
    for qid in sorted(dataset, key=id_sort_key):
        record = dataset[qid]
        row: dict[str, object] = {
            "question_id": qid,
            "split": record["split"],
            "cancer": record["cancer"],
            "correct_answer": str(record["correct_answer"]).lower(),
            "question": record["question"],
        }
        for condition in CONDITIONS:
            for prompt in PROMPTS:
                prediction = predictions[(condition, prompt)].get(qid)
                prefix = f"{condition}_{prompt}"
                row[prefix] = "" if prediction is None else str(prediction).lower()
                row[f"{prefix}_correct"] = "" if prediction is None else prediction is record["correct_answer"]
        prediction_rows.append(row)
    prediction_fields = ["question_id", "split", "cancer", "correct_answer"]
    for condition in CONDITIONS:
        for prompt in PROMPTS:
            prediction_fields.extend([f"{condition}_{prompt}", f"{condition}_{prompt}_correct"])
    prediction_fields.append("question")
    write_csv(output / "predictions.csv", prediction_rows, prediction_fields)

    disagreement_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for prompt in PROMPTS:
        for qid in sorted(pair_ids_by_prompt[prompt], key=id_sort_key):
            rag_value = predictions[("rag", prompt)][qid]
            non_rag_value = predictions[("non_rag", prompt)][qid]
            if rag_value is not non_rag_value:
                record = dataset[qid]
                disagreement_rows.append(
                    {
                        "prompt": prompt,
                        "question_id": qid,
                        "split": record["split"],
                        "cancer": record["cancer"],
                        "correct_answer": str(record["correct_answer"]).lower(),
                        "rag_answer": str(rag_value).lower(),
                        "non_rag_answer": str(non_rag_value).lower(),
                        "favored_condition": "rag" if rag_value is record["correct_answer"] else "non_rag",
                        "question": record["question"],
                    }
                )
    for condition in CONDITIONS:
        for prompt in PROMPTS:
            for qid, prediction in predictions[(condition, prompt)].items():
                record = dataset[qid]
                if prediction is not record["correct_answer"]:
                    error_rows.append(
                        {
                            "condition": condition,
                            "prompt": prompt,
                            "question_id": qid,
                            "split": record["split"],
                            "cancer": record["cancer"],
                            "correct_answer": str(record["correct_answer"]).lower(),
                            "predicted_answer": str(prediction).lower(),
                            "error_type": "false_positive" if prediction else "false_negative",
                            "question": record["question"],
                        }
                    )
    write_csv(
        output / "disagreements.csv",
        disagreement_rows,
        ["prompt", "question_id", "split", "cancer", "correct_answer", "rag_answer", "non_rag_answer", "favored_condition", "question"],
    )
    write_csv(
        output / "errors.csv",
        sorted(error_rows, key=lambda row: (str(row["condition"]), str(row["prompt"]), id_sort_key(str(row["question_id"])))),
        ["condition", "prompt", "question_id", "split", "cancer", "correct_answer", "predicted_answer", "error_type", "question"],
    )

    write_overall_metrics_plot(plots_dir / "overall_metrics.svg", paired_metrics)
    write_delta_plot(plots_dir / "accuracy_delta.svg", comparisons)
    write_outcome_plot(plots_dir / "paired_outcomes.svg", comparisons)
    write_confusion_plot(plots_dir / "confusion_matrices.svg", paired_metrics)
    (output / "report.md").write_text(
        build_report(dataset, predictions, paired_metrics, comparisons, all_common_ids),
        encoding="utf-8",
    )

    manifest = {
        "analysis": {
            "positive_class": "true = contains a false or materially misleading assumption",
            "prompts": list(PROMPTS),
            "conditions": list(CONDITIONS),
            "comparison_scope": "pairwise question-ID intersection for each prompt",
            "subgroup_scope": "intersection of all six result files",
            "accuracy_interval": "95% Wilson score interval",
            "paired_test": "two-sided exact McNemar test; unadjusted exploratory p-values",
        },
        "dataset": {
            "path": display_path(dataset_path),
            "sha256": hash_file(dataset_path),
            "records": len(dataset),
        },
        "prediction_files": [
            {
                "condition": condition,
                "prompt": prompt,
                "path": display_path(source_paths[(condition, prompt)]),
                "sha256": hash_file(source_paths[(condition, prompt)]),
                "records": len(predictions[(condition, prompt)]),
            }
            for condition in CONDITIONS
            for prompt in PROMPTS
        ],
        "six_way_common_records": len(all_common_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(dataset):,} reference questions.")
    print(f"Wrote report and artifacts to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
