# RAG and non-RAG result-analysis plan

## 1. Objective

Create one reproducible analysis that answers whether adding NCI PDQ retrieval
changes cancer-myth classification quality when the prompt variant is held
constant. Compare `basic`, `oncology_expert`, and `patient_education` under both
RAG and non-RAG conditions, expose the error-level changes, and retain enough
metadata to reproduce every reported number.

## 2. Inputs

The reference dataset is `data/cancermyth_screening_dataset.json`. Its `id` is
the join key and `correct_answer` is the Boolean target. `split`, `cancer`, and
`question` are used only for stratification and error inspection.

The six prediction inputs are:

| Condition | Prompt | File |
| --- | --- | --- |
| RAG | Basic | `rag/rag_run_all/answers_basic.csv` |
| RAG | Oncology expert | `rag/rag_run_all/answers_oncology_expert.csv` |
| RAG | Patient education | `rag/rag_run_all/answers_patient_education.csv` |
| Non-RAG | Basic | `non_rag/gpt-5.6-luna/answers_basic.csv` |
| Non-RAG | Oncology expert | `non_rag/gpt-5.6-luna/answers_oncology_expert.csv` |
| Non-RAG | Patient education | `non_rag/gpt-5.6-luna/answers_patient_education.csv` |

Prediction CSVs must contain exactly `question_id,answer`. Row order is not
meaningful because batch requests finish concurrently.

## 3. Validation and joining

Before calculating metrics, `rag/result-analysis/analyze.py` checks that:

1. the dataset is a non-empty JSON array with unique IDs and Boolean labels;
2. every required dataset field is present;
3. every prediction file exists and has the exact two-column schema;
4. answers normalize to `true` or `false`;
5. question IDs exist in the reference dataset; and
6. no prediction file contains duplicate IDs.

Metrics for a single run may use all available predictions. A direct RAG versus
non-RAG comparison uses the question-ID intersection for the same prompt.
Subgroup metrics use the six-way intersection, making every condition/prompt
cell directly comparable within a group. Coverage is always reported so a
partial run cannot silently look like a full-dataset result.

## 4. Metrics

`true` is the positive class and means that the question contains at least one
false or materially misleading medical assumption.

For each condition and prompt, calculate:

- accuracy with a 95% Wilson score interval;
- precision, recall, specificity, F1, and balanced accuracy; and
- true-positive, true-negative, false-positive, and false-negative counts.

Undefined ratios are left blank in CSV output and shown as `N/A` in the report.
Balanced accuracy is undefined when either recall or specificity is undefined.

## 5. Paired retrieval comparison

For every prompt, classify matched questions into four outcomes:

- both RAG and non-RAG correct;
- RAG only correct;
- non-RAG only correct; and
- both wrong.

The primary descriptive effect is the matched accuracy difference:

```text
RAG accuracy - non-RAG accuracy
```

A two-sided exact McNemar test uses the two discordant counts to test whether
paired error rates differ. The resulting p-values are exploratory and are not
corrected for the three prompt comparisons. Statistical significance should
not replace inspection of effect size and disagreement examples.

## 6. Subgroup and error analysis

Metrics are stratified by dataset `split` and `cancer`. Cancer groups may be
small, so these rows are intended for hypothesis generation and error tracing,
not stable league tables. `disagreements.csv` shows exactly where retrieval
changed a prediction and which condition was correct. `errors.csv` provides all
false positives and false negatives with their original question text.

No sensitive patient records are introduced: the analysis uses only the
project's existing benchmark questions and predictions.

## 7. Artifact contract

The notebook calls the tested analysis module, which regenerates
`rag/result-analysis/output/` deterministically from the current inputs:

| Artifact | Purpose |
| --- | --- |
| `report.md` | Human-readable summary, tables, plots, and caveats |
| `metrics.csv` | Run-level available and matched classification metrics |
| `paired_comparison.csv` | Accuracy deltas, overlap counts, and McNemar tests |
| `subgroup_metrics.csv` | Six-way-matched split and cancer-type metrics |
| `predictions.csv` | One joined audit row per reference question |
| `disagreements.csv` | Matched RAG/non-RAG prediction changes |
| `errors.csv` | Every incorrect run-level prediction |
| `manifest.json` | Input paths, SHA-256 hashes, row counts, and assumptions |
| `plots/*.svg` | Overall metrics, deltas, overlap, and confusion matrices |

The implementation uses only the Python standard library. SVG provides
dependency-free, text-based plots that render in browsers and Markdown.

## 8. Reproduction procedure

From the project root:

```powershell
uv run --with jupyterlab jupyter lab rag/result-analysis
```

Open `result_analysis.ipynb` and run all cells. For automated or headless use,
run the equivalent command-line entry point:

```powershell
uv run python rag/result-analysis/analyze.py
uv run python -m unittest discover -s tests -v
```

Review `rag/result-analysis/output/manifest.json` alongside reported results. If a
prediction file changes, rerun the analyzer and commit the refreshed artifacts
together so the source hashes and report stay synchronized.

## 9. Interpretation limits

- The comparison is observational across saved model outputs. Paired matching
  controls question composition, but model nondeterminism or run timing may
  still contribute to differences.
- Accuracy does not measure evidence faithfulness, retrieval relevance,
  calibration, explanation quality, clinical safety, latency, token use, or
  API cost.
- The current CSV contract stores only Boolean predictions, so those missing
  dimensions cannot be reconstructed after a run.
- Per-cancer estimates often have small denominators and wide uncertainty.
- Model aliases can change over time; a dated model snapshot is preferable for
  a future confirmatory experiment.

## 10. Acceptance criteria

The work is complete when the Jupyter notebook runs the analyzer, validates the
six inputs, regenerates all documented artifacts, compares only matched IDs,
passes its unit tests, and the report states both numerical results and their
limitations.
