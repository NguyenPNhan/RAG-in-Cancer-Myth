# Result analysis

This folder contains the reproducible comparison of the three RAG runs and
their matching non-RAG baselines. The analyzer validates every input, joins
predictions by `question_id`, calculates both ordinary classification metrics
and paired RAG/non-RAG comparisons, and creates plots without third-party
Python packages.

## Run in Jupyter

From the repository root:

```powershell
uv run --with jupyterlab jupyter lab result-analysis
```

Open `result_analysis.ipynb` and select **Run All Cells**. The notebook validates
the inputs, regenerates all outputs, displays the matched metrics and plots, and
provides interactive cells for subgroup and error inspection.

For non-interactive automation, the same tested analysis is available as:

```powershell
uv run python result-analysis/analyze.py
```

The default inputs are:

```text
data/cancermyth_screening_dataset.json
rag/rag_run_all/answers_{prompt}.csv
non_rag/gpt-5.6-luna/answers_{prompt}.csv
```

The command-line version accepts `--dataset`, `--rag-dir`, `--non-rag-dir`, or
`--output` overrides. Run `uv run python result-analysis/analyze.py --help` for
details.

## Outputs

Generated artifacts are written to `result-analysis/output/`:

- `report.md` is the human-readable result summary.
- `metrics.csv` contains available-run and pairwise-matched metrics.
- `paired_comparison.csv` measures the retrieval effect on matched questions
  and includes two-sided exact McNemar tests.
- `subgroup_metrics.csv` contains results by split and cancer type.
- `predictions.csv` joins the reference data to all six prediction columns.
- `disagreements.csv` isolates RAG/non-RAG prediction changes.
- `errors.csv` contains all false positives and false negatives.
- `manifest.json` records source paths, SHA-256 hashes, counts, and statistical
  assumptions.
- `plots/` contains SVG charts for overall metrics, accuracy differences,
  paired outcomes, and confusion matrices.

SVG was chosen because it is portable, version-control friendly, and viewable
in browsers and Markdown without adding plotting dependencies.

## Statistical scope

Each prompt's direct RAG/non-RAG comparison uses the intersection of question
IDs present in that prompt's two files. The general subgroup table uses the
intersection across all six files. This ensures compared values refer to the
same questions even when a future run is incomplete.

The positive class is `true`: a question contains a false or materially
misleading assumption. Accuracy receives a 95% Wilson score interval. The exact
McNemar test evaluates whether the paired error rates differ, using only cases
where one condition is correct and the other is wrong. Its three p-values are
exploratory and unadjusted.

For the full design, validation rules, artifact contract, and limitations, see
[`../plan/result_analysis_plan.md`](../plan/result_analysis_plan.md).
