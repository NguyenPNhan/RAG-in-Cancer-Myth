# GPT-5.6 Terra Cancer Myth RAG

This folder runs the existing cancer-myth retrieval-augmented generation (RAG)
pipeline with `gpt-5.6-terra` pinned as the classifier. It reuses the project's
NCI PDQ corpus loader, BM25 retriever, three prompt templates, and strict Boolean
JSON-schema output contract.

The model cannot be changed through `LLM_MODEL` in this experiment. The base URL,
API key, and timeout remain configurable through `LLM_BASE_URL`, `LLM_API_KEY`
(or `OPENAI_API_KEY`), and `LLM_TIMEOUT_SECONDS`.

## Setup

From the project root:

```powershell
uv sync
$env:OPENAI_API_KEY = "sk-your_key_here"
```

## Interactive app

```powershell
uv run streamlit run rag-terra/app.py
```

The app retrieves the top six matching NCI PDQ passages, sends them to
`gpt-5.6-terra` with the selected prompt, and displays the Boolean result and
retrieved evidence.

## Batch experiment

Run all 735 questions with all three prompt variants:

```powershell
uv run python rag-terra/run_all_questions.py
```

For a small trial:

```powershell
uv run python rag-terra/run_all_questions.py --n-questions 10 --workers 2
```

Useful options include `--top-k`, `--max-attempts`, `--workers`, `--prompts`,
and `--output-dir`. Run with `--help` for the complete interface.

Each prompt writes its own checkpoint file under `rag-terra/results/`:

- `answers_basic.csv`
- `answers_oncology_expert.csv`
- `answers_patient_education.csv`

Every completed answer is flushed immediately. Rerunning the command skips valid
question IDs already in the corresponding CSV. Rows may be in completion order,
so comparisons must join on `question_id`.

## Evaluate

After generating all three answer files:

```powershell
uv run python rag-terra/evaluate.py
```

The evaluator reports coverage, accuracy, precision, recall, specificity, F1,
balanced accuracy, and confusion-matrix counts for each prompt. It also reports
the number of question IDs common to all three result files.

## Notebook batch workflow

The notebook workflow matching `rag/rag_run_all` is available in
`rag-terra/rag_run_all`:

```powershell
uv run --with jupyterlab jupyter lab rag-terra/rag_run_all
```

It contains one runner per prompt, an evaluation notebook, a wrong-answer
detector, and separate CSV checkpoints. See
[`rag_run_all/README.md`](rag_run_all/README.md) for details.

## RAG versus non-RAG result analysis

After the Terra RAG and non-RAG CSVs have been generated, launch the dedicated
comparison notebook:

```powershell
uv run --with jupyterlab jupyter lab rag-terra/result-analysis
```

See [`result-analysis/README.md`](result-analysis/README.md) for the input paths,
metrics, paired tests, and generated artifacts.

Running the batch experiment makes paid API requests. The reference
`correct_answer` field is used only by the evaluator and is never sent to the
model.
