# GPT-5.6 Luna non-RAG baseline

This directory provides the non-RAG baseline for the cancer-myth prompt
comparison. It classifies the same questions with the same `gpt-5.6-luna`
model, prompt templates, concurrency settings, and strict Boolean JSON schema as
the RAG experiment, but it does **not** retrieve or send NCI PDQ passages.

Holding everything except retrieval constant makes it possible to compare:

- RAG versus non-RAG classification quality; and
- the `basic`, `oncology_expert`, and `patient_education` prompts within the
  non-RAG condition.

## Files

- `run_all_questions.ipynb` runs all three prompt variants and creates one CSV
  per prompt.
- `evaluate.ipynb` evaluates and compares the three non-RAG CSVs.
- `wrong_detect.ipynb` lists incorrectly classified question IDs for a selected
  answer CSV.
- `answers_basic.csv`, `answers_oncology_expert.csv`, and
  `answers_patient_education.csv` are created when the runner is executed.

## Setup

From the project root:

```powershell
uv sync
$env:OPENAI_API_KEY = "sk-your_key_here"
uv run --with jupyterlab jupyter lab non_rag/gpt-5.6-luna
```

## Run the experiment

Open `run_all_questions.ipynb`. The main controls are:

```python
PROMPT_KEYS = ("basic", "oncology_expert", "patient_education")
N_QUESTIONS = len(questions)
N_WORKERS = max(1, os.cpu_count() or 4)
MAX_ATTEMPTS = 3
```

Set `N_QUESTIONS` to an integer to run only the first N dataset records. Reduce
`N_WORKERS` if API rate limits are reached.

For each prompt, the notebook writes exactly two columns:

```csv
question_id,answer
1,true
2,false
```

Results are checkpointed after every completed request. On rerun, existing
valid `question_id` values are skipped, so already answered questions do not
generate another API request.

The reference `correct_answer` field is never included in the model input.

## Evaluate

Run `evaluate.ipynb` after producing the answer files. It reports accuracy,
precision, recall, specificity, F1, balanced accuracy, confusion-matrix counts,
and coverage for each prompt. It also performs a fair comparison over the
intersection of question IDs answered by all three prompts and lists prompt
disagreements.

To compare RAG and non-RAG results, use the corresponding CSVs in
`rag/rag_run_all` and this directory. Always join by `question_id`, because
concurrent requests may write rows in completion order.

## Cost note

Running the batch notebook makes paid OpenAI API requests. The notebook defaults
to the complete dataset for all three prompts, so set a smaller `N_QUESTIONS`
for a trial run if desired.
