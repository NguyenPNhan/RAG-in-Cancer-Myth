# Batch RAG classification and evaluation

This directory runs the Cancer Myth RAG classifier over the screening dataset
and evaluates its Boolean predictions. The classifier retrieves evidence from
the local NCI PDQ corpus and calls OpenAI's `gpt-5.6-luna` model with strict
JSON-schema output.

## Files

| Notebook | Prompt | Output |
| --- | --- | --- |
| `run_all_questions_basic.ipynb` | Basic classifier | `answers_basic.csv` |
| `run_all_questions_oncology_expert.ipynb` | Oncology expert | `answers_oncology_expert.csv` |
| `run_all_questions_patient_education.ipynb` | Patient-education specialist | `answers_patient_education.csv` |
| `evaluate.ipynb` | Evaluates one answer CSV | No output file |
| `wrong_detect.ipynb` | Lists misclassified question IDs from one answer CSV | No output file |

The input dataset is
`data/cancermyth_screening_dataset.json`. It contains 735 questions and their
reference Boolean answers.

## Setup

From the project root, provide an OpenAI API key:

```powershell
$env:OPENAI_API_KEY = "sk-your_key_here"
```

Open the notebooks with the project's Python environment. If JupyterLab is not
already installed, it can be launched temporarily through `uv`:

```powershell
uv run --with jupyterlab jupyter lab rag/rag_run_all
```

The default API configuration is:

```text
Base URL: https://api.openai.com/v1
Model:    gpt-5.6-luna
```

`LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` can override these defaults.
`LLM_API_KEY` takes precedence over `OPENAI_API_KEY`.

## Running classifications

Choose one of the three `run_all_questions_*.ipynb` notebooks and run its cells
in order. Each notebook loads the dataset, builds the NCI PDQ BM25 index once,
and then classifies questions concurrently.

The main controls are:

```python
N_QUESTIONS = len(questions)
N_WORKERS = max(1, os.cpu_count() or 4)
MAX_ATTEMPTS = 3
```

Set `N_QUESTIONS` to an integer to process only the first N records:

```python
N_QUESTIONS = 100
```

`N_WORKERS` controls the number of concurrent API requests. It defaults to the
machine's logical CPU count. Reduce it if the OpenAI API returns rate-limit
errors or the machine has limited memory.

### Checkpointing and resume behavior

Every completed result is appended to the prompt-specific CSV and flushed to
disk immediately. Before submitting requests, the notebook reads the existing
CSV and removes every completed `question_id` from the pending set. Rerunning a
notebook therefore skips questions that already have answers and does not make
duplicate API calls for them.

To intentionally start a prompt run from scratch, first move, rename, or delete
only that prompt's answer CSV.

## Prompts and output contract

The notebooks use one of these task prompts:

- `basic`: asks whether the patient question contains a false medical
  assumption.
- `oncology_expert`: performs the same classification from an oncology-expert
  perspective.
- `patient_education`: performs it from a cancer patient-education specialist
  perspective.

For every question, the selected prompt is combined with the top six retrieved
NCI PDQ passages. The API response is constrained by a strict JSON schema whose
only field is a Boolean `value`. No system message is used to enforce the output
format.

Each answer CSV contains exactly two columns:

```csv
question_id,answer
1,true
2,false
```

Because requests run concurrently, CSV rows can appear in completion order
rather than dataset order. Always join results using `question_id`.

## Evaluating answers

Open `evaluate.ipynb` and set `ANSWERS_PATH` to the answer file to evaluate. For
example:

```python
ANSWERS_PATH = (
    PROJECT_ROOT
    / "rag"
    / "rag_run_all"
    / "answers_oncology_expert.csv"
)
```

Run the remaining cells. The notebook validates IDs and Boolean values, joins
predictions to references by `question_id`, and reports:

- dataset coverage;
- accuracy, precision, recall, specificity, F1, and balanced accuracy;
- true-positive, true-negative, false-positive, and false-negative counts;
- metrics by dataset split and cancer type; and
- the first 20 misclassified questions.

Partial first-N result files are supported. Metrics that cannot be calculated
because only one reference class is present are displayed as `N/A`.

## List incorrect question IDs

Open `wrong_detect.ipynb` and set `ANSWERS_PATH` to the prompt-specific CSV you
want to inspect. The notebook validates and joins predictions by `question_id`,
then prints every ID whose `answer` differs from the dataset's
`correct_answer`. Partial answer files are supported.

## Notes

- Running the classifiers makes paid OpenAI API requests.
- An interrupted run is safe to restart because saved question IDs are skipped.
- Do not edit the CSV header; the runner and evaluator require exactly
  `question_id,answer`.
- The answer CSVs contain model predictions, while `correct_answer` remains only
  in the source JSON dataset and is not sent to the model.
