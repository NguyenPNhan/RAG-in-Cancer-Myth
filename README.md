# Cancer Myth RAG

This project explores whether retrieval-augmented generation (RAG) can identify
false or materially misleading medical assumptions in cancer-related patient
questions. It also compares how efficiently different prompt personas perform
when the model, retrieved evidence, dataset, and output format are held
constant.

The system retrieves relevant passages from the National Cancer Institute's
Physician Data Query (NCI PDQ) collection and gives those passages to
`gpt-5.6-luna`. For each patient question, the model returns one Boolean value:

- `true`: the question contains at least one false or materially misleading
  medical assumption;
- `false`: no such assumption was identified.

This is an experimental classification project, not a medical diagnosis or
patient-care tool.

## Research goal

The main goal is to compare three prompting strategies on the same 735-question
cancer-myth screening dataset:

| Prompt | Role given to the model |
| --- | --- |
| `basic` | Directly asks whether the question contains a false medical assumption |
| `oncology_expert` | Frames the model as an oncology expert |
| `patient_education` | Frames the model as a cancer patient-education specialist |

Keeping the remaining pipeline fixed makes the comparison more meaningful. All
three variants use the same GPT model, BM25 retrieval index, top-six NCI PDQ
passages, Boolean JSON schema, and reference labels.

In the current implementation, prompt efficiency primarily means
classification effectiveness: accuracy, precision, recall, specificity, F1,
balanced accuracy, and error patterns. The batch workflow also improves
operational efficiency through concurrent requests, per-question checkpoints,
and automatic resume behavior. Token usage, latency, and API cost are not yet
recorded as evaluation metrics.

## How it works

1. Load and chunk the local NCI PDQ cancer and general-topic documents.
2. Build an in-memory BM25 index over the passages and their metadata.
3. Retrieve the six most relevant passages for each patient question.
4. Combine the question, selected prompt, and retrieved evidence.
5. Ask `gpt-5.6-luna` for a strict JSON-schema response containing one Boolean
   `value`.
6. Save predictions by `question_id` and compare them with the reference labels.

The output format is enforced by the API's structured-output schema rather than
by relying on a system-message formatting instruction.

## Project structure

```text
data/
  cancermyth_screening_dataset.json   Labeled evaluation questions
  nci_pdq/                            Local NCI PDQ source documents
data retrieval/
  nci_pdq_crawler.py                  NCI PDQ collection script
rag/
  rag_model/                          Retrieval, LLM client, service, and web app
  rag_run_all/                        Batch notebooks, predictions, and evaluation
tests/
  test_rag_model.py                   Unit tests
```

See [`rag/rag_model/README.md`](rag/rag_model/README.md) for details about the
interactive application and [`rag/rag_run_all/README.md`](rag/rag_run_all/README.md)
for the complete batch and evaluation workflow.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key with access to `gpt-5.6-luna`

Install the locked project dependencies from the repository root:

```powershell
uv sync
```

Set the API key for the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "sk-your_key_here"
```

The default model configuration can be overridden when needed:

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_MODEL = "gpt-5.6-luna"
$env:LLM_API_KEY = $env:OPENAI_API_KEY
```

`LLM_API_KEY` takes precedence over `OPENAI_API_KEY`.

## Run the interactive website

Start the Streamlit application from the project root:

```powershell
uv run streamlit run rag/rag_model/app.py
```

The website accepts a patient question, lets the user select one of the three
prompts, retrieves NCI PDQ evidence, and displays the Boolean classification.

## Run the prompt comparison

Launch JupyterLab with the project environment:

```powershell
uv run --with jupyterlab jupyter lab rag/rag_run_all
```

Run one or more prompt-specific notebooks:

- `run_all_questions_basic.ipynb`
- `run_all_questions_oncology_expert.ipynb`
- `run_all_questions_patient_education.ipynb`

Each notebook exposes these controls:

```python
N_QUESTIONS = len(questions)
N_WORKERS = max(1, os.cpu_count() or 4)
MAX_ATTEMPTS = 3
```

Set `N_QUESTIONS` to process only the first N dataset records. The worker pool
runs requests concurrently. Each completed answer is immediately written to a
prompt-specific CSV with exactly these columns:

```csv
question_id,answer
1,true
2,false
```

Before submitting work, a notebook reads its existing CSV and skips every
`question_id` that already has an answer. Interrupted runs can therefore be
restarted without repeating completed API calls.

## Evaluate predictions

Open `rag/rag_run_all/evaluate.ipynb`, point `ANSWERS_PATH` at one of the
prompt-specific CSV files, and run all cells. The notebook joins predictions to
the dataset by `question_id` and reports:

- coverage of the dataset;
- accuracy, precision, recall, specificity, F1, and balanced accuracy;
- confusion-matrix counts;
- results by dataset split and cancer type; and
- examples of misclassified questions.

Evaluate all three CSV files separately to compare prompt performance under the
same RAG and model configuration.

## Run tests

```powershell
uv run python -m unittest discover -s tests -v
```

The tests cover prompt rendering, structured Boolean parsing, OpenAI request
construction, configuration, corpus loading, and BM25 retrieval.

## Reproducibility notes

- The batch notebooks may write CSV rows in completion order because API calls
  run concurrently. Use `question_id`, not row position, when joining results.
- Existing answers are treated as checkpoints and are not requested again.
- The reference `correct_answer` is used only during evaluation and is never
  included in the model prompt.
- Running the batch notebooks makes paid OpenAI API requests.
- Model behavior can change when an alias is updated. Pin a dated model snapshot
  when exact long-term reproducibility is required.
