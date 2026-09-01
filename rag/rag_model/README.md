# Cancer Myth RAG Streamlit app

This app retrieves relevant passages from both local NCI PDQ collections and
asks OpenAI's `gpt-5.6-luna` model for a strict JSON-schema response containing
a boolean `value` through the Chat Completions API.

## Configuration

Create an OpenAI API key at <https://platform.openai.com/api-keys>, then expose
it to the app before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "sk-your_key_here"
```

The app defaults to the following settings, which can be overridden with
environment variables or in the sidebar:

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_MODEL = "gpt-5.6-luna"
$env:LLM_API_KEY = $env:OPENAI_API_KEY
```

`LLM_API_KEY` takes precedence over `OPENAI_API_KEY` when both are set. This keeps
the connection generic: to use Ollama, SGLang, vLLM, or another compatible
provider instead, override the base URL, model, and API key as needed.

## Run

From the project root:

```powershell
uv run streamlit run rag/rag_model/app.py
```

The first run parses and indexes the local JSON corpus in memory. Streamlit
caches the retriever for later questions in the same app process.
