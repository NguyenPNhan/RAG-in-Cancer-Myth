# Cancer Myth RAG Streamlit app

This app retrieves relevant passages from both local NCI PDQ collections and
asks Qwen3-8B to return only `true` or `false`. The default configuration uses
the Ollama model name `qwen3:8b` through Ollama's OpenAI-compatible endpoint.

Qwen3 thinking is disabled with `/no_think` because this classifier needs a
short, strictly formatted answer.

## Configuration

Install Ollama separately, then download and start Qwen3-8B:

```powershell
ollama pull qwen3:8b
ollama serve
```

The Ollama model is about 5.2 GB. If Ollama is already running as a service,
only the `pull` command is needed.

The app defaults to the following settings, which can be overridden with
environment variables or in the sidebar:

```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL = "qwen3:8b"
$env:LLM_API_KEY = ""
```

`LLM_API_KEY` is optional for local endpoints that do not require authentication.
For an SGLang or vLLM server, set `LLM_BASE_URL` to that server and use its
served model identifier, commonly `Qwen/Qwen3-8B`.

## Run

From the project root:

```powershell
uv run streamlit run rag/rag_model/app.py
```

The first run parses and indexes the local JSON corpus in memory. Streamlit
caches the retriever for later questions in the same app process.
