from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# `streamlit run rag/rag_model/app.py` executes this file as a script. Ensure the
# project root is importable even if Streamlit only adds this file's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_model.config import (  # noqa: E402
    LLMSettings,
    PDQ_CANCERS_DIR,
    PDQ_GENERAL_TOPICS_DIR,
)
from rag.rag_model.corpus import load_pdq_chunks  # noqa: E402
from rag.rag_model.prompts import PROMPT_OPTIONS  # noqa: E402
from rag.rag_model.retriever import BM25Retriever  # noqa: E402
from rag.rag_model.service import CancerMythRAG  # noqa: E402


st.set_page_config(page_title="Cancer Myth RAG", page_icon="🔎")


@st.cache_resource(show_spinner="Loading and indexing NCI PDQ data...")
def build_rag() -> CancerMythRAG:
    chunks = load_pdq_chunks((PDQ_CANCERS_DIR, PDQ_GENERAL_TOPICS_DIR))
    return CancerMythRAG(BM25Retriever(chunks), top_k=6)


defaults = LLMSettings.from_environment()

st.title("Cancer Myth RAG — Qwen3-8B")
st.caption("Classify whether a patient question contains a false medical assumption.")

with st.sidebar:
    st.header("Model connection")
    base_url = st.text_input("Base URL", value=defaults.base_url)
    model = st.text_input("Qwen model", value=defaults.model)
    api_key = st.text_input("API key", value=defaults.api_key, type="password")
    st.caption("Uses an OpenAI-compatible chat-completions endpoint.")

question = st.text_area(
    "Patient question",
    height=160,
    placeholder="Enter a cancer-related patient question...",
)

labels_by_key = {option.key: option.label for option in PROMPT_OPTIONS}
prompt_key = st.radio(
    "Prompt",
    options=list(labels_by_key),
    format_func=labels_by_key.get,
)

if st.button("Classify", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a patient question.")
    elif not model.strip():
        st.warning("Enter the model name in the sidebar.")
    else:
        settings = LLMSettings(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=defaults.timeout_seconds,
        )
        try:
            with st.spinner("Retrieving NCI evidence and classifying..."):
                result = build_rag().classify(question, prompt_key, settings)
        except Exception as exc:
            st.error(f"Classification failed: {exc}")
        else:
            st.subheader("Answer")
            st.code(result.text, language=None)
