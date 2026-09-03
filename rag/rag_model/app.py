from __future__ import annotations

import json
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
from rag.rag_model.retriever import BM25Retriever, format_evidence  # noqa: E402
from rag.rag_model.service import CancerMythRAG  # noqa: E402


st.set_page_config(page_title="Cancer Myth RAG", page_icon="🔎")


@st.cache_resource(show_spinner="Loading and indexing NCI PDQ data...")
def build_rag() -> CancerMythRAG:
    chunks = load_pdq_chunks((PDQ_CANCERS_DIR, PDQ_GENERAL_TOPICS_DIR))
    return CancerMythRAG(BM25Retriever(chunks), top_k=6)


defaults = LLMSettings.from_environment()

st.title("Cancer Myth RAG — GPT-5.6 Luna")
st.caption("Classify whether a patient question contains a false medical assumption.")

with st.sidebar:
    st.header("Model connection")
    base_url = st.text_input("Base URL", value=defaults.base_url)
    model = st.text_input("OpenAI model", value=defaults.model)
    api_key = st.text_input("OpenAI API key", value=defaults.api_key, type="password")
    st.caption("OpenAI is the default; any compatible endpoint can be configured.")

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

            st.subheader("Retrieved NCI PDQ evidence")
            st.caption(
                "These are the passages retrieved for this question and supplied "
                "to the model. Passage text is limited to the same 1,800 characters "
                "used in the model request."
            )

            if not result.evidence:
                st.info("No matching NCI PDQ passage was retrieved.")
            else:
                evidence_records = []
                for item in result.evidence:
                    chunk = item.chunk
                    passage = chunk.text[:1_800]
                    evidence_records.append(
                        {
                            "rank": item.rank,
                            "score": item.score,
                            "chunk_id": chunk.chunk_id,
                            "title": chunk.title,
                            "cancer_type": chunk.cancer_type,
                            "topic": chunk.topic,
                            "audience": chunk.audience,
                            "section": chunk.section,
                            "section_path": list(chunk.section_path),
                            "url": chunk.url,
                            "source_file": chunk.source_file,
                            "passage": passage,
                        }
                    )

                    label = f"#{item.rank} · {chunk.title} — {chunk.section}"
                    with st.expander(label, expanded=item.rank == 1):
                        st.markdown(f"**BM25 score:** `{item.score:.4f}`")
                        st.markdown(
                            f"**Cancer/topic:** {chunk.cancer_type or 'General cancer topic'} "
                            f"/ {chunk.topic}"
                        )
                        st.markdown(f"**Audience:** {chunk.audience}")
                        st.markdown(f"**Section path:** {' > '.join(chunk.section_path)}")
                        if chunk.url:
                            st.markdown(f"**NCI source:** [{chunk.url}]({chunk.url})")
                        st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                        st.markdown("**Passage sent to the model:**")
                        st.write(passage)

                st.download_button(
                    "Download retrieved evidence (JSON)",
                    data=json.dumps(evidence_records, indent=2, ensure_ascii=False),
                    file_name="retrieved_evidence.json",
                    mime="application/json",
                    use_container_width=True,
                )

                with st.expander("View exact evidence block sent to the model"):
                    st.code(format_evidence(result.evidence), language=None)
