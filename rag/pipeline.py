from __future__ import annotations

from .config import load_config
from .llm import OpenAICompatibleChatClient
from .query_analyzer import analyze_question
from .generation import generate_answer
from .retrieval import HybridRetriever


class CancerMythRAG:
    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        self.llm = OpenAICompatibleChatClient.from_config(self.cfg)

        r_cfg = self.cfg["retrieval"]
        reranker_model = (
            r_cfg["reranker_model"]
            if r_cfg.get("enable_reranker", True)
            else None
        )

        self.retriever = HybridRetriever(
            chunks_file=self.cfg["paths"]["chunks_file"],
            qdrant_dir=self.cfg["paths"]["qdrant_dir"],
            collection_name=self.cfg["index"]["collection_name"],
            embedding_model=self.cfg["index"]["embedding_model"],
            reranker_model=reranker_model,
        )

    def retrieve(self, question: str, cancer_hint: str | None = None):
        analysis = analyze_question(self.llm, question, cancer_hint=cancer_hint)
        r_cfg = self.cfg["retrieval"]

        candidates = []
        for q in analysis.search_queries:
            candidates.extend(
                self.retriever.hybrid_search(
                    query=q,
                    dense_k=r_cfg["dense_k"],
                    sparse_k=r_cfg["sparse_k"],
                    fused_k=r_cfg["fused_k"],
                    rrf_k=r_cfg["rrf_k"],
                    cancer_family=analysis.cancer_family,
                )
            )

        candidates = self.retriever.deduplicate(candidates)

        # First rank by max available RRF score before cross-encoder.
        candidates.sort(
            key=lambda x: x.get("_rrf_score", 0.0),
            reverse=True,
        )
        candidates = candidates[: max(r_cfg["fused_k"], 50)]

        evidence = self.retriever.rerank(
            question,
            candidates,
            top_k=r_cfg["rerank_k"],
        )

        return analysis, evidence

    def answer(self, question: str, cancer_hint: str | None = None) -> dict:
        analysis, evidence = self.retrieve(question, cancer_hint)
        result = generate_answer(
            self.llm,
            question=question,
            analysis=analysis,
            evidence=evidence,
        )

        result["_analysis"] = analysis.model_dump()
        result["_retrieved_evidence"] = [
            {
                "chunk_id": e.get("chunk_id"),
                "title": e.get("title"),
                "section": e.get("section"),
                "audience": e.get("audience"),
                "cancer_type": e.get("cancer_type"),
                "url": e.get("url"),
                "rerank_score": e.get("_rerank_score"),
                "text": e.get("text"),
            }
            for e in evidence
        ]
        return result
