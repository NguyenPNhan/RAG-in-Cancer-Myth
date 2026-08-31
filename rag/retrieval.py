from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
import json
import re

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .cancer_mapping import canonical_cancer_family


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-_/+.]*", text.lower())


class HybridRetriever:
    def __init__(
        self,
        chunks_file: str,
        qdrant_dir: str,
        collection_name: str,
        embedding_model: str,
        reranker_model: str | None = None,
    ):
        self.chunks = []
        with Path(chunks_file).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(json.loads(line))

        self.embedding_model = SentenceTransformer(embedding_model)
        self.client = QdrantClient(path=qdrant_dir)
        self.collection_name = collection_name

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        self.reranker = CrossEncoder(reranker_model) if reranker_model else None

    def dense_search(
        self,
        query: str,
        top_k: int = 30,
        cancer_family: str | None = None,
    ) -> list[dict]:
        vector = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        qfilter = None
        if cancer_family:
            qfilter = Filter(
                should=[
                    FieldCondition(
                        key="cancer_family",
                        match=MatchValue(value=cancer_family),
                    )
                ]
            )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for rank, h in enumerate(hits, start=1):
            payload = dict(h.payload or {})
            results.append({
                **payload,
                "_dense_score": float(h.score),
                "_dense_rank": rank,
            })
        return results

    def sparse_search(
        self,
        query: str,
        top_k: int = 30,
        cancer_family: str | None = None,
    ) -> list[dict]:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)))

        if cancer_family:
            mask = np.array([
                c.get("cancer_family") == cancer_family
                or not c.get("cancer_family")
                for c in self.chunks
            ])
            scores = np.where(mask, scores, -np.inf)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if not np.isfinite(scores[idx]):
                continue
            c = dict(self.chunks[int(idx)])
            c["_sparse_score"] = float(scores[idx])
            c["_sparse_rank"] = rank
            results.append(c)
        return results

    @staticmethod
    def rrf(
        result_lists: list[list[dict]],
        rrf_k: int = 60,
        limit: int = 30,
    ) -> list[dict]:
        scores = defaultdict(float)
        best_payload = {}

        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                cid = item["chunk_id"]
                scores[cid] += 1.0 / (rrf_k + rank)
                if cid not in best_payload:
                    best_payload[cid] = dict(item)
                else:
                    best_payload[cid].update({
                        k: v for k, v in item.items() if k.startswith("_")
                    })

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        out = []
        for rank, cid in enumerate(ranked_ids, start=1):
            item = best_payload[cid]
            item["_rrf_score"] = scores[cid]
            item["_rrf_rank"] = rank
            out.append(item)
        return out

    def hybrid_search(
        self,
        query: str,
        dense_k: int = 30,
        sparse_k: int = 30,
        fused_k: int = 30,
        rrf_k: int = 60,
        cancer_family: str | None = None,
    ) -> list[dict]:
        """
        Soft cancer-family preference.

        We retrieve both:
          1. cancer-family-scoped evidence, and
          2. global PDQ evidence.

        This prevents a metadata label mismatch from excluding useful general
        screening/supportive-care evidence while still rewarding same-family hits.
        """
        lists = []

        if cancer_family:
            lists.append(self.dense_search(query, dense_k, cancer_family))
            lists.append(self.sparse_search(query, sparse_k, cancer_family))

        lists.append(self.dense_search(query, dense_k, None))
        lists.append(self.sparse_search(query, sparse_k, None))

        return self.rrf(lists, rrf_k=rrf_k, limit=fused_k)

    @staticmethod
    def deduplicate(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for item in items:
            cid = item["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                out.append(item)
        return out

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        if not candidates:
            return []
        if self.reranker is None:
            return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["_rerank_score"] = float(score)

        return sorted(
            candidates,
            key=lambda x: x["_rerank_score"],
            reverse=True,
        )[:top_k]
