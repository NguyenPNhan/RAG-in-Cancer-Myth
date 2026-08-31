from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def build_dense_index(
    chunks_file: str | Path,
    qdrant_dir: str | Path,
    collection_name: str,
    embedding_model: str,
    batch_size: int = 32,
    recreate: bool = True,
):
    chunks = read_jsonl(chunks_file)
    if not chunks:
        raise RuntimeError("No chunks found. Run scripts/prepare_corpus.py first.")

    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError("Could not determine embedding dimension.")

    client = QdrantClient(path=str(qdrant_dir))

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for start in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        points = []
        for idx, (chunk, vector) in enumerate(zip(batch, vectors), start=start):
            # Qdrant local accepts integer IDs reliably; keep chunk_id in payload.
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload=chunk,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    print(f"Indexed {len(chunks)} chunks into collection '{collection_name}'.")
