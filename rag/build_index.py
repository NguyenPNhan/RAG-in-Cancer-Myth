from cancer_rag.config import load_config
from cancer_rag.indexing import build_dense_index


def main():
    cfg = load_config("config.yaml")
    build_dense_index(
        chunks_file=cfg["paths"]["chunks_file"],
        qdrant_dir=cfg["paths"]["qdrant_dir"],
        collection_name=cfg["index"]["collection_name"],
        embedding_model=cfg["index"]["embedding_model"],
        batch_size=cfg["index"]["batch_size"],
        recreate=True,
    )


if __name__ == "__main__":
    main()
