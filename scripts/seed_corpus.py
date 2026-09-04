"""Rebuild the ChromaDB corpus from ./data reproducibly.

Resets the target collection, then chunk + embed + index every .txt/.pdf file
in the data directory. Run from the project root:

    python scripts/seed_corpus.py

Use --collection and --persist-dir to target a different store.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("seed_corpus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.config import load_config
from src.rag.embedding import get_embedding_provider
from src.rag.ingestion import DocumentLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ChromaDB corpus from data/")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--persist-dir", default=None)
    args = parser.parse_args()

    config = load_config()
    collection = args.collection or config.vectorstore.collection_name
    persist_dir = args.persist_dir or config.vectorstore.persist_dir

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=persist_dir)
    for name in client.list_collections():
        if name.name == collection:
            client.delete_collection(collection)
            log.info("reset existing collection '%s'", collection)

    store = __import__("src.rag.retrieval", fromlist=["VectorStore"]).VectorStore(
        collection_name=collection,
        persist_dir=persist_dir,
    )

    loader = DocumentLoader(
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
    )
    documents = loader.load_directory(args.data_dir)
    if not documents:
        raise SystemExit(f"no documents found under {args.data_dir}")

    emb = get_embedding_provider(provider=config.embedding.provider, model=config.embedding.model)
    contents = [doc.content for doc in documents]
    log.info("embedding %d chunks...", len(contents))
    embeddings = emb.embed_texts(contents)

    doc_dicts = [{"content": doc.content, "metadata": doc.metadata} for doc in documents]
    store.add_documents(doc_dicts, embeddings)

    sources = sorted({m["source"] for m in store.collection.get(include=["metadatas"])["metadatas"]})
    log.info("done. chunks=%d sources=%d", store.get_collection_stats()["count"], len(sources))
    for s in sources:
        log.info("  %s", s)


if __name__ == "__main__":
    main()