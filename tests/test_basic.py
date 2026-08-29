import pytest
from src.config import load_config, Config


def test_config_loads():
    config = load_config()
    assert config.llm.model == "gemini-2.5-flash"
    assert config.rag.chunk_size == 512
    assert config.vectorstore.provider == "chromadb"
    assert config.embedding.provider == "local"


def test_document_loader():
    from src.rag.ingestion import DocumentLoader

    loader = DocumentLoader(chunk_size=100, chunk_overlap=10)
    assert loader.chunk_size == 100
    assert loader.chunk_overlap == 10


def test_vector_store_init():
    from src.rag.retrieval import VectorStore

    store = VectorStore(
        collection_name="test_collection",
        persist_dir="./tmp/test_chroma",
    )
    stats = store.get_collection_stats()
    assert stats["count"] == 0


def test_embedding_factory():
    from src.rag.embedding import get_embedding_provider, VertexAIEmbeddings, LocalEmbeddings

    vertex_provider = get_embedding_provider("vertexai", model="text-embedding-004")
    assert isinstance(vertex_provider, VertexAIEmbeddings)
    assert vertex_provider.model == "text-embedding-004"

    local_provider = get_embedding_provider("local", model="all-MiniLM-L6-v2")
    assert isinstance(local_provider, LocalEmbeddings)
    assert local_provider.model == "all-MiniLM-L6-v2"
