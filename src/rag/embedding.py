from typing import List
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass


class VertexAIEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-004", location: str = "us-central1"):
        self.model = model
        self.location = location
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from vertexai.language_models import TextEmbeddingModel
                self._client = TextEmbeddingModel.from_pretrained(self.model)
            except ImportError:
                raise ImportError(
                    "Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform"
                )
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        embeddings = client.get_embeddings(texts)
        return [emb.values for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        result = self.embed_texts([query])
        return result[0]


class LocalEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._client = SentenceTransformer(self.model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        embeddings = client.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


def get_embedding_provider(provider: str = "vertexai", **kwargs) -> EmbeddingProvider:
    if provider == "vertexai":
        return VertexAIEmbeddings(**kwargs)
    elif provider == "local":
        return LocalEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
