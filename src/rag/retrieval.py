from typing import List, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(
        self,
        collection_name: str = "insurance_docs",
        persist_dir: str = "./data/chroma_db",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[dict], embeddings: List[List[float]]):
        self.collection.add(
            documents=[doc["content"] for doc in documents],
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in documents],
            ids=[f"doc_{i}" for i in range(len(documents))],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[RetrievalResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieval_results = []
        for i in range(len(results["documents"][0])):
            score = 1 - results["distances"][0][i]
            if score >= threshold:
                retrieval_results.append(
                    RetrievalResult(
                        content=results["documents"][0][i],
                        score=score,
                        metadata=results["metadatas"][0][i],
                    )
                )

        return retrieval_results

    def get_collection_stats(self) -> dict:
        return {"count": self.collection.count()}
