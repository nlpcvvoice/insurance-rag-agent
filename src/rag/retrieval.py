import re
from typing import List, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: dict


def _tokenize(text: str) -> List[str]:
    """Simple English tokenizer: lowercase + split on non-alphanumerics."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(
    dense_ranked: List[RetrievalResult],
    keyword_ranked: List[RetrievalResult],
    k: int = 60,
) -> List[RetrievalResult]:
    """Reciprocal Rank Fusion: merge two ranked lists by summed 1/(k+rank).

    Deterministic, order-stable per list; equal fused scores keep dense-first order.
    """
    rank_of = {}
    for ranked, offset in ((dense_ranked, 0), (keyword_ranked, 1)):
        for i, res in enumerate(ranked):
            key = (res.content, res.metadata.get("source"))
            rank_of.setdefault(key, 0.0)
            rank_of[key] += 1.0 / (k + i + 1)

    # preserve the original RetrievalResult objects, not just content
    seen = set()
    order = []
    for ranked in (dense_ranked, keyword_ranked):
        for res in ranked:
            key = (res.content, res.metadata.get("source"))
            if key not in seen and key in rank_of:
                seen.add(key)
                order.append(res)

    order.sort(key=lambda r: rank_of[(r.content, r.metadata.get("source"))], reverse=True)
    return order


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
        self._bm25 = None

    def add_documents(self, documents: List[dict], embeddings: List[List[float]]):
        ids = []
        for i, doc in enumerate(documents):
            meta = doc.get("metadata", {})
            source = str(meta.get("source", "unknown"))
            chunk = meta.get("chunk_index", i)
            ids.append(f"{source}::chunk_{chunk}")
        self.collection.add(
            documents=[doc["content"] for doc in documents],
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in documents],
            ids=ids,
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

    def _ensure_bm25(self) -> BM25Okapi:
        """Build/lazy-cache a BM25 index over all collection documents (id -> original order)."""
        if self._bm25 is None:
            all_docs = self.collection.get(include=["documents", "metadatas"])
            docs = all_docs["documents"]
            self._bm25_metas = all_docs["metadatas"]
            self._bm25_corpus = docs
            self._bm25 = BM25Okapi([_tokenize(d) for d in docs])
        return self._bm25

    def search_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
        keyword_top_k: int = 10,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """Hybrid retrieval: BM25 keyword + dense vector, fused with Reciprocal Rank Fusion."""
        dense = self.search(query_embedding, top_k=top_k, threshold=threshold)

        bm25 = self._ensure_bm25()
        tok = _tokenize(query)
        if not tok:
            return dense
        scores = bm25.get_scores(tok)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        keyword_results = []
        for i in order[:keyword_top_k]:
            if scores[i] <= 0:
                continue
            keyword_results.append(
                RetrievalResult(
                    content=self._bm25_corpus[i],
                    score=float(scores[i]),
                    metadata=self._bm25_metas[i],
                )
            )

        return rrf(dense, keyword_results, k=rrf_k)[:top_k]

    def get_collection_stats(self) -> dict:
        return {"count": self.collection.count()}
