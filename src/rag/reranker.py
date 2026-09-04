from typing import List, Optional
from .retrieval import RetrievalResult


class CrossEncoderReranker:
    """Re-rank retrieved results with a cross-encoder (query, doc) relevance model.

    A cross-encoder jointly encodes query+document and outputs a single relevance
    score, giving more accurate ranking than dot-product (bi-encoder) similarity.
    Model is loaded lazily on first use to keep startup cheap.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Re-rank results by cross-encoder relevance and return the top_k.

        If results is empty, returns it unchanged. Each result keeps its original
        object; only the order (and score) changes. If top_k is None, keep all.
        """
        if not results:
            return results

        top_k = len(results) if top_k is None else min(top_k, len(results))
        model = self._load()
        pairs = [(query, r.content) for r in results]
        scores = model.predict(pairs)

        ranked = sorted(zip(results, scores), key=lambda rs: rs[1], reverse=True)
        derived = [
            RetrievalResult(
                content=res.content,
                score=float(sc),
                metadata=res.metadata,
            )
            for res, sc in ranked[:top_k]
        ]
        return derived
