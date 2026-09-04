import pytest
from src.rag.retrieval import RetrievalResult
from src.rag.reranker import CrossEncoderReranker


def _res(content, source="doc"):
    return RetrievalResult(content=content, score=1.0, metadata={"source": source})


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_rerank_returns_top_k_preserving_objects(reranker):
    results = [
        _res("Policy covers fire damage basics.", "a"),
        _res("Unrelated about car rental terms.", "b"),
        _res("Homeowners dwelling coverage details.", "c"),
    ]
    reranked = reranker.rerank("what does dwelling coverage cover", results, top_k=2)
    assert len(reranked) == 2
    # cross-encoder should rank the dwelling-related doc first
    assert reranked[0].content.startswith("Homeowners dwelling")


def test_rerank_empty_inputs(reranker):
    assert reranker.rerank("any query", []) == []


def test_rerank_top_k_none_keeps_all(reranker):
    results = [_res("one", "a"), _res("two", "b"), _res("three", "c")]
    reranked = reranker.rerank("one two three", results, top_k=None)
    assert len(reranked) == 3
