import pytest
from src.rag.retrieval import rrf, _tokenize, RetrievalResult


def _res(content, source="doc"):
    return RetrievalResult(content=content, score=1.0, metadata={"source": source})


def test_tokenize_lowercase_split():
    assert _tokenize("HO-3 & Homeowners Deductible!") == [
        "ho", "3", "homeowners", "deductible"
    ]


def test_rrf_merges_and_ranks_by_fusion_score():
    dense = [_res("a", "d1"), _res("b", "d2")]
    keyword = [_res("b", "d2"), _res("c", "d3")]
    merged = rrf(dense, keyword, k=60)
    # "b" appears in both lists -> highest fused rank; "a"/"c" appear once
    assert [r.content for r in merged][0] == "b"
    assert {r.content for r in merged} == {"a", "b", "c"}


def test_rrf_deduplicates_same_item():
    dense = [_res("a", "d1"), _res("b", "d2")]
    keyword = [_res("a", "d1"), _res("c", "d3")]
    merged = rrf(dense, keyword, k=60)
    contents = [r.content for r in merged]
    # "a" duplicated across lists appears only once after fusion
    assert contents.count("a") == 1


def test_rrf_empty_inputs():
    assert rrf([], [], k=60) == []
