import json
from types import SimpleNamespace
from pathlib import Path

from src.evaluation.sample_artifact import (
    benchmark_digest,
    corpus_digest,
    load_samples,
    save_samples,
    sample_digest,
    sample_key,
    samples_path,
)


def _q(query="Is liability covered by umbrella insurance we sell?",
       reference="Umbrella extends liability above primary policies.",
       source="umbrella_insurance.txt"):
    return {
        "query": query,
        "reference": reference,
        "expected_source": source,
        "relevant_sources": [source],
    }


def _cfg(**over):
    rag = dict(
        retrieval_mode="hybrid", top_k=5, similarity_threshold=0.4,
        keyword_top_k=10, rrf_k=60, rerank=False, reranker_top_k=10,
    )
    rag.update(over)
    return SimpleNamespace(
        rag=SimpleNamespace(**rag),
        llm=SimpleNamespace(provider="vertexai", model="gemini-2.5-flash", temperature=0.3),
        embedding=SimpleNamespace(provider="local", model="all-MiniLM-L6-v2"),
        vectorstore=SimpleNamespace(collection_name="insurance_docs", persist_dir="./data/chroma_db"),
    )


def _data(tmp_path):
    d = Path(tmp_path) / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.txt").write_text("umbrella liability excess")
    (d / "b.txt").write_text("flood exclusions apply")
    return d


def _sample(i=0):
    return {
        "user_input": f"q{i}",
        "response": f"answer{i}",
        "retrieved_contexts": ["ctx"],
        "reference": "ref",
    }


def test_benchmark_digest_stable_for_same_questions():
    qs = [_q(), _q("second question?", "ref2", "auto_insurance.txt")]
    assert benchmark_digest(qs) == benchmark_digest(qs)


def test_benchmark_digest_changes_with_question_text():
    qs = [_q()]
    changed = [_q(query="completely different wording")]
    assert benchmark_digest(qs) != benchmark_digest(changed)


def test_corpus_digest_changes_with_content():
    d1 = _data(Path("/tmp/cd_a"))
    d2 = _data(Path("/tmp/cd_b"))
    (d1 / "a.txt").write_text("umbrella liability excess")
    (d2 / "a.txt").write_text("umbrella liability excess CHANGED")
    assert corpus_digest(d1) != corpus_digest(d2)


def test_sample_digest_changes_when_rerank_flips():
    tmp = Path("/tmp/sd_tmp")
    tmp.mkdir(exist_ok=True)
    qs = [_q()]
    d1 = sample_digest(_cfg(), qs, data_dir=tmp)
    d2 = sample_digest(_cfg(rerank=True), qs, data_dir=tmp)
    assert d1 != d2


def test_sample_digest_changes_when_benchmark_count_changes():
    tmp = Path("/tmp/sd_tmp2")
    tmp.mkdir(exist_ok=True)
    d1 = sample_digest(_cfg(), [_q()], data_dir=tmp)
    d2 = sample_digest(_cfg(), [_q(), _q("extra?", "r", "auto_insurance.txt")], data_dir=tmp)
    assert d1 != d2


def test_sample_key_contains_all_components():
    key = sample_key(_cfg(), [_q()], data_dir=Path("/tmp/sk"))
    for field in ("benchmark_digest", "corpus_digest", "retrieval_mode",
                  "rerank", "top_k", "generator", "embedding", "collection"):
        assert field in key


def test_samples_path_has_digest_suffix(tmp_path):
    path = samples_path(_cfg(), [_q()], data_dir=tmp_path)
    assert path.name.startswith("samples_") and path.name.endswith(".json")


def test_save_then_load_roundtrip(tmp_path):
    qs = [_q()]
    path = tmp_path / "samples_abc.json"
    save_samples(path, [_sample(0)])
    loaded = load_samples(path, qs)
    assert loaded == [{**_sample(0)}]


def test_load_returns_none_when_missing(tmp_path):
    qs = [_q()]
    assert load_samples(tmp_path / "nope.json", qs) is None


def test_load_returns_none_when_wrong_count(tmp_path):
    path = tmp_path / "samples_x.json"
    save_samples(path, [_sample(0), _sample(1)])
    assert load_samples(path, [_q()]) is None  # 2 stored vs 1 expected


def test_load_returns_none_when_fields_missing(tmp_path):
    path = tmp_path / "samples_y.json"
    save_samples(path, [{"user_input": "q", "response": "a"}])
    assert load_samples(path, [_q()]) is None


def test_save_is_atomic_rename(tmp_path):
    path = tmp_path / "samples_z.json"
    save_samples(path, [_sample(0)])
    assert not Path(str(path) + ".tmp").exists()  # tmp cleaned up
    assert json.loads(path.read_text()) == [_sample(0)]