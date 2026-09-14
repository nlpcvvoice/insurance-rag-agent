"""Shared, keyed sample artifacts for the two evaluation tracks.

A "prepared sample" is one benchmark question ready for scoring: the
question text, the retrieved context(s) used for generation, the generated
answer, and the gold reference. Both evaluation tracks score the SAME
samples:

- low-cost track (``run_evaluation.py``) generates + persists the artifact;
- RAGAS track (``run_ragas_evaluation.py``) reuses it with zero
  re-generation / re-retrieval.

The cache file name embeds a digest of every input that determines what a
generation pass produces (retrieval mode, rerank, corpus, benchmark,
generator). Changing any key component invalidates the cache instead of
silently reusing stale data.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional

SAMPLE_FIELDS = ("user_input", "response", "retrieved_contexts", "reference")


def benchmark_digest(questions) -> str:
    """Stable digest of the question set (query/reference/gold sources)."""
    payload = json.dumps(
        [
            {
                "query": q.get("query"),
                "reference": q.get("reference"),
                "expected_source": q.get("expected_source"),
                "relevant_sources": q.get("relevant_sources"),
            }
            for q in questions
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def corpus_digest(data_dir) -> str:
    """Stable digest of the source documents that build the vector corpus."""
    files = sorted(
        p for p in Path(data_dir).glob("*")
        if p.suffix.lower() in {".txt", ".pdf", ".md"}
    )
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _data_dir(config, data_dir):
    if data_dir is None:
        return Path(config.vectorstore.persist_dir).parent
    return Path(data_dir)


def sample_key(config, questions, data_dir=None) -> dict:
    """Every input a single generation pass depends on (key components)."""
    return {
        "benchmark_questions": len(questions),
        "benchmark_digest": benchmark_digest(questions),
        "corpus_digest": corpus_digest(_data_dir(config, data_dir)),
        "embedding": f"{config.embedding.provider}:{config.embedding.model}",
        "collection": config.vectorstore.collection_name,
        "retrieval_mode": config.rag.retrieval_mode,
        "top_k": config.rag.top_k,
        "similarity_threshold": config.rag.similarity_threshold,
        "keyword_top_k": config.rag.keyword_top_k,
        "rrf_k": config.rag.rrf_k,
        "rerank": config.rag.rerank,
        "reranker_top_k": config.rag.reranker_top_k,
        "generator": f"{config.llm.provider}:{config.llm.model}:{config.llm.temperature}",
    }


def sample_digest(config, questions, data_dir=None) -> str:
    key = sample_key(config, questions, data_dir)
    payload = "|".join(f"{k}={key[k]}" for k in sorted(key))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def samples_path(config, questions, data_dir=None) -> Path:
    return (
        Path(__file__).resolve().parent
        / f"samples_{sample_digest(config, questions, data_dir)}.json"
    )


def load_samples(path: Path, questions) -> Optional[List[dict]]:
    """Return prepared samples if the artifact exists with the right count
    and every entry carries the required fields; otherwise None.

    Validating count + fields is what prevents reusing a stale artifact
    after the benchmark set or an earlier partial run changed shape.
    """
    if not Path(path).exists():
        return None
    try:
        samples = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if len(samples) != len(questions):
        return None
    if any(not all(k in s for k in SAMPLE_FIELDS) for s in samples):
        return None
    return samples


def save_samples(path: Path, samples) -> None:
    """Atomically write the sample artifact (tmp file then rename)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(samples))
    tmp.replace(path)