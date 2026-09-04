from typing import List, Dict
import time


def _is_relevant(result: Dict, sources: List[str]) -> bool:
    """True if a retrieved result's source matches any relevant source."""
    src = (result.get("metadata") or {}).get("source", "")
    return any(s in src for s in sources)


def retrieval_hit_rate(
    retrieved: List[Dict],
    expected_source: str,
) -> float:
    """Return 1.0 if expected_source appears in retrieved results, else 0.0."""
    return _is_relevant(retrieved[0], [expected_source]) if retrieved else 0.0


def hit_rate_at_k(
    retrieved: List[Dict],
    expected_source: str,
    k: int,
) -> float:
    """Return 1.0 if a relevant doc appears within the first k results."""
    slice_ = retrieved[:k]
    return 1.0 if any(_is_relevant(r, [expected_source]) for r in slice_) else 0.0


def recall_at_k(
    retrieved: List[Dict],
    relevant_sources: List[str],
    k: int,
) -> float:
    """Fraction of relevant docs found within the top k results."""
    slice_ = retrieved[:k]
    found = set()
    for r in slice_:
        src = (r.get("metadata") or {}).get("source", "")
        for s in relevant_sources:
            if s in src:
                found.add(s)
    return len(found) / len(relevant_sources) if relevant_sources else 0.0


def precision_at_k(
    retrieved: List[Dict],
    relevant_sources: List[str],
    k: int,
) -> float:
    """Fraction of the top k results that are relevant."""
    slice_ = retrieved[:k]
    if not slice_:
        return 0.0
    count = sum(1 for r in slice_ if _is_relevant(r, relevant_sources))
    return count / len(slice_)


def mrr(
    retrieved: List[Dict],
    relevant_sources: List[str],
) -> float:
    """Reciprocal rank of the first relevant result (0 if none found)."""
    for i, r in enumerate(retrieved, start=1):
        if _is_relevant(r, relevant_sources):
            return 1.0 / i
    return 0.0


def answer_rouge(
    generated: str,
    reference: str,
    rouge_type: str = "rougeL",
) -> float:
    """Compute ROUGE-L F1 between generated and reference answer."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        score = scorer.score(reference, generated)
        return score[rouge_type].fmeasure
    except Exception:
        return 0.0


def answer_bertscore(
    generated: str,
    reference: str,
    model_name: str = "distilbert-base-uncased",
) -> float:
    """Semantic similarity (BERTScore F1) between generated and reference."""
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(
            [generated], [reference], model_type=model_name, lang="en",
            verbose=False,
        )
        return float(f1[0])
    except Exception:
        return 0.0


def measure_latency(
    retrieve_fn,
    generate_fn,
    query: str,
    query_embedding: List[float],
):
    """Measure retrieval and generation latency in seconds."""
    t0 = time.perf_counter()
    results = retrieve_fn(query_embedding)
    retrieval_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    contexts = [r["content"] for r in results]
    answer = generate_fn(query, contexts)
    generation_s = time.perf_counter() - t0

    return {
        "retrieval_s": retrieval_s,
        "generation_s": generation_s,
        "total_s": retrieval_s + generation_s,
        "answer": answer,
        "retrieved": results,
    }
