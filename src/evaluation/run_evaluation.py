"""Run the RAG benchmark and log aggregated metrics to MLflow.

Also persists the generated samples into the shared, keyed sample artifact
so the RAGAS track can score the SAME answers with zero re-generation.
Answers from a valid existing artifact are reused (no Gemini call)."""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.rag.embedding import get_embedding_provider
from src.rag.retrieval import VectorStore
from src.rag.generation import LLMGenerator
from src.mlops.logging_setup import setup_logging, get_logger
from src.mlops.tracking import ExperimentTracker
from src.evaluation.metrics import (
    retrieval_hit_rate,
    hit_rate_at_k,
    recall_at_k,
    precision_at_k,
    mrr,
    answer_rouge,
    answer_bertscore,
)
from src.evaluation.benchmark_questions import BENCHMARK_QUESTIONS
from src.evaluation.sample_artifact import (
    load_samples,
    save_samples,
    samples_path,
    sample_digest,
    sample_key,
)


def run_evaluation(sample: int = 0, config_path: Optional[str] = None):
    config = load_config(config_path)
    log = setup_logging(config.logging.level, "tmp/logs/evaluation.log")

    emb = get_embedding_provider(
        provider=config.embedding.provider,
        model=config.embedding.model,
    )
    store = VectorStore(
        collection_name=config.vectorstore.collection_name,
        persist_dir=config.vectorstore.persist_dir,
    )
    generator = LLMGenerator(
        model=config.llm.model,
        temperature=config.llm.temperature,
    )

    questions = BENCHMARK_QUESTIONS
    if sample > 0:
        questions = questions[:sample]

    artifact_path = samples_path(config, questions)
    cached = load_samples(artifact_path, questions)
    tracker = ExperimentTracker(
        tracking_uri=config.mlops.tracking_uri,
        experiment_name=config.mlops.experiment_name,
    )
    tracker.start_run(
        run_name="rag-evaluation",
        tags={"task": "rag-benchmark", "embedding": config.embedding.model,
              "llm": config.llm.model},
    )
    tracker.log_params({
        "embedding_provider": config.embedding.provider,
        "embedding_model": config.embedding.model,
        "llm_model": config.llm.model,
        "top_k": config.rag.top_k,
        "similarity_threshold": config.rag.similarity_threshold,
        "retrieval_mode": config.rag.retrieval_mode,
        "rerank": config.rag.rerank,
        "num_questions": len(questions),
        "sample_artifact": artifact_path.name,
        "sample_artifact_digest": sample_digest(config, questions),
        "samples_reused": cached is not None,
        "benchmark_digest": sample_key(config, questions)["benchmark_digest"],
        "corpus_digest": sample_key(config, questions)["corpus_digest"],
    })

    total_hit = 0.0
    total_hit3 = 0.0
    total_hit5 = 0.0
    total_mrr = 0.0
    total_prec5 = 0.0
    total_recall5 = 0.0
    total_rouge = 0.0
    total_bertscore = 0.0
    total_retrieval_s = 0.0
    total_generation_s = 0.0
    n_generated = 0

    shared = []

    for i, q in enumerate(questions):
        query = q["query"]
        query_vec = emb.embed_query(query)

        cached_answer = cached[i]["response"] if cached is not None else None

        from src.evaluation.metrics import measure_latency
        def retrieve(q, v):
            if config.rag.retrieval_mode == "hybrid":
                return [
                    {"content": r.content, "metadata": r.metadata}
                    for r in store.search_hybrid(
                        query=q,
                        query_embedding=v,
                        top_k=config.rag.top_k,
                        threshold=config.rag.similarity_threshold,
                        keyword_top_k=config.rag.keyword_top_k,
                        rrf_k=config.rag.rrf_k,
                        rerank=config.rag.rerank,
                        reranker_top_k=config.rag.reranker_top_k,
                    )
                ]
            return [
                {"content": r.content, "metadata": r.metadata}
                for r in store.search(
                    v,
                    top_k=config.rag.top_k,
                    threshold=config.rag.similarity_threshold,
                )
            ]
        def generate(query_, contexts_):
            if cached_answer is not None:
                return cached_answer
            return generator.generate(query=query_, context=contexts_).answer

        try:
            lat = measure_latency(retrieve, generate, query, query_vec)
        except Exception as e:
            log.warning(f"Question {i} failed (no context?): {e}")
            tracker.log_metrics({f"q{i}_hit": 0.0, f"q{i}_rouge": 0.0})
            shared.append({
                "user_input": query,
                "response": "",
                "retrieved_contexts": [],
                "reference": q.get("reference", ""),
            })
            continue

        shared.append({
            "user_input": query,
            "response": lat["answer"],
            "retrieved_contexts": [lat["retrieved"][0]["content"]] if lat["retrieved"] else [],
            "reference": q.get("reference", ""),
        })

        hit = retrieval_hit_rate(lat["retrieved"], q["expected_source"])
        hit3 = hit_rate_at_k(lat["retrieved"], q["expected_source"], k=3)
        hit5 = hit_rate_at_k(lat["retrieved"], q["expected_source"], k=5)
        relevant = q.get("relevant_sources", [q["expected_source"]])
        mr = mrr(lat["retrieved"], relevant)
        prec5 = precision_at_k(lat["retrieved"], relevant, k=5)
        rec5 = recall_at_k(lat["retrieved"], relevant, k=5)
        rouge = answer_rouge(lat["answer"], q["reference"])
        bs = answer_bertscore(lat["answer"], q["reference"])

        total_hit += hit
        total_hit3 += hit3
        total_hit5 += hit5
        total_mrr += mr
        total_prec5 += prec5
        total_recall5 += rec5
        total_rouge += rouge
        total_bertscore += bs
        total_retrieval_s += lat["retrieval_s"]
        if cached_answer is None:
            # generation latency is only real when we actually generated;
            # reused answers (warm run) do not count toward generation stats
            total_generation_s += lat["generation_s"]
            n_generated += 1

        log.info(
            f"Q{i}: hit={hit} mrr={mr:.2f} prec@5={prec5:.2f} "
            f"recall@5={rec5:.2f} rouge={rouge:.3f} bert={bs:.3f} "
            f"retr={lat['retrieval_s']:.2f}s gen={lat['generation_s']:.2f}s"
        )
        tracker.log_metrics({
            f"q{i}_hit": hit,
            f"q{i}_hit@3": hit3,
            f"q{i}_hit@5": hit5,
            f"q{i}_mrr": mr,
            f"q{i}_precision@5": prec5,
            f"q{i}_recall@5": rec5,
            f"q{i}_rouge": rouge,
            f"q{i}_bertscore": bs,
            f"q{i}_retrieval_s": lat["retrieval_s"],
            f"q{i}_generation_s": lat["generation_s"],
        })

    n = len(questions)
    agg = {
        "avg_retrieval_hit_rate": total_hit / n,
        "avg_hit@3": total_hit3 / n,
        "avg_hit@5": total_hit5 / n,
        "avg_mrr": total_mrr / n,
        "avg_precision@5": total_prec5 / n,
        "avg_recall@5": total_recall5 / n,
        "avg_answer_rougeL": total_rouge / n,
        "avg_answer_bertscore": total_bertscore / n,
        "avg_retrieval_s": total_retrieval_s / n,
        "avg_generation_s": total_generation_s / n_generated if n_generated else float("nan"),
        "avg_total_s": (total_retrieval_s / n) + (total_generation_s / n_generated if n_generated else float("nan")),
        "questions_generated": n_generated,
    }

    if cached is None:
        # No valid artifact existed: persist what we just generated so the
        # RAGAS track can score these SAME answers later without re-generating.
        save_samples(artifact_path, shared)
        log.info(f"Persisted shared sample artifact: {artifact_path.name}")
    else:
        log.info(f"Reused cached answers from artifact: {artifact_path.name}")

    tracker.log_metrics(agg)
    tracker.end_run()

    log.info(f"AGGREGATED: {agg}")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="number of questions (0 = all)")
    parser.add_argument("--config", default=None,
                        help="path to a YAML config file (default: configs/config.yaml)")
    args = parser.parse_args()
    agg = run_evaluation(sample=args.sample, config_path=args.config)
    for k, v in agg.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
