"""RAGAS LLM-as-Judge evaluation using a local Ollama model (resumable).

Flow:
1. For each benchmark question, run our RAG pipeline (local embedding retrieval
   + Gemini generation) to obtain retrieved contexts and the generated answer.
   These prepared samples are cached to disk so a restart does not re-run Gemini.
2. Score faithfulness + answer_relevancy with RAGAS, using a local Ollama
   model as the judge LLM (0 GCP credit).
3. Persist progress after every question (resumable) and log to MLflow.

Usage:
  python src/evaluation/run_ragas_evaluation.py --judge gemma4:12b [--sample N]
  python src/evaluation/run_ragas_evaluation.py --judge-backend openrouter \
      --judge openrouter/free --judge-relevancy openrouter/free
"""
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.rag.embedding import get_embedding_provider
from src.rag.retrieval import VectorStore
from src.rag.generation import LLMGenerator
from src.mlops.logging_setup import setup_logging, get_logger
from src.mlops.tracking import ExperimentTracker
from src.evaluation.benchmark_questions import BENCHMARK_QUESTIONS


def _prepare_samples(questions, config, log, cache_path: Path):
    """Build per-question prepared samples (answer + core context), cached."""
    if cache_path.exists():
        log.info(f"Loading prepared samples from cache: {cache_path}")
        return json.loads(cache_path.read_text())

    emb = get_embedding_provider(
        provider=config.embedding.provider, model=config.embedding.model,
    )
    store = VectorStore(
        collection_name=config.vectorstore.collection_name,
        persist_dir=config.vectorstore.persist_dir,
    )
    generator = LLMGenerator(
        model=config.llm.model, temperature=config.llm.temperature,
    )

    samples = []
    for i, q in enumerate(questions):
        query_vec = emb.embed_query(q["query"])
        if config.rag.retrieval_mode == "hybrid":
            results = store.search_hybrid(
                query=q["query"], query_embedding=query_vec,
                top_k=config.rag.top_k, threshold=config.rag.similarity_threshold,
                keyword_top_k=config.rag.keyword_top_k, rrf_k=config.rag.rrf_k,
            )
        else:
            results = store.search(
                query_vec, top_k=config.rag.top_k,
                threshold=config.rag.similarity_threshold,
            )
        contexts = [r.content for r in results]
        core_context = [contexts[0]] if contexts else []
        answer = generator.generate(query=q["query"], context=contexts).answer
        samples.append({
            "user_input": q["query"],
            "response": answer,
            "retrieved_contexts": core_context,
            "reference": q.get("reference", ""),
        })
        log.info(f"Prepared sample {i + 1}/{len(questions)}")
        cache_path.write_text(json.dumps(samples))  # incremental persist

    return samples


def run_ragas_evaluation(
    sample: int = 0,
    judge_model: str = None,
    judge_relevancy: str = None,
    judge_backend: str = "ollama",
):
    import os
    import warnings
    warnings.filterwarnings("ignore")

    # Load secrets from test/.env (outside the git repo) if present.
    # NOTE: never print the value of any secret; only booleans/lengths.
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        pass

    config = load_config()
    log = setup_logging(config.logging.level, "tmp/logs/ragas_evaluation.log")

    # Resolve judge models: explicit CLI arg wins, else env default (openrouter)
    # or gemma3:4b (local ollama). Never logs the API key value here.
    if not judge_model:
        if judge_backend == "openrouter":
            judge_model = os.environ.get("OPENROUTER_DEFAULT_MODEL", "openrouter/free").strip() or "openrouter/free"
        else:
            judge_model = "gemma3:4b"
    if not judge_relevancy:
        judge_relevancy = judge_model

    questions = BENCHMARK_QUESTIONS
    if sample > 0:
        questions = questions[:sample]

    cache_path = Path(__file__).resolve().parent / "ragas_samples.json"
    samples = _prepare_samples(questions, config, log, cache_path)

    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import SingleTurnSample
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    )
    from ragas.metrics._answer_similarity import AnswerSimilarity
    import asyncio, gc

    def _make_chat_llm(model=None):
        """Build a ChatOpenAI (OpenRouter) or ChatOllama (local) llm.
        For OpenRouter: if model is None, fall back to the OPENROUTER_DEFAULT_MODEL
        env var (from .env). Never logs the API key value."""
        if judge_backend == "openrouter":
            if not model:
                model = os.environ.get(
                    "OPENROUTER_DEFAULT_MODEL", "openrouter/free"
                ).strip() or "openrouter/free"
            api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            # only log presence/length, never the key itself
            log.info(
                f"Judge backend=openrouter model={model} key_set={bool(api_key)}"
            )
            timeout = int(os.environ.get("OPENROUTER_TIMEOUT", "120"))
            max_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", "3"))
            base_url = os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            return ChatOpenAI(
                model=model,
                api_key=api_key or "sk-or-v1-missing",
                base_url=base_url,
                temperature=0,
                timeout=timeout,
                max_retries=max_retries,
            )
        # local ollama backend (default)
        return ChatOllama(
            model=model, base_url="http://localhost:11434",
            temperature=0, num_ctx=16384, num_predict=2048,
        )

    # faithfullness uses the faithful judge model, answer_relevancy the relevancy model
    judge_faith = LangchainLLMWrapper(_make_chat_llm(judge_model))
    judge_rel = LangchainLLMWrapper(_make_chat_llm(judge_relevancy))
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Each metric carries its own judge LLM: they are scored in separate phases
    # so the heavy relevance model is only loaded at the end.
    faithfulness.llm = judge_faith
    answer_relevancy.llm = judge_rel
    answer_relevancy.embeddings = judge_emb
    context_recall.llm = judge_rel
    context_precision.llm = judge_rel
    answer_correctness.llm = judge_rel
    answer_correctness.embeddings = judge_emb
    answer_correctness.answer_similarity = AnswerSimilarity(
        embeddings=judge_emb
    )

    samples_ragas = [SingleTurnSample(**s) for s in samples]
    if sample > 0:
        samples_ragas = samples_ragas[:sample]

    tracker = ExperimentTracker(
        tracking_uri=config.mlops.tracking_uri,
        experiment_name=config.mlops.experiment_name,
    )
    tracker.start_run(
        run_name="ragas-judge",
        tags={"task": "ragas-llm-as-judge",
              "judge_faithfulness": judge_model,
              "judge_answer_relevancy": judge_relevancy,
              "generator_llm": config.llm.model,
              "judge_backend": judge_backend},
    )
    tracker.log_params({
        "judge_faithfulness_model": judge_model,
        "judge_answer_relevancy_model": judge_relevancy,
        "generator_llm": config.llm.model,
        "metrics": "faithfulness,answer_relevancy",
        "num_questions": len(questions),
        "context_mode": "core_top1",
    })

    # Progress persistence
    import re
    def _slug(s):
        return re.sub(r'[^A-Za-z0-9_.-]', '_', s)
    progress_path = Path(__file__).resolve().parent / f"ragas_progress_{_slug(judge_model)}_{_slug(judge_relevancy)}.json"
    progress = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())

    total_start = time.time()

    async def _score_one(s, m):
        return await m.single_turn_ascore(s)

    def score_one_sync(s, m):
        return asyncio.run(_score_one(s, m))

    # Score each metric in its own phase so the heavy relevancy model is only
    # loaded after faithfulness (light model) finishes.
    metric_phases = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    # Fallback free models used ONLY on transient failures (402/502/timeout)
    # of the primary judge model. Rotation is transient: on the *next* question
    # we always go back to the primary model (judge_model) if it succeeds.
    FALLBACK_MODELS = (
        "z-ai/glm-5.2:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ) if judge_backend == "openrouter" else ()

    for m in metric_phases:
        if sample > 0 and m is answer_relevancy:
            # keep small-sample runs light
            pass
        log.info(f"=== Scoring metric: {m.name} ===")
        phase_start = time.time()
        n_scored = 0
        for i in range(len(samples_ragas)):
            key = f"q{i}_{m.name}"
            if key in progress and progress[key] is not None:
                n_scored += 1
                continue
            log.info(f"  [{m.name}] Q{i}: {questions[i]['query'][:40]}...")
            # Start with the primary model; rotate through fallbacks only if it
            # fails, then reset back to the primary model for the next question.
            val = None
            attempts = [judge_model] + list(FALLBACK_MODELS)
            for attempt_model in attempts:
                if judge_backend == "openrouter":
                    m.llm = LangchainLLMWrapper(_make_chat_llm(attempt_model))
                    if attempt_model != judge_model:
                        log.info(
                            f"  [{m.name}] Q{i}: rotating judge -> {attempt_model}"
                        )
                try:
                    val = score_one_sync(samples_ragas[i], m)
                    break  # success, keep this model for rest of question
                except Exception as e:
                    log.warning(
                        f"  [{m.name}] Q{i}: {attempt_model} failed: {e}"
                    )
                    val = None
            progress[key] = val
            progress_path.write_text(json.dumps(progress, indent=2))
            if val is not None:
                n_scored += 1
                tracker.log_metrics({f"q{i}_{m.name}": float(val)})
        log.info(
            f"=== {m.name} phase done in {time.time() - phase_start:.0f}s "
            f"(scored {n_scored}/{len(samples_ragas)}) ==="
        )
        gc.collect()

    # Aggregated over any scored (non-None) values
    agg = {}
    for m in metric_phases:
        vals = []
        for i in range(len(samples_ragas)):
            v = progress.get(f"q{i}_{m.name}")
            if v is not None:
                vals.append(float(v))
        agg[f"avg_{m.name}"] = (sum(vals) / len(vals)) if vals else float("nan")
        agg[f"scored_{m.name}"] = len(vals)
    agg["total_elapsed_s"] = time.time() - total_start
    tracker.log_metrics(agg)
    tracker.end_run()

    log.info(f"RAGAS AGGREGATED: {agg}")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="number of questions (0 = all)")
    parser.add_argument("--judge", default=None,
                        help="judge model (faithfulness). openrouter backend uses .env OPENROUTER_DEFAULT_MODEL if omitted; ollama defaults gemma3:4b")
    parser.add_argument("--judge-relevancy", default=None,
                        help="judge model for answer_relevancy (defaults to --judge)")
    parser.add_argument("--judge-backend", choices=["ollama", "openrouter"], default="ollama",
                        help="judge LLM backend (default: ollama local)")
    args = parser.parse_args()
    run_ragas_evaluation(
        sample=args.sample,
        judge_model=args.judge,
        judge_relevancy=args.judge_relevancy,
        judge_backend=args.judge_backend,
    )
