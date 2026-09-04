"""Eval Harness: single-command orchestrator that runs all RAG evaluation.

Orchestrates (reuses, does not duplicate) the two existing metric families:
  1. Low-cost metrics  -> run_evaluation.run_evaluation()
  2. RAGAS LLM-judge   -> run_ragas_evaluation.run_ragas_evaluation()

Combines both into one run + generates a unified markdown report.

Usage:
  python -m src.evaluation.harness --backend openrouter --sample N
  python -m src.evaluation.harness --backend ollama
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.mlops.logging_setup import setup_logging
from src.mlops.tracking import ExperimentTracker
from src.evaluation.run_evaluation import run_evaluation
from src.evaluation.run_ragas_evaluation import run_ragas_evaluation


def run_harness(
    backend: str = "ollama",
    sample: int = 0,
    judge: str = None,
    judge_relevancy: str = None,
    ctx_mode: str = "top1",
):
    config = load_config()
    log = setup_logging(config.logging.level, "tmp/logs/harness.log")
    log.info(
        f"Eval harness start: backend={backend} sample={sample} "
        f"ctx_mode={ctx_mode} judge={judge or 'default'}"
    )

    t0 = time.time()

    # Phase 1: low-cost metrics (no LLM judge)
    lowcost_agg = run_evaluation(sample=sample)
    lowcost_s = time.time() - t0

    # Phase 2: RAGAS LLM-judge metrics
    t1 = time.time()
    ragas_agg = run_ragas_evaluation(
        sample=sample,
        judge_model=judge,
        judge_relevancy=judge_relevancy,
        judge_backend=backend,
    )
    ragas_s = time.time() - t1

    total_s = time.time() - t0

    # Phase 3: merge + persist one unified summary under a single summary run
    merged = {**lowcost_agg, **ragas_agg}
    tracker = ExperimentTracker(
        tracking_uri=config.mlops.tracking_uri,
        experiment_name=config.mlops.experiment_name,
    )
    tracker.start_run(
        run_name="eval-harness-summary",
        tags={
            "task": "eval-harness",
            "judge_backend": backend,
            "ctx_mode": ctx_mode,
            "lowcost_phase_s": round(lowcost_s, 3),
            "ragas_phase_s": round(ragas_s, 3),
        },
    )
    tracker.log_params({
        "sample": sample,
        "backend": backend,
        "ctx_mode": ctx_mode,
        "judge": judge or "default",
    })
    tracker.log_metrics(merged | {"harness_total_s": total_s})
    tracker.end_run()

    log.info(
        f"Harness done: lowcost={lowcost_s:.1f}s ragas={ragas_s:.1f}s "
        f"total={total_s:.1f}s"
    )
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "openrouter"],
                        default="ollama", help="RAGAS judge backend")
    parser.add_argument("--sample", type=int, default=0,
                        help="number of questions (0 = all)")
    parser.add_argument("--judge", default=None,
                        help="RAGAS faithfulness judge model (else env/default)")
    parser.add_argument("--judge-relevancy", default=None,
                        help="RAGAS relevancy judge model (defaults to --judge)")
    parser.add_argument("--ctx-mode", choices=["top1", "full"], default="top1",
                        help="context mode (reserved; top1 default)")
    args = parser.parse_args()

    merged = run_harness(
        backend=args.backend,
        sample=args.sample,
        judge=args.judge,
        judge_relevancy=args.judge_relevancy,
        ctx_mode=args.ctx_mode,
    )
    print("\n=== EVAL HARNESS AGGREGATED ===")
    for k, v in merged.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
