# Insurance Knowledge Assistant

Production-style **Retrieval-Augmented Generation (RAG)** knowledge assistant with a **dual-track LLM evaluation harness**, **MLOps (MLflow)** experiment tracking, and production-observability design — built for the **Claims and Service DS Knowledge & Development Solutions (KDS)** use case.

> Turns insurance documents (PDF/TXT) into grounded, auditable Q&A with quantified retrieval + generation quality.

---

## Why this project stands out

| Differentiator | What it does |
|----------------|--------------|
| **Dual-track evaluation harness** | Unifies low-cost metrics (hit/MRR/precision@k/recall@k/ROUGE/BERTScore/latency) **and** RAGAS LLM-as-judge metrics (faithfulness/answer_relevancy/context_recall/context_precision/answer_correctness) in one engine |
| **LLM-as-judge with auto-rotation** | Pluggable judge backend (local Ollama + OpenRouter free) with **fixed primary + auto-rotate on failure** — cost-efficient and robust to quota/rate limits |
| **4 failure-layer eval coverage** | Context precision/recall (retrieval) + faithfulness/answer_relevancy (generation) + answer correctness |
| **Resumable long-horizon evaluation** | Per-question progress JSON survives restarts (13h+ local jobs) |
| **MLflow experiment tracking** | Structured experiment/run/param/metric/tag recording with an orchestrator harness |
| **Insurance-domain focus** | Grounded answers + citation grounding built for auditability and compliance |

## Features

- 📄 **Document Ingestion** — load insurance docs (PDF/TXT), chunk into semantic units
- 🔍 **Semantic Search** — vector similarity retrieval (ChromaDB, local all-MiniLM-L6-v2 embeddings)
- 🤖 **RAG Answer Generation** — LLM answers with retrieved context + source citations
- 🧪 **Dual-track Evaluation** — low-cost + RAGAS semantics in one `harness`
- ⚖️ **LLM-as-Judge** — pluggable + auto-rotating judges (local Ollama / OpenRouter free)
- 📊 **MLOps** — MLflow experiment tracking, logging, structured report generation
- 🧰 **Pluggable Embeddings** — swap cloud (Vertex AI) / local (sentence-transformers)

## Current Evaluation Results (RAGAS, 10-question insurance benchmark)

| Metric | Avg | Failure layer |
|--------|-----|---------------|
| answer_correctness | **0.87** | answer quality |
| answer_relevancy | **0.85** | answer quality |
| context_precision | **0.90** | retrieval |
| context_recall * | **0.75** | retrieval |
| faithfulness | **0.70** | groundedness / hallucination |

\* recall computed on `core_top1` (top-1 context) — a conservative minimal-context baseline; full `top_k` context available via `--ctx-mode full`.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (Python) |
| Generator LLM | Vertex AI Gemini 2.5 Flash / OpenRouter |
| Judge LLM | Ollama (local gemma3) / OpenRouter (minimax-m3:free) |
| Embeddings | sentence-transformers (local) / Vertex AI |
| Vector DB | ChromaDB |
| Evaluation | Low-cost custom metrics + RAGAS 0.4.3 |
| MLOps / tracking | MLflow (SQLite local) |

## Key Skills Demonstrated (ATS keywords)

RAG · retrieval-augmented generation · chunking · embeddings · semantic search · hybrid search · vector database · ChromaDB · LLM application development · prompt engineering · structured output · function calling · LLM-as-a-judge · RAGAS · evaluation / evals · faithfulness · context precision · context recall · answer relevancy · answer correctness · MRR · recall@k · precision@k · ROUGE · BERTScore · retrieval quality metrics · regression testing · MLOps · MLflow · experiment tracking · model evaluation · observability · model monitoring · LLMOps · metadata · FastAPI · REST API · Python · Pydantic · YAML configuration · TypeScript · testability · golden dataset · benchmark

## Architecture

```
Insurance Docs ──► Chunking ──► Embedding ──► ChromaDB (vector store)
                                                       ▲
User Query ──► Embed ──► Vector Search ────────┘
        │
        └──► [Eval Harness] ──► Low-cost metrics + RAGAS LLM-judge metrics ──► MLflow
```

## Project Structure

```
insurance-rag-agent/
├── src/
│   ├── api/               # FastAPI endpoints (ingest / query)
│   ├── rag/               # RAG pipeline
│   │   ├── ingestion.py   # document loading & chunking
│   │   ├── embedding.py   # embedding abstraction (Vertex/local)
│   │   ├── retrieval.py   # ChromaDB vector store & search
│   │   └── generation.py  # LLM answer generation
│   ├── evaluation/        # dual-track eval harness
│   │   ├── harness.py             # orchestrator (low-cost + RAGAS)
│   │   ├── run_evaluation.py      # low-cost metrics
│   │   ├── run_ragas_evaluation.py# RAGAS LLM-judge metrics
│   │   ├── metrics.py             # low-cost metric functions
│   │   └── benchmark_questions.py # 10-question golden set
│   ├── mlops/
│   │   ├── tracking.py            # MLflow ExperimentTracker wrapper
│   │   └── logging_setup.py       # structured logging
│   └── config.py           # type-safe config loading
├── configs/config.yaml    # YAML configuration
├── data/                  # sample insurance documents
└── tests/                 # unit & integration tests
```

## Run the Evaluation Harness

```bash
# Low-cost + RAGAS dual-track, LLM judge from OpenRouter (minimax-m3:free)
python -m src.evaluation.harness --backend openrouter --sample 10

# Low-cost only
python -m src.evaluation.run_evaluation --sample 10

# RAGAS LLM-judge with explicit model + context mode
python -m src.evaluation.run_ragas_evaluation --backend openrouter \
    --judge minimax/minimax-m3:free --ctx-mode top1
```

Results are recorded to MLflow (`sqlite:///mlflow.db`); launch `mlflow ui` to inspect runs.

## Getting Started

### Prerequisites

- Python 3.10+
- GCP account with Vertex AI (for Gemini generator) — or OpenRouter key for judge
- ChromaDB (local)

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GCP / OpenRouter credentials
```

### Run the API

```bash
uvicorn src.api.main:app --reload
```

Swagger docs at `http://localhost:8000/docs`

### Ingest & Query

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/sample_policy.txt"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does standard homeowners insurance cover?"}'
```

## Roadmap

- [x] Project foundation & FastAPI skeleton
- [x] RAG core pipeline (ingestion/embedding/retrieval/generation)
- [x] Dual-track evaluation harness (low-cost + RAGAS 5 metrics)
- [x] LLM-as-judge with auto-rotation
- [x] MLflow experiment tracking
- [x] Hybrid retrieval (BM25 + dense + RRF)
- [x] Cross-encoder reranker
- [ ] Citation grounding (real source references)
- [ ] DeepEval integration (agent/tool-call eval)
- [ ] PII masking + prompt-injection guardrails (insurance compliance)
- [ ] pytest + CI regression gate
- [ ] Prometheus monitoring + cost tracking
- [ ] Docker + Cloud Run deployment
- [ ] Expand golden dataset to ~50 QA pairs

## License

[MIT](LICENSE)
