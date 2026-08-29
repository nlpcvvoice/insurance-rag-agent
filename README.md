# Insurance Knowledge Assistant

Insurance domain knowledge assistant built with **Retrieval-Augmented Generation (RAG)** and **LLM**. Helps employees find, learn, and use insurance information quickly through natural language Q&A.

> Built for the **Claims and Service DS Knowledge & Development Solutions** use case — improving knowledge discovery, content effectiveness, and employee enablement.

## Features

- 📄 **Document Ingestion** — Load insurance docs (PDF/TXT), chunk into semantic units
- 🔍 **Semantic Search** — Find relevant documents via vector similarity (ChromaDB)
- 🤖 **RAG Answer Generation** — LLM (Gemini) answers with retrieved context + citations
- 🧩 **Pluggable Embeddings** — Swap between Vertex AI (cloud) and local sentence-transformers
- 🧪 **Evaluation** — Measure retrieval quality, answer accuracy, latency
- 📊 **MLOps** — Experiment tracking, monitoring, logging

## Architecture

```
User Query ──► Embedding ──► Vector Search (ChromaDB) ──► Context
                                                    │
User Query ───────────────────────────────────────────┼──► LLM (Gemini) ──► Answer + Sources
                                                      ▲
Insurance Docs ──► Chunking ──► Embedding ──► Vector Store
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI (Python) |
| LLM | Vertex AI Gemini |
| Embeddings | Vertex AI / sentence-transformers |
| Vector DB | ChromaDB |
| MLOps | MLflow |
| Testing | pytest |

## Project Structure

```
insurance-rag-agent/
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── rag/           # RAG pipeline
│   │   ├── ingestion.py    # Document loading & chunking
│   │   ├── embedding.py    # Embedding abstraction (Vertex/local)
│   │   ├── retrieval.py    # ChromaDB vector store & search
│   │   └── generation.py   # LLM answer generation
│   ├── config.py      # Type-safe config loading
│   ├── evaluation/    # Metrics & evaluation
│   └── mlops/         # Experiment tracking, monitoring
├── data/              # Sample insurance documents
├── tests/             # Unit & integration tests
├── configs/           # YAML configuration
└── docs/              # Architecture & documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- Google Cloud account with Vertex AI enabled

### Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your GCP credentials
```

### Run the API

```bash
uvicorn src.api.main:app --reload
```

Swagger docs at `http://localhost:8000/docs`

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./data/sample_policy.txt"}'
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does standard homeowners insurance cover?"}'
```

### Run Tests

```bash
pytest tests/ -v
```

## Configuring Embeddings

Edit `configs/config.yaml`:

```yaml
embedding:
  provider: vertexai      # or "local"
  model: text-embedding-004
```

- **vertexai** — cloud embeddings via GCP
- **local** — on-device via sentence-transformers (free, no API)

## Roadmap

- [x] Project foundation & FastAPI skeleton
- [x] RAG core pipeline (ingestion/embedding/retrieval/generation)
- [x] Basic unit tests
- [ ] Evaluation metrics (retrieval quality, answer accuracy)
- [ ] MLOps (experiment tracking, monitoring, logging)
- [ ] Cloud Run deployment
- [ ] End-to-end integration tests

## License

[MIT](LICENSE)
