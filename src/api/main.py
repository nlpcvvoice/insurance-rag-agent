from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.config import load_config
from src.rag.ingestion import DocumentLoader
from src.rag.embedding import get_embedding_provider
from src.rag.retrieval import VectorStore
from src.rag.generation import LLMGenerator


app = FastAPI(
    title="Insurance Knowledge Assistant",
    description="RAG-based insurance knowledge assistant",
    version="0.1.0",
)

config = load_config()
loader = DocumentLoader(
    chunk_size=config.rag.chunk_size,
    chunk_overlap=config.rag.chunk_overlap,
)
embedding_provider = get_embedding_provider(
    provider=config.embedding.provider,
    model=config.embedding.model,
)
vector_store = VectorStore(
    collection_name=config.vectorstore.collection_name,
    persist_dir=config.vectorstore.persist_dir,
)
generator = LLMGenerator(
    model=config.llm.model,
    temperature=config.llm.temperature,
)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    model: str


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    message: str
    num_chunks: int


@app.get("/")
def root():
    return {"message": "Insurance Knowledge Assistant API"}


@app.get("/health")
def health():
    stats = vector_store.get_collection_stats()
    return {"status": "healthy", "documents": stats["count"]}


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    try:
        documents = loader.load_file(request.file_path)
        contents = [doc.content for doc in documents]
        embeddings = embedding_provider.embed_texts(contents)

        doc_dicts = [
            {"content": doc.content, "metadata": doc.metadata, "source": doc.source}
            for doc in documents
        ]
        vector_store.add_documents(doc_dicts, embeddings)

        return IngestResponse(
            message=f"Successfully ingested {request.file_path}",
            num_chunks=len(documents),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        query_embedding = embedding_provider.embed_query(request.query)
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k if request.top_k else config.rag.top_k,
            threshold=config.rag.similarity_threshold,
        )

        contexts = [result.content for result in results]
        generation_result = generator.generate(
            query=request.query,
            context=contexts,
        )

        return QueryResponse(
            answer=generation_result.answer,
            sources=[result.metadata.get("source", "") for result in results],
            model=config.llm.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host if hasattr(config, "api") else "0.0.0.0",
        port=8000,
        reload=True,
    )
