import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "vertexai"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.4


@dataclass
class VectorStoreConfig:
    provider: str = "chromadb"
    collection_name: str = "insurance_docs"
    persist_dir: str = "./data/chroma_db"


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


@dataclass
class Config:
    llm: LLMConfig
    rag: RAGConfig
    vectorstore: VectorStoreConfig
    embedding: EmbeddingConfig

    @classmethod
    def from_yaml(cls, config_path: str = "configs/config.yaml") -> "Config":
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            llm=LLMConfig(**raw.get("llm", {})),
            rag=RAGConfig(**raw.get("rag", {})),
            vectorstore=VectorStoreConfig(**raw.get("vectorstore", {})),
            embedding=EmbeddingConfig(**raw.get("embedding", {})),
        )


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "config.yaml")
    return Config.from_yaml(config_path)
