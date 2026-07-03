"""
config.py
Centralized configuration for PolicyQA.
Nothing below should be hardcoded anywhere else in the codebase — if you
need to tune chunk size, model choice, or retrieval depth, change it here.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # reads .env if present


@dataclass(frozen=True)
class Settings:
    # --- API ---
    # Groq gives a free API key at console.groq.com — used ONLY for the LLM
    # (query rewriting + answer generation), which is low-volume (2 calls per
    # question). Embeddings are NOT sent to any API — see embedding_model below.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # --- Models ---
    # Hosted, free-tier, via Groq. llama-3.1-8b-instant is fast and within
    # generous free rate limits; swap to llama-3.3-70b-versatile for better
    # quality if you're not hitting rate limits.
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    # Runs LOCALLY on your machine, not an API call. Deliberate choice:
    # indexing a document fires one embedding call per chunk (potentially
    # dozens at once) — that volume would blow through a free API's
    # rate limit fast. Local avoids that entirely, at the cost of a one-time
    # model download (~1GB) and using your own CPU/RAM.
    # This one is multilingual (matches the project's multilingual requirement).
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # Also local, unchanged from before — cross-encoder reranking never
    # needed an API key in the first place.
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "models")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Retrieval ---
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "10"))      # FAISS results
    sparse_top_k: int = int(os.getenv("SPARSE_TOP_K", "10"))    # BM25 results
    rerank_candidate_k: int = int(os.getenv("RERANK_CANDIDATE_K", "20"))  # merged pool sent to reranker
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))       # chunks actually sent to the LLM
    ensemble_weights: tuple = (0.5, 0.5)  # (dense_weight, sparse_weight)

    # --- Storage ---
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "vector_store")

    # --- OCR ---
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")  # only needed on Windows, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe

    # --- Summarization ---
    summary_chunk_size: int = int(os.getenv("SUMMARY_CHUNK_SIZE", "3000"))
    summary_chunk_overlap: int = int(os.getenv("SUMMARY_CHUNK_OVERLAP", "300"))


settings = Settings()


def validate_settings() -> None:
    """Fail loudly at startup rather than silently at the first LLM call."""
    if not settings.groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys, "
            "then copy .env.example to .env and add it."
        )
