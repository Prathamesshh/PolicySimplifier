"""Download local ML models used by the Streamlit app."""

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import settings


def main() -> None:
    print(f"Caching embedding model: {settings.embedding_model}")
    HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.model_cache_dir,
    )

    print(f"Caching reranker model: {settings.reranker_model}")
    CrossEncoder(settings.reranker_model, cache_dir=settings.model_cache_dir)

    print(f"Done. Models cached under: {settings.model_cache_dir}")


if __name__ == "__main__":
    main()
