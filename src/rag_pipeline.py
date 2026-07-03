"""
rag_pipeline.py
The actual retrieval-augmented generation logic:
  1. Dual-indexing: FAISS (dense/semantic) + BM25 (sparse/keyword)
  2. Hybrid retrieval: EnsembleRetriever merges both with weighted RRF
  3. Cross-encoder reranking: re-scores the merged pool, keeps the top N
  4. History-aware query rewriting: follow-up questions are made standalone
  5. Multilingual, citation-returning generation

Every design choice here has a stated tradeoff — see README.md's
"Known Limitations" section. Don't present this as bulletproof; it isn't.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger(__name__)


class Citation(TypedDict):
    source: str
    chunk_id: int
    text: str


class AnswerResult(TypedDict):
    answer: str
    citations: List[Citation]
    standalone_query: str  # the rewritten query actually used for retrieval


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_indexes(documents: List[Document]) -> Dict[str, Any]:
    """Build both a FAISS vector index and a BM25 keyword index from the same chunks.

    Embeddings run LOCALLY (no API call) — see config.py for why. First call
    will download the model (~1GB) and cache it; subsequent runs are fast.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.model_cache_dir,
    )
    faiss_index = FAISS.from_documents(documents, embeddings)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = settings.sparse_top_k

    dense_retriever = faiss_index.as_retriever(search_kwargs={"k": settings.dense_top_k})

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=list(settings.ensemble_weights),
    )

    return {"faiss_index": faiss_index, "bm25_retriever": bm25_retriever, "ensemble_retriever": ensemble}


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder.

    Loaded once and reused — instantiating this per-query would be slow
    (it's a real neural forward pass, not a lookup).
    """

    def __init__(self, model_name: str = settings.reranker_model):
        self._model = CrossEncoder(model_name, cache_dir=settings.model_cache_dir)

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        scored = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]


# ---------------------------------------------------------------------------
# History-aware query rewriting
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question, rewrite the question "
     "as a standalone question that can be understood without the chat history. "
     "Do NOT answer it. If the question is already standalone, return it unchanged. "
     "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def rewrite_query(llm: ChatGroq, question: str, chat_history: List[BaseMessage]) -> str:
    if not chat_history:
        return question
    chain = _REWRITE_PROMPT | llm
    result = chain.invoke({"question": question, "chat_history": chat_history})
    return result.content.strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are PolicyQA, an assistant that answers questions strictly using the "
     "provided policy document excerpts.\n\n"
     "CRITICAL INSTRUCTIONS:\n"
     "1. Detect the language of the user's question and respond in that exact language, "
     "regardless of the language the source excerpts are written in.\n"
     "2. Base your answer ONLY on the excerpts below. If the excerpts do not contain "
     "the answer, say so explicitly — do not guess or use outside knowledge.\n"
     "3. Be concise and direct. Do not pad the answer with filler.\n\n"
     "Excerpts:\n{context}"),
    ("human", "{question}"),
])


def _format_context(documents: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        src = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        blocks.append(f"[{i}] (source: {src}, chunk: {chunk_id})\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer_question(
    llm: ChatGroq,
    reranker: Reranker,
    ensemble_retriever: EnsembleRetriever,
    question: str,
    chat_history: List[BaseMessage],
) -> AnswerResult:
    """Full pipeline: rewrite -> hybrid retrieve -> rerank -> generate with citations."""
    standalone_query = rewrite_query(llm, question, chat_history)

    candidates = ensemble_retriever.invoke(standalone_query)[: settings.rerank_candidate_k]
    top_docs = reranker.rerank(standalone_query, candidates, settings.final_top_k)

    context = _format_context(top_docs)
    chain = _ANSWER_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    citations: List[Citation] = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "text": doc.page_content,
        }
        for doc in top_docs
    ]

    return {"answer": response.content, "citations": citations, "standalone_query": standalone_query}


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_document(llm: ChatGroq, full_text: str) -> str:
    """Map-reduce summarization: summarize chunks independently, then combine."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.summary_chunk_size,
        chunk_overlap=settings.summary_chunk_overlap,
    )
    docs = [Document(page_content=t) for t in splitter.split_text(full_text)]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]
