# PolicyQA

Multilingual, multimodal document Q&A. Upload a policy document (PDF, image,
or pasted text), ask questions about it in any language, and get answers
with cited source chunks. Also generates a map-reduce summary.

## Architecture

```
PolicyQA/
├── src/
│   ├── config.py              # all tunable parameters, single source of truth
│   ├── document_processor.py  # PDF / OCR / text -> normalized chunks
│   └── rag_pipeline.py        # indexing, hybrid retrieval, reranking, generation
├── app.py                     # Streamlit UI only — no business logic
├── requirements.txt
├── .env.example
└── README.md
```

### Retrieval pipeline

1. **Ingestion** — PDF text via `pypdf`, image text via `pytesseract` OCR,
   or raw pasted text. All three are normalized to plain text before
   anything else happens.
2. **Chunking** — `RecursiveCharacterTextSplitter`, 1000 chars / 150 overlap
   by default (tune in `.env` or `config.py`).
3. **Dual indexing** — every chunk goes into both a FAISS vector index
   (semantic/dense) and a BM25 index (keyword/sparse) at the same time.
4. **Query rewriting** — if there's chat history, an LLM call rewrites the
   follow-up question into a standalone query first (e.g. "does it apply to
   contractors?" → "does the maternity leave policy apply to contractors?").
5. **Hybrid retrieval** — `EnsembleRetriever` runs FAISS and BM25 in
   parallel and merges results (weighted 50/50 by default). This exists
   because vector search alone misses exact-keyword queries like form
   numbers or defined terms.
6. **Reranking** — the merged candidate pool (up to 20 chunks) is scored by
   a cross-encoder (`BAAI/bge-reranker-base`) against the query, and only
   the top 5 are kept. This fixes the "lost in the middle" problem where a
   relevant chunk buried in position 8 gets ignored by the LLM.
7. **Generation** — the top chunks are injected into a prompt that instructs
   the model to answer only from the given context, in the same language as
   the question, and the UI displays the exact source chunks used.

## Cost: what's free and why

| Component | Where it runs | Cost |
|---|---|---|
| LLM (query rewrite + answer generation) | Groq API | Free tier — rate-limited, not unlimited |
| Embeddings | Locally, on your machine | Free — no API call at all, one-time model download |
| Reranker (cross-encoder) | Locally, on your machine | Free — no API call at all |

This is a **deliberate split**, not "everything free-tier":

- Embeddings happen in bulk — every chunk of a document needs one, so
  uploading a document can mean 20-50+ calls at once. A free API's rate
  limit would choke on that burst. Running the embedding model locally
  avoids that risk entirely, at the cost of a one-time ~1GB download and
  using your own CPU/RAM instead of someone else's server.
- The LLM is called only twice per question (rewrite, then answer) — low
  enough volume that Groq's free tier realistically handles it.

**Get a free Groq key:** https://console.groq.com/keys — no credit card
required at time of writing, but free tiers change; check their current
limits before you rely on this for a live demo.

**Real risk to know about:** free tiers get rate-limited or occasionally
throttled/down. If you're demoing this live to a recruiter, test it
beforehand — don't find out it's rate-limited in the room.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GROQ_API_KEY
```

First run will download the local embedding model (~1GB) and the reranker
model — this happens once and is cached afterward. Expect the first
document upload to be slower than subsequent ones.

**OCR requires a system binary**, not just a Python package:

- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: install from [UB-Mannheim's build](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env`

Run it:

```bash
streamlit run app.py
```

## Known limitations — read this before calling it "enterprise-grade"

Be accurate about what this is in an interview. It's a well-structured RAG
prototype with some genuinely non-trivial retrieval engineering (hybrid
search + reranking + history-aware queries). It is **not** production
infrastructure. Specifically:

- **No persistence** — the FAISS/BM25 index lives in memory (`st.session_state`)
  and is lost on restart. A real deployment needs the index saved to disk
  or a managed vector DB (Pinecone, Weaviate, pgvector).
- **No auth, no multi-user isolation** — anyone with the URL can use your
  API key's quota. Fine for a demo, not fine for a shared deployment.
- **No automated tests** — nothing here is verified beyond manual runs.
  A recruiter who asks "how do you know the reranker is actually helping"
  currently has no answer from this repo alone.
- **No logging/observability beyond stdlib `logging`** — no tracing of
  latency per pipeline stage, no eval harness for answer quality.
- **OCR quality is untested against real scanned policy documents** —
  Tesseract is decent on clean scans, poor on skewed/low-res ones.
- **Free-tier rate limits, not infinite scale** — every question makes 2
  Groq API calls (query rewrite + answer). Fine for a demo and casual use;
  a free tier can throttle you under sustained or concurrent load. This is
  not a "scales to many users" architecture — it's a "works reliably for
  one person demoing it" architecture. Say that distinction out loud if
  asked, don't let "free" get conflated with "production-scale."
- **Local compute cost, not zero cost** — embeddings and reranking run on
  your own CPU. On a low-spec machine, indexing a long document can be
  noticeably slow. This tradeoff was made deliberately (see "Cost: what's
  free and why" above) but it's still a real cost, just not a dollar one.

If you want to genuinely close some of these gaps rather than just talk
around them: add `pytest` tests for the retrieval functions, add a
retrieval-quality eval set (query + expected chunk), persist the FAISS
index with `save_local`/`load_local`, and add basic request logging with
timing per stage. That's a more credible "I hardened this for production
concerns" story than the feature list alone.
