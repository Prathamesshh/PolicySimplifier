"""
app.py
Streamlit frontend for PolicyQA. Contains UI logic only — all AI/retrieval
logic lives in src/. This file should stay thin; if you find yourself
writing retrieval or prompting logic here, it belongs in src/rag_pipeline.py.
"""

import logging

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from src.config import settings, validate_settings
from src.document_processor import normalize_input, chunk_text, DocumentProcessingError
from src.rag_pipeline import build_indexes, Reranker, answer_question, summarize_document

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="PolicyQA", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(model=settings.llm_model, temperature=settings.llm_temperature, api_key=settings.groq_api_key)


@st.cache_resource(show_spinner=False)
def get_reranker() -> Reranker:
    return Reranker()


def init_session_state():
    defaults = {
        "indexes": None,
        "raw_text": None,
        "messages": [],       # list[HumanMessage | AIMessage] for the LLM
        "display_messages": [],  # list[dict] with role/content/citations for the UI
        "summary": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def process_uploaded_document():
    st.subheader("1. Load a document")
    input_type = st.radio("Input type", ["PDF", "Image", "Text"], horizontal=True)

    file_bytes, raw_text, source_name = None, None, "uploaded_document"

    if input_type == "PDF":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded:
            file_bytes, source_name = uploaded.read(), uploaded.name
    elif input_type == "Image":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded:
            file_bytes, source_name = uploaded.read(), uploaded.name
    else:
        raw_text = st.text_area("Paste text", height=200)
        source_name = "pasted_text"

    if st.button("Process document", type="primary", disabled=not (file_bytes or raw_text)):
        with st.spinner("Extracting and indexing..."):
            try:
                text = normalize_input(
                    file_bytes, raw_text, input_type.lower() if input_type != "Image" else "image"
                )
                docs = chunk_text(text, source_name=source_name)
                st.session_state.indexes = build_indexes(docs)
                st.session_state.raw_text = text
                st.session_state.messages = []
                st.session_state.display_messages = []
                st.session_state.summary = None
                st.success(f"Indexed {len(docs)} chunks from '{source_name}'.")
            except DocumentProcessingError as exc:
                st.error(str(exc))


def chat_tab():
    st.subheader("2. Ask questions")
    if st.session_state.indexes is None:
        st.info("Load and process a document first.")
        return

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources used"):
                    for i, c in enumerate(msg["citations"], start=1):
                        st.markdown(f"**[{i}] {c['source']} — chunk {c['chunk_id']}**")
                        st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

    question = st.chat_input("Ask a question about the document (any language)")
    if question:
        st.session_state.display_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                llm = get_llm()
                reranker = get_reranker()
                result = answer_question(
                    llm=llm,
                    reranker=reranker,
                    ensemble_retriever=st.session_state.indexes["ensemble_retriever"],
                    question=question,
                    chat_history=st.session_state.messages,
                )
                st.markdown(result["answer"])
                if result["citations"]:
                    with st.expander("Sources used"):
                        for i, c in enumerate(result["citations"], start=1):
                            st.markdown(f"**[{i}] {c['source']} — chunk {c['chunk_id']}**")
                            st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

        st.session_state.messages.append(HumanMessage(content=question))
        st.session_state.messages.append(AIMessage(content=result["answer"]))
        st.session_state.display_messages.append(
            {"role": "assistant", "content": result["answer"], "citations": result["citations"]}
        )


def summary_tab():
    st.subheader("3. Document summary")
    if st.session_state.raw_text is None:
        st.info("Load and process a document first.")
        return
    if st.button("Generate summary"):
        with st.spinner("Summarizing (map-reduce)..."):
            st.session_state.summary = summarize_document(get_llm(), st.session_state.raw_text)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)


def main():
    st.title("📄 PolicyQA")
    st.caption("Multilingual, multimodal document Q&A with hybrid retrieval and cited answers.")

    try:
        validate_settings()
    except EnvironmentError as exc:
        st.error(str(exc))
        st.stop()

    init_session_state()
    process_uploaded_document()
    st.divider()
    tab1, tab2 = st.tabs(["💬 Chat", "📝 Summary"])
    with tab1:
        chat_tab()
    with tab2:
        summary_tab()


if __name__ == "__main__":
    main()
