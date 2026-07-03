"""
document_processor.py
Handles all input ingestion: PDF, image (OCR), and raw text.
Everything is normalized to plain text + LangChain Document objects
before it ever touches the RAG pipeline, so downstream code does not
need to know or care what the original input format was.
"""

from __future__ import annotations

import io
import logging
from typing import List, Literal

import pytesseract
from PIL import Image
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

InputType = Literal["pdf", "image", "text"]


class DocumentProcessingError(Exception):
    """Raised when a file cannot be parsed into usable text."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file's bytes."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open PDF: {exc}") from exc

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
        else:
            logger.warning("Page %d produced no extractable text (likely a scanned image).", i)

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise DocumentProcessingError(
            "No extractable text found in PDF. If this is a scanned document, "
            "convert its pages to images and use the image (OCR) input instead."
        )
    return full_text


def extract_text_from_image(file_bytes: bytes) -> str:
    """Run OCR (Tesseract) on an image and return extracted text."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise DocumentProcessingError(
            "Tesseract OCR engine not found on this machine. Install it separately "
            "(it is a system binary, not a pip package) — see README.md."
        ) from exc
    except Exception as exc:
        raise DocumentProcessingError(f"Could not process image: {exc}") from exc

    if not text.strip():
        raise DocumentProcessingError(
            "OCR produced no text. Check the image is legible and right-side up."
        )
    return text


def normalize_input(file_bytes: bytes | None, raw_text: str | None, input_type: InputType) -> str:
    """Single entry point: takes whatever the user gave us and returns plain text."""
    if input_type == "pdf":
        if not file_bytes:
            raise DocumentProcessingError("PDF input selected but no file bytes provided.")
        return extract_text_from_pdf(file_bytes)
    if input_type == "image":
        if not file_bytes:
            raise DocumentProcessingError("Image input selected but no file bytes provided.")
        return extract_text_from_image(file_bytes)
    if input_type == "text":
        if not raw_text or not raw_text.strip():
            raise DocumentProcessingError("Text input selected but no text provided.")
        return raw_text
    raise DocumentProcessingError(f"Unknown input_type: {input_type}")


def chunk_text(text: str, source_name: str = "uploaded_document") -> List[Document]:
    """Split normalized text into overlapping chunks with source metadata for citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_id": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    if not documents:
        raise DocumentProcessingError("Text splitting produced zero chunks — input may be empty.")
    return documents
