"""
main.py — FastAPI application for the Document Q&A RAG API.

Endpoints:
    POST /upload   — Ingest one or more documents into the vector store.
    POST /query    — Ask a question against the indexed documents.
    GET  /documents — List all indexed document filenames.
    DELETE /reset  — Clear the vector store.

Run locally:
    uvicorn main:app --reload
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipeline import (
    load_document_chunk,
    build_vector_store,
    get_vector_store,
    reset_vector_store,
    search_documents,
    get_answer,
    detect_language,
    get_llm,
)


# ── Lifespan: warm up the embedding model on startup ──────────────────────────
# The LLM itself is intentionally NOT loaded here — it is loaded lazily on the
# first /query request. The embedding model is lightweight and safe to warm up.

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting — embedding model is ready.")
    # Trigger embedding model init (it lives in pipeline module scope)
    yield
    print("Server shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document Q&A RAG API",
    description=(
        "Upload PDF / TXT / DOCX documents, then ask questions. "
        "Answers are grounded in the uploaded content using RAG "
        "(Retrieval-Augmented Generation) with LLaMA 3.1 and ChromaDB."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# In-memory registry: tracks filenames of indexed documents this session
_indexed_files: list[str] = []


# ── Request / Response Schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    filter_source: Optional[str] = None       # restrict to a specific filename
    filter_language: Optional[str] = None     # 'en' or 'hi'
    top_k: int = 6                            # number of chunks to retrieve


class SourceCitation(BaseModel):
    source: str
    page: str
    language: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    language_detected: str
    sources: list[SourceCitation]
    chunks_used: int


class UploadResponse(BaseModel):
    message: str
    files_processed: list[str]
    total_chunks_indexed: int


class DocumentsResponse(BaseModel):
    indexed_files: list[str]
    total_files: int


# ── Helpers ────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return ext


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Document Q&A RAG API is running."}


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDF, TXT, or DOCX files.

    - Extracts text and splits into overlapping chunks (500 chars, 50 overlap).
    - Embeds chunks using a multilingual sentence-transformer model.
    - Stores embeddings in an in-memory ChromaDB collection.
    - Previously indexed documents are preserved (additive upload).

    Returns the number of chunks indexed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    temp_paths: list[str] = []
    name_map: dict[str, str] = {}   # temp_basename → original filename

    try:
        # Save uploads to temp files
        for upload in files:
            ext = _validate_extension(upload.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                content = await upload.read()
                tmp.write(content)
                temp_paths.append(tmp.name)
                name_map[os.path.basename(tmp.name)] = upload.filename

        # Ingest and chunk
        chunks = load_document_chunk(temp_paths)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the uploaded files."
            )

        # Fix temp-path source names → original filenames
        for chunk in chunks:
            temp_source = chunk.metadata.get("source", "")
            if temp_source in name_map:
                chunk.metadata["source"] = name_map[temp_source]

        # Merge with any existing vector store
        existing_db = get_vector_store()
        if existing_db is not None:
            # Add new chunks to the existing collection
            existing_db.add_documents(chunks)
        else:
            build_vector_store(chunks)

        # Track filenames
        for filename in name_map.values():
            if filename not in _indexed_files:
                _indexed_files.append(filename)

        return UploadResponse(
            message="Documents indexed successfully.",
            files_processed=list(name_map.values()),
            total_chunks_indexed=len(chunks),
        )

    finally:
        # Always clean up temp files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
def query_documents(request: QueryRequest):
    """
    Ask a question against the indexed documents.

    - Retrieves the top-k most relevant chunks using MMR (Maximal Marginal
      Relevance) for diversity-aware retrieval.
    - Supports metadata filtering by filename (`filter_source`) or language
      (`filter_language`: 'en' or 'hi').
    - Auto-detects the question language and instructs the LLM to respond
      in the same language.
    - Returns the answer with inline citations (source + page).

    The LLM (LLaMA 3.1-8B-Instruct) is initialised on the first call to this
    endpoint (lazy loading) to avoid blocking server startup.
    """
    if get_vector_store() is None:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. Use POST /upload first."
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Validate optional filters
    if request.filter_language and request.filter_language not in ("en", "hi"):
        raise HTTPException(
            status_code=400,
            detail="filter_language must be 'en' or 'hi'."
        )

    # Retrieve relevant chunks
    try:
        context_texts, metadatas = search_documents(
            query=question,
            k=request.top_k,
            fetch_k=max(request.top_k * 3, 15),
            filter_source=request.filter_source,
            filter_language=request.filter_language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not context_texts:
        return QueryResponse(
            question=question,
            answer="I could not find the answer in the provided documents.",
            language_detected="en",
            sources=[],
            chunks_used=0,
        )

    # Detect language and generate answer
    language = detect_language(question)
    answer = get_answer(
        question=question,
        context_texts=context_texts,
        metadatas=metadatas,
        language=language,
    )

    # Build deduplicated source list for the response
    seen_sources: set[str] = set()
    sources: list[SourceCitation] = []
    for meta in metadatas:
        source = meta.get("source", "unknown")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(SourceCitation(
                source=source,
                page=str(meta.get("page", "N/A")),
                language=meta.get("language", "en"),
            ))

    return QueryResponse(
        question=question,
        answer=answer,
        language_detected=language,
        sources=sources,
        chunks_used=len(context_texts),
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
def list_documents():
    """
    List all documents currently indexed in the vector store.
    """
    return DocumentsResponse(
        indexed_files=_indexed_files,
        total_files=len(_indexed_files),
    )


@app.delete("/reset", tags=["Documents"])
def reset_documents():
    """
    Clear the entire vector store and document registry.
    Use this before uploading a completely new set of documents.
    """
    reset_vector_store()
    _indexed_files.clear()
    return {"message": "Vector store cleared. All indexed documents removed."}
