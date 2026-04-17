import os
import shutil
from functools import lru_cache
from typing import Optional

from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

DetectorFactory.seed = 0

# ── Language Detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect whether the text is Hindi ('hi') or English ('en')."""
    if len(text.strip()) < 20:
        return "en"
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except Exception:
        return "en"


# ── Document Ingestion & Chunking ──────────────────────────────────────────────

def load_document_chunk(filepaths: list[str]) -> list:
    """
    Load documents from the given file paths, split into chunks, and enrich
    each chunk's metadata with source filename and detected language.
    """
    all_documents = []

    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".txt":
            loader = TextLoader(filepath, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(filepath)
        else:
            print(f"Unsupported file type: {ext}, skipping.")
            continue

        try:
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            continue

    if not all_documents:
        return []

    # chunk_size=500 balances context density with embedding precision.
    # chunk_overlap=50 (10%) preserves context at split boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)

    # Filter out near-empty chunks that would add noise to the vector store
    chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 20]

    # Enrich metadata: clean up source path, add language tag
    for chunk in chunks:
        full_path = chunk.metadata.get("source", "unknown")
        chunk.metadata["source"] = os.path.basename(full_path)
        chunk.metadata["language"] = detect_language(chunk.page_content)

    print(f"Loaded {len(all_documents)} docs → {len(chunks)} valid chunks")
    return chunks


# ── Embedding Model (singleton — safe to init at module level) ─────────────────

# Embeddings are stateless and cheap to hold in memory permanently.
# paraphrase-multilingual-MiniLM-L12-v2 → 384-dim, supports Hindi & English.
_embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # cosine similarity via dot product
)


# ── LLM — Lazy Initialisation ──────────────────────────────────────────────────
# The LLM is NOT created at import time. It is created once on first use via
# get_llm(). This prevents blocking FastAPI startup and avoids unnecessary
# HuggingFace API calls when the server is just warming up.

_llm_instance = None

def get_llm() -> ChatHuggingFace:
    """Return a cached LLM instance, initialising it on first call."""
    global _llm_instance
    if _llm_instance is None:
        print("Initialising LLM (first call)...")
        endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            max_new_tokens=1024,
        )
        _llm_instance = ChatHuggingFace(llm=endpoint)
        print("LLM ready.")
    return _llm_instance


# ── Vector Store ───────────────────────────────────────────────────────────────

_db: Optional[Chroma] = None

def build_vector_store(chunks: list) -> None:
    """Embed chunks and load them into an in-memory ChromaDB collection."""
    global _db
    _db = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding,
        collection_name="rag-doc-chatbot",
    )
    print(f"Vector store built: {len(chunks)} chunks indexed.")

def get_vector_store() -> Optional[Chroma]:
    """Return the current vector store, or None if no documents have been loaded."""
    return _db

def reset_vector_store() -> None:
    """Clear the in-memory vector store (e.g. when user uploads new documents)."""
    global _db
    _db = None


# ── Retrieval with Metadata Filtering ─────────────────────────────────────────

def search_documents(
    query: str,
    k: int = 6,
    fetch_k: int = 15,
    filter_source: Optional[str] = None,
    filter_language: Optional[str] = None,
) -> tuple[list[str], list[dict]]:
    """
    Retrieve top-k chunks relevant to the query using MMR.

    Args:
        query:           User question.
        k:               Number of chunks to return after MMR reranking.
        fetch_k:         Candidate pool size before MMR diversity filtering.
        filter_source:   Optional filename to restrict retrieval to one document.
        filter_language: Optional language tag ('en' or 'hi') for metadata filtering.

    Returns:
        (context_texts, metadatas) — parallel lists.
    """
    if _db is None:
        raise ValueError("No documents have been indexed yet. Call build_vector_store() first.")

    # Build ChromaDB metadata filter if any filters are specified
    where: Optional[dict] = None
    if filter_source and filter_language:
        where = {"$and": [{"source": {"$eq": filter_source}}, {"language": {"$eq": filter_language}}]}
    elif filter_source:
        where = {"source": {"$eq": filter_source}}
    elif filter_language:
        where = {"language": {"$eq": filter_language}}

    retriever = _db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.5,   # 0 = max diversity, 1 = max relevance
            **({"filter": where} if where else {}),
        },
    )

    docs = retriever.invoke(query)
    context_texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    return context_texts, metadatas


# ── Prompt Engineering ────────────────────────────────────────────────────────
# Improvements over v1:
#   1. Explicit citation instruction — LLM must tag [Source: filename, Page: N].
#   2. Chain-of-thought nudge — "Think step by step" before answering.
#   3. Numbered context blocks — LLM can reference [Context 1], [Context 2], etc.
#   4. Strict hallucination guard — explicit "DO NOT use prior knowledge".
#   5. Language instruction injected dynamically per request.

SYSTEM_PROMPT = """\
You are a precise, helpful document assistant. You answer questions ONLY using \
the numbered context passages provided below. 

RULES (follow strictly):
1. Base your answer exclusively on the context passages. DO NOT use prior knowledge \
or make assumptions beyond what is written.
2. After each key claim, cite the source inline as [Source: <filename>, Page: <page>].
3. If multiple passages support a point, cite all relevant ones.
4. If the answer cannot be found in the context, respond with exactly: \
"I could not find the answer in the provided documents."
5. Think step by step before writing your final answer.
6. {lang_instruction}

CONTEXT PASSAGES:
{context}
"""

def _format_context(context_texts: list[str], metadatas: list[dict]) -> str:
    """Format retrieved chunks as numbered, labelled context blocks for the LLM."""
    blocks = []
    for i, (text, meta) in enumerate(zip(context_texts, metadatas), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        blocks.append(f"[Context {i}] (Source: {source}, Page: {page})\n{text.strip()}")
    return "\n\n".join(blocks)


def get_answer(
    question: str,
    context_texts: list[str],
    metadatas: list[dict],
    language: str,
) -> str:
    """
    Run the RAG chain: format context → fill prompt → call LLM → parse output.

    Args:
        question:      User's question.
        context_texts: Retrieved chunk texts.
        metadatas:     Corresponding chunk metadata (source, page, language).
        language:      Detected language of the question ('en' or 'hi').

    Returns:
        LLM-generated answer string.
    """
    lang_instruction = (
        "कृपया हिंदी में उत्तर दें।" if language == "hi"
        else "Please respond in English."
    )

    formatted_context = _format_context(context_texts, metadatas)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        RunnableParallel({
            "context": RunnablePassthrough() | (lambda _: formatted_context),
            "lang_instruction": RunnablePassthrough() | (lambda _: lang_instruction),
            "question": RunnablePassthrough(),
        })
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain.invoke(question)
