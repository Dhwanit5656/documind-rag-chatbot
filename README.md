# 📘 DocuMind — Document Q&A with RAG

> Ask intelligent questions about your documents • Powered by LLaMA 3.1 + ChromaDB

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://6kxhlb5wbfae7syfqqd6f5.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-2C3E50)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)

---

## 🧠 What is DocuMind?

DocuMind is a **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF, TXT, or DOCX documents and ask natural language questions about their content. It retrieves the most relevant passages from your documents and uses **LLaMA 3.1-8B-Instruct** to generate accurate, cited answers — without hallucinating beyond the provided context.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 Language Detection (langdetect)
      │
      ▼
 Embedding Query (paraphrase-multilingual-MiniLM-L12-v2)
      │
      ▼
 ChromaDB — MMR Retrieval (top-k chunks)
      │
      ▼
 Prompt Construction (numbered context blocks + citation rules)
      │
      ▼
 LLaMA 3.1-8B-Instruct (via HuggingFace Inference API)
      │
      ▼
 Answer with inline [Source: filename, Page: N] citations
```

---

## ✨ Features

### Core
- **Document Ingestion** — Upload PDF, TXT, or DOCX files via API or UI
- **Semantic Chunking** — 500-char chunks with 50-char overlap using `RecursiveCharacterTextSplitter`
- **Vector Search** — ChromaDB with MMR (Maximal Marginal Relevance) for diversity-aware retrieval
- **Multilingual Support** — Auto-detects Hindi vs English; responds in the question's language
- **Cited Answers** — Every claim tagged with `[Source: filename, Page: N]`
- **Hallucination Guard** — System prompt explicitly forbids answers outside the provided context

### Bonus
- ✅ Metadata filtering (by source filename and/or language)
- ✅ Streamlit UI with chat history
- ✅ FastAPI REST backend
- ✅ Lazy LLM loading (doesn't block server startup)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd documind
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

### 4. Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## 🔌 API Reference

### `POST /upload`
Upload one or more documents for indexing.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document.pdf"
```

**Response:**
```json
{
  "message": "Documents indexed successfully.",
  "files_processed": ["document.pdf"],
  "total_chunks_indexed": 42
}
```

### `POST /query`
Ask a question against indexed documents.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 6}'
```

**Response:**
```json
{
  "question": "What is the main topic?",
  "answer": "The main topic is ... [Source: document.pdf, Page: 3]",
  "language_detected": "en",
  "sources": [{"source": "document.pdf", "page": "3", "language": "en"}],
  "chunks_used": 6
}
```

**Optional filters:**
```json
{
  "question": "What are the key findings?",
  "filter_source": "report.pdf",
  "filter_language": "en",
  "top_k": 4
}
```

### `GET /documents`
List all indexed documents.

### `DELETE /reset`
Clear the vector store and all indexed documents.

---

## 🗂️ Project Structure

```
├── app.py              # Streamlit UI
├── main.py             # FastAPI REST API
├── pipeline.py         # Core RAG pipeline (ingestion, retrieval, generation)
├── requirements.txt
└── .env                # API keys (not committed)
```

---

## 🔧 Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | ChromaDB (in-memory) | Zero-setup, great for prototyping; persistent mode easy to add |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, supports Hindi & English, CPU-friendly |
| Retrieval | MMR (`lambda_mult=0.5`) | Balances relevance and diversity to reduce redundant chunks |
| Chunk size | 500 chars / 50 overlap | Balances context density with embedding precision |
| LLM | LLaMA 3.1-8B-Instruct | Open-source, instruction-tuned, strong reasoning |
| LLM loading | Lazy (first `/query` call) | Avoids blocking FastAPI startup |

---

## 🎯 Interview Q&A

**Why 500-char chunks?**
Balances context density and embedding precision. Too large → noisy embeddings; too small → missing context at boundaries. The 50-char overlap (10%) preserves context across splits.

**What happens with large documents?**
The splitter handles arbitrarily large documents by recursively splitting on `\n\n`, `\n`, `.`, and spaces. Only chunks with >20 characters are indexed to filter noise.

**How do you reduce hallucination?**
Three layers: (1) system prompt explicitly forbids using prior knowledge, (2) numbered context blocks let the LLM reference exactly what it used, (3) the fallback response "I could not find the answer" is triggered when retrieval returns nothing.

**FAISS vs ChromaDB?**
FAISS is a pure vector similarity library — fast, no metadata filtering. ChromaDB is a full vector *database* with metadata storage, filtering, and a collection API. ChromaDB was chosen here for its metadata filtering support (filter by source/language).

**Embedding dimension?**
384 dimensions (paraphrase-multilingual-MiniLM-L12-v2). Embeddings are L2-normalized (`normalize_embeddings=True`), so cosine similarity reduces to dot product.

**How would you scale this?**
Switch ChromaDB to persistent/server mode or replace with Pinecone/Qdrant; move embeddings to GPU; add a Redis cache for repeated queries; containerise with Docker; use async FastAPI endpoints for concurrent requests.

---

## 📦 Dependencies

```
langchain, langchain-huggingface, langchain-community
langchain-chroma, langchain-text-splitters, langchain-core
chromadb
huggingface_hub, sentence-transformers
pypdf, docx2txt
langdetect
fastapi, uvicorn[standard], python-multipart
python-dotenv, streamlit
```

---

## 🌐 Live Demo

Try it now: **[https://6kxhlb5wbfae7syfqqd6f5.streamlit.app/](https://6kxhlb5wbfae7syfqqd6f5.streamlit.app/)**

---

## 📄 License

 Apache-2.0 license
