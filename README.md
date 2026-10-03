# TeleRAG: RAG-Based Telegram PDF Question Answering Bot

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![RAG Architecture](https://img.shields.io/badge/RAG-Local%20Embeddings%20%2B%20OpenRouter-orange)]()

**TeleRAG** is a lightweight, self-contained **Retrieval-Augmented Generation (RAG)** chatbot for Telegram. It enables users to upload local PDF files or supply public PDF URLs directly in chat, automatically extracts and chunks textual content, generates dense semantic embeddings locally using Sentence Transformers, and retrieves strictly relevant passages to answer natural language questions through an instruction-tuned Large Language Model (LLM) via OpenRouter.

Designed as an educational, zero-external-vector-database architecture, TeleRAG achieves complete document grounding, multi-user isolation, and robust server-side request protection without heavy cloud infrastructure.

---

## 📚 Project Documentation

The documentation is organized into three dedicated documents:

1. **[Feature & Architecture Overview (README.md)](README.md)** *(this document)*: High-level overview, feature catalog, system architecture, tech stack, and repository structure.
2. **[Setup & Operational Guide (INSTRUCTIONS.md)](INSTRUCTIONS.md)**: Detailed step-by-step installation, virtual environment setup, Telegram BotFather & OpenRouter key configuration, `.env` guide, Telegram command reference, test execution, and troubleshooting.
3. **[Academic Engineering Report (PROJECT_REPORT.md)](PROJECT_REPORT.md)**: Formal university-level project report covering the academic problem formulation, mathematical foundation, security threat matrix, *Think Python* real-world benchmarks, and design trade-offs.

---

## 🌟 Key Features

### 1. Document Ingestion & Validation
* **Direct File Upload**: Upload local PDF documents up to 20 MB directly through Telegram.
* **URL-Based Ingestion**: Supply remote PDF web links. Downloads stream with `User-Agent` and MIME type verification.
* **SSRF Defense**: Strict URL validation blocks access to private RFC1918 subnets (`10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`), loopback (`127.0.0.1`), and cloud metadata IP addresses (`169.254.169.254`).
* **Resource Bounds**: Hard size caps and streaming limits prevent memory exhaustion and Denial-of-Service.

### 2. Text Processing & Chunking
* **Robust Text Extraction**: Extracts and cleans raw text using `PyPDF2` with automatic whitespace normalization.
* **Sliding-Window Chunking**: Configurable token chunk size (`CHUNK_SIZE=1000`) and overlap (`CHUNK_OVERLAP=200`) to preserve sentence context across chunk boundaries.

### 3. Local Dense Embeddings & Vector Search
* **Local Bi-Encoder Model**: Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` locally on the host CPU. No external embedding API calls required.
* **Strict Cosine Similarity Search**: Scans chunk vectors using NumPy and scikit-learn cosine similarity.
* **Zero-Hallucination Thresholding**: Enforces a strict relevance cutoff (`SIMILARITY_THRESHOLD=0.3`). Queries without matching context trigger immediate, deterministic abstention rather than falling back to arbitrary top-$k$ chunks.

### 4. LLM Generation via OpenRouter
* **Free-Tier & Custom Inference**: Integrated with OpenRouter API, configured by default with high-performance free models (`qwen/qwen3.8-27b:free`) and customizable via `OPENROUTER_MODEL`.
* **Grounded Context Prompts**: Dynamically compiles top-$k$ passages into a concise, hallucination-resistant prompt.

### 5. Multi-User Isolation & Bot Management
* **Tenant Isolation**: All SQLite metadata and chunk records are indexed and scoped by Telegram `user_id`. Users can only query and manage their own documents.
* **Foreign Key Cascades**: SQLite database enforces `PRAGMA foreign_keys = ON`. Deleting a document instantly cascades and removes all associated chunk vectors and disk files.
* **Interactive Bot Commands**:
  * `/start` – Welcome message and system overview.
  * `/help` – Usage guide and syntax.
  * `/list` – View your currently active uploaded documents.
  * `/delete <filename>` – Delete a specific document and its vector indices.
  * `/clear` – Clear all uploaded documents for your session.

---

## 🏗️ System Architecture

```
User (Telegram Client)
         │
         │  (HTTPS Telegram Bot API)
         ▼
┌────────────────────────────────────────────────────────┐
│             BotHandlers (Telegram Interface)           │
│  - User session & tenant isolation                     │
│  - SSRF URL verification & streaming download          │
│  - Command routing: /start, /list, /delete, /clear     │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
      [Document Upload]             [User Question]
               │                          │
               ▼                          ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│       PDF Ingestion          │ │      Retrieval Pipeline      │
│  - PDFProcessor (PyPDF2)     │ │  - EmbeddingGenerator        │
│  - TextChunker               │ │    (all-MiniLM-L6-v2)        │
│    (1000 chars, 200 overlap) │ │  - DocumentRetriever         │
│  - EmbeddingGenerator        │ │    (Cosine similarity ≥ 0.3) │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│       DocumentStore          │ │      OpenRouter Client       │
│  - SQLite (Metadata & chunks)│ │  - Grounded prompt assembly  │
│  - JSON per-document vectors │ │  - qwen/qwen3.8-27b:free     │
└──────────────────────────────┘ └──────────────┬───────────────┘
                                                │
                                                ▼
                                    Telegram Answer Output
```

---

## 💻 Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10 – 3.12 (Tested on 3.12.10) | Compatible with `sentence-transformers==2.7.0` |
| **Bot Framework** | `python-telegram-bot` (v20+) | Async Telegram Bot API client |
| **PDF Extraction** | `PyPDF2` (v3.0.1) | Lightweight, pure-Python text extraction |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | High-speed, local CPU 384-d semantic representations |
| **Vector Search** | `scikit-learn` / `numpy` | Pure in-memory cosine similarity without external DB |
| **Relational Storage**| `sqlite3` | Zero-dependency local ACID storage with foreign key cascade |
| **LLM Provider** | OpenRouter API (`qwen/qwen3.8-27b:free`) | High-quality open-weights inference with zero upfront cost |

---

## 📁 Repository Structure

```
TeleRAG/
├── main.py                     # Bot entry point and lifecycle manager
├── requirements.txt            # Pinned dependency definitions
├── .env.example                # Configuration template
├── .gitignore                  # Git exclusion rules (.env, .venv, data/, logs/)
├── INSTRUCTIONS.md             # Complete installation, configuration & user manual
├── PROJECT_REPORT.md           # Formal academic engineering report
├── readme.md                   # Repository features & architecture overview
│
├── src/
│   ├── config.py               # Central environment variable configuration
│   ├── bot/
│   │   ├── handlers.py         # Telegram message & command handlers, SSRF checks
│   │   └── __init__.py
│   ├── processors/
│   │   ├── pdf_processor.py    # PyPDF2 extraction, validation, and TextChunker
│   │   └── __init__.py
│   ├── rag/
│   │   ├── embeddings.py       # SentenceTransformers embedding generation
│   │   ├── retriever.py        # Strict cosine similarity search & prompt assembly
│   │   ├── openrouter_client.py# OpenRouter HTTP API client & response parsing
│   │   └── __init__.py
│   └── storage/
│       ├── database.py         # SQLite schema, multi-user queries & cascade logic
│       └── __init__.py
│
├── tests/                      # Automated unit and integration test suite
│   ├── test_pdf_processor.py   # Tests for extraction, chunking, and edge cases
│   ├── test_rag.py             # Tests for embeddings, cosine search, and thresholds
│   └── test_storage.py         # Tests for database CRUD, isolation, and cascades
│
├── data/                       # Local document storage (auto-generated, git-ignored)
└── logs/                       # Application run logs (auto-generated, git-ignored)
```

---

## 🚀 Quick Start Summary

For comprehensive setup, refer to the full **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.

### 1. Prerequisites & Environment
```bash
python --version   # Must be Python 3.10, 3.11, or 3.12
python -m venv .venv
.\.venv\Scripts\activate   # Windows (or source .venv/bin/activate on Linux)
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
```ini
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_from_botfather
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3.8-27b:free
```

### 3. Verify & Run
```bash
# Execute test suite (all 21 tests should pass)
python -m unittest discover tests

# Launch the bot
python main.py
```

---

## 🔒 Security & Privacy Highlights

* **No Leaked Secrets**: `.env` and sensitive files are strictly excluded from version control.
* **Per-User Scoping**: User documents, chunks, and metadata are strictly isolated by Telegram `user_id`. Users cannot query or delete documents owned by other users.
* **Safe SSRF Filtering**: URL-based ingestion explicitly blocks private IPs, loopback, link-local, and cloud metadata addresses before opening network sockets.
* **Sanitized Logging**: Application logs trace operations, HTTP status codes, and chunk metrics without recording user authentication tokens or full private document text.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
