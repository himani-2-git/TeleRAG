# TeleRAG: Retrieval-Augmented Generation PDF Question-Answering System via Telegram

**Technical Engineering Report**  
*System Architecture, Retrieval Mechanics & Empirical Evaluation*  

---

## Abstract

Traditional Large Language Models (LLMs) suffer from knowledge cutoff dates, lack of domain-specific data access, and a tendency to generate plausible but factually incorrect assertions (hallucinations). This report details the design, architecture, and implementation of **TeleRAG**, an autonomous Retrieval-Augmented Generation (RAG) system delivered through a Telegram bot interface. TeleRAG accepts user-provided PDF files or remote PDF URLs, extracts textual data, chunks the content using sliding-window token preservation, generates dense 384-dimensional semantic embeddings via a local bi-encoder model (`all-MiniLM-L6-v2`), and indexes the vectors into a lightweight hybrid SQLite and JSON storage architecture. When a user queries their document repository, cosine similarity search extracts the top-$k$ most relevant context passages above a calibrated similarity threshold ($\tau = 0.3$), which are then synthesized into a grounded prompt for an instruction-tuned LLM via the OpenRouter API. The system operates without heavyweight managed vector databases, provides multi-tenant per-user data isolation, protects against Server-Side Request Forgery (SSRF), and enforces strict deterministic abstention on out-of-context queries.

---

## 1. Introduction & Problem Statement

### 1.1 Context
In research and technical environments, users frequently need to extract precise information from dense, multi-page PDF documents such as textbooks, research articles, and technical manuals. While consumer LLM interfaces allow document uploads, they often require recurring cloud subscriptions, transmit entire documents across proprietary networks without transparent chunking controls, or fail to provide per-user multi-document lifecycle management.

### 1.2 Problem Statement
Developing a self-contained, low-resource RAG application requires addressing four core technical challenges:
1. **Infrastructure Overhead**: Managed vector databases (e.g. Pinecone, Milvus, Qdrant) introduce networking latency, maintenance complexity, and hosting costs unsuitable for lightweight, single-node or local deployments.
2. **Hallucination & False Grounding**: Standard similarity retrieval algorithms that fallback to returning top-$k$ passages regardless of similarity score inadvertently pass arbitrary document fragments into the context window, causing models to hallucinate answers based on unrelated sample dialogues.
3. **Multi-User Security & Isolation**: In a public Telegram interface, multi-user document collisions, unbounded streaming downloads, and malicious internal network probes (SSRF) present severe security vulnerabilities.
4. **Compute Efficiency**: Embedding generation and vector mathematics must execute efficiently on consumer-grade CPU hardware without mandatory GPU acceleration.

---

## 2. System Architecture & Component Design

TeleRAG follows a decoupled, modular service-oriented architecture:

```
┌────────────────────────────────────────────────────────┐
│                   Telegram Client                      │
└───────────────────────────┬────────────────────────────┘
                            │ (HTTPS Webhook / Long Polling)
                            ▼
┌────────────────────────────────────────────────────────┐
│           Telegram Bot Controller (BotHandlers)        │
│  - User Authentication & Scoping                       │
│  - Command Routing (/start, /list, /delete, /clear)     │
│  - SSRF Verification & Bounded File Download           │
└──────────────┬──────────────────────────┬──────────────┘
               │ (PDF Ingestion)          │ (Query Pipeline)
               ▼                          ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│        PDF Processing        │ │      Retriever & LLM         │
│  ┌────────────────────────┐  │ │  ┌────────────────────────┐  │
│  │     PDFProcessor       │  │ │  │   EmbeddingGenerator   │  │
│  │ (Text & Whitespace)    │  │ │  │  (all-MiniLM-L6-v2)    │  │
│  └───────────┬────────────┘  │ │  └───────────┬────────────┘  │
│              ▼               │ │              ▼               │
│  ┌────────────────────────┐  │ │  ┌────────────────────────┐  │
│  │      TextChunker       │  │ │  │   DocumentRetriever    │  │
│  │ (Window: 1000, Ov: 200)│  │ │  │ (Cosine Sim >= 0.3)   │  │
│  └───────────┬────────────┘  │ │  └───────────┬────────────┘  │
│              ▼               │ │              ▼               │
│  ┌────────────────────────┐  │ │  ┌────────────────────────┐  │
│  │   EmbeddingGenerator   │  │ │  │   OpenRouterClient     │  │
│  │  (all-MiniLM-L6-v2)    │  │ │  │ (Qwen3.8 27B Instruct) │  │
│  └───────────┬────────────┘  │ │  └───────────┬────────────┘  │
└──────────────┼───────────────┘ └──────────────┼───────────────┘
               ▼                                ▼
┌────────────────────────────────────────────────────────┐
│                   Persistence Layer                    │
│  - SQLite Database: Metadata, User Mapping, Chunks     │
│  - JSON Vector Store: 384-d Embedding Arrays           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Pipeline Implementation

### 3.1 Document Ingestion & Text Normalization
* **File Upload**: Direct document transmission via Telegram's MTProto API or remote PDF URL ingestion.
* **Extraction Engine**: Implemented in `PDFProcessor` using `PyPDF2.PdfReader`.
* **Sanitization**: Raw extracted streams frequently contain carriage returns (`\r\n`), intra-token control codes (`[\x00-\x1f]`), and erratic tab spacing. TeleRAG applies horizontal whitespace compression (`[ \t]+` $\to$ `' '`) and newline collapsing (`\n{3,}` $\to$ `\n\n`) while strictly preserving paragraph boundaries. This guarantees that semantic paragraph breaks remain available for downstream chunking.

### 3.2 Sliding-Window Text Chunking
Implemented in `TextChunker`:
* **Chunk Size ($C$)**: 1,000 characters.
* **Overlap Size ($O$)**: 200 characters (20% overlap).
* **Rationale**: Fixed-character chunking with semantic overlap ensures that concepts spanning adjacent blocks are not fragmented across retrieval boundaries. Each chunk is indexed with an ordinal `chunk_index` and validated for non-zero content before persistence.

### 3.3 Semantic Embedding Generation
Implemented in `EmbeddingGenerator`:
* **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
* **Architecture**: 6-layer MiniLM bi-encoder with mean pooling.
* **Embedding Dimension**: 384 dimensions.
* **Inference Strategy**: Document chunks are vectorized in batches during ingestion. Query embeddings are generated on-the-fly during query processing. Runs locally on CPU via PyTorch with average latency of $\approx 15\text{ ms}$ per query.

### 3.4 Hybrid Persistence Architecture
Implemented in `DocumentStore` and `Database`:
* **Relational Store (SQLite)**:
  - `documents` table: Stores `id`, `user_id`, `filename`, `file_hash`, `file_size`, `chunk_count`, and ISO-8601 `upload_date`.
  - `document_chunks` table: Stores `id`, `document_id`, `chunk_index`, `content`, `embedding_path`, and `char_count`.
  - **Referential Integrity**: Enforces `PRAGMA foreign_keys = ON` with `FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE`.
* **Vector Store (Local JSON)**:
  - High-dimensional vector lists are serialized to discrete per-document JSON arrays at `data/embeddings/user_{user_id}_doc_{doc_id}.json`.
  - **Advantages**: Avoids BLOB overhead in SQLite while maintaining trivial file-based backups and instant array deserialization using NumPy.

### 3.5 Semantic Retrieval & Cosine Similarity Ranking
Implemented in `DocumentRetriever`:
* **Mathematical Formulation**: For query vector $\mathbf{q} \in \mathbb{R}^{384}$ and chunk embedding $\mathbf{d}_i \in \mathbb{R}^{384}$:
  $$\text{Sim}(\mathbf{q}, \mathbf{d}_i) = \frac{\mathbf{q} \cdot \mathbf{d}_i}{\|\mathbf{q}\|_2 \|\mathbf{d}_i\|_2}$$
* **Threshold Filtering ($\tau = 0.3$)**: Only chunks where $\text{Sim}(\mathbf{q}, \mathbf{d}_i) \ge 0.3$ are admitted.
* **Hallucination Prevention**: Earlier RAG implementations often included a fallback branch returning the top-$k$ chunks when no chunk met the threshold. In TeleRAG, this fallback was explicitly removed. When zero chunks satisfy $\tau \ge 0.3$, `search_similar()` returns an empty list, allowing the bot controller to abstain deterministically without invoking the LLM.

### 3.6 LLM Synthesis & Prompt Grounding
Implemented in `OpenRouterClient`:
* **Model**: `qwen/qwen3.8-27b:free` via OpenRouter (262k token context window).
* **System Prompt Guardrails**:
  ```text
  You are a helpful assistant that answers questions based on provided document context.
  Use only the information from the context to answer questions.
  If the context doesn't contain relevant information, say so clearly.
  Be concise and accurate.
  ```
* **Fault Tolerance**: HTTP requests feature exponential backoff retry logic (up to 3 attempts) handling HTTP 429 rate limits and socket timeouts.

---

## 4. Multi-Tenant Security & Defensive Engineering

| Threat Vector | Mitigation Strategy | Implementation |
| :--- | :--- | :--- |
| **Server-Side Request Forgery (SSRF)** | Restrict URL schemes to `http`/`https`. Perform pre-flight DNS resolution using `socket.getaddrinfo()` and block private IPv4/IPv6 subnets (`127.0.0.1`, `10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`), link-local IPs, and cloud metadata addresses (`169.254.169.254`). | `BotHandlers._is_safe_url()` |
| **Denial of Service via Unbounded Downloads** | Impose strict streaming byte counting during HTTP downloads. If transferred bytes exceed `MAX_FILE_SIZE_MB * 1024 * 1024`, the connection is immediately aborted. | `BotHandlers._handle_url_upload()` |
| **Multi-User Data Leakage** | All database retrieval and chunk search operations strictly filter on `WHERE documents.user_id = ?`. Users have zero visibility into other users' documents. | `Database.get_all_user_chunks()` |
| **Client Header Blocking (HTTP 406)** | Many web servers with Apache Mod_Security block generic `python-requests` user agents. TeleRAG transmits standards-compliant identification headers (`User-Agent: TeleRAG/1.0`, `Accept: application/pdf`). | `BotHandlers._handle_url_upload()` |
| **Database Orphan Leaks** | Deleting a document automatically cleans associated chunks and index entries via SQLite cascading foreign keys. | SQLite `PRAGMA foreign_keys = ON` |

---

## 5. Experimental Evaluation & Verification

### 5.1 Test Suite & Verification Matrix
The codebase includes 21 automated unit and regression tests executing in $\approx 8\text{ seconds}$:
* **PDF Extraction & Chunking**: Validates clean text normalization, control code removal, character boundaries, and empty input handling (`test_pdf_processor.py`).
* **Storage & Relational Cascades**: Tests document insertion, chunk metadata linkage, retrieval scoping, and cascade deletion (`test_storage.py`).
* **Retrieval & Strict Thresholding**: Validates cosine scoring, descending ranking, and verifies that queries with sub-threshold similarity return empty sets without fallback hallucinations (`test_rag.py`).
* **LLM Client**: Tests prompt serialization, backoff retries, and choice extraction (`test_rag.py`).

### 5.2 Real-World Case Study: *Think Python (2nd Edition)*
* **Ingested Document**: *Think Python, 2nd Edition* by Allen Downey (899.8 KB, 240 pages).
* **Ingestion Metrics**: Extracted 444,777 characters, producing 587 chunks and 587 embeddings in 19.0 seconds on CPU.
* **Retrieval Dynamics**:

| Test Query | Max Cosine Similarity | Chunks $\ge 0.3$ | Execution Path | Observed Response Quality |
| :--- | :---: | :---: | :---: | :--- |
| `"what is recursion?"` | **0.5933** | 111 | LLM Synthesis | Accurate, grounded definition citing base cases and recursive steps. |
| `"define class"` | **0.5388** | 23 | LLM Synthesis | Grounded explanation of class objects as instance factories. |
| `"what's your name"` | **0.1899** | 0 | Deterministic Abstention | Returns *"I couldn't find relevant information in your documents..."* without LLM call. |
| `"how to apply paint on wall"`| **0.1091** | 0 | Deterministic Abstention | Returns clean abstention message. Zero API tokens consumed. |

---

## 6. Architectural Trade-offs & Limitations

1. **In-Memory Cosine Similarity vs. HNSW Indexing**:
   - *Trade-off*: NumPy pairwise calculation scales linearly $\mathcal{O}(N \cdot D)$ with chunk count $N$.
   - *Justification*: For individual user libraries ($N < 5,000$ chunks), linear calculation takes $< 5\text{ ms}$, entirely eliminating the memory footprint and operational complexity of approximate nearest neighbor (ANN) vector database servers.
2. **Text-Only PDF Extraction**:
   - *Limitation*: Relies on `PyPDF2`, meaning scanned image-only PDFs without an embedded OCR text stream cannot be extracted.
3. **Bi-Encoder vs. Cross-Encoder Reranking**:
   - *Trade-off*: Bi-encoder retrieval (`all-MiniLM-L6-v2`) provides fast vector search but lacks the fine-grained cross-attention ranking of cross-encoders.
   - *Justification*: Sufficiently accurate for single-domain document QA while keeping local CPU overhead negligible.

---

## 7. Conclusion & Future Work

TeleRAG demonstrates that an effective, reliable, and secure Retrieval-Augmented Generation system does not require heavy multi-service infrastructure or complex orchestration frameworks. By applying strict similarity thresholding, defensive HTTP ingestion, relational foreign key cascades, and local sentence embeddings, TeleRAG achieves robust factual grounding and low-latency interaction over Telegram.

### Future Work
* Integration of lightweight Tesseract OCR for scanned PDF support.
* Cross-encoder reranking (e.g. `bge-reranker-base`) for complex multi-document queries.
* Conversion of SQLite vector storage to the embedded `sqlite-vec` extension for unified relational and vector indexing in a single binary file.
