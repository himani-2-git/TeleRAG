# TeleRAG: Setup & Operational Instructions

This document provides a comprehensive, step-by-step guide to installing, configuring, running, testing, and troubleshooting the TeleRAG Telegram PDF Question Answering application.

---

## 1. System Prerequisites

### Supported Python Versions
* **Required**: Python 3.10, 3.11, or 3.12 (Tested on **Python 3.12.10**)
* **Important**: Python 3.13+ is **not supported** due to upstream constraints in `sentence-transformers==2.7.0` (`requires-python: <3.13, >=3.8`).

### Verify Your Python Installation
Open a terminal / PowerShell window:

```bash
python --version
```
If your default Python is 3.13 or newer, specify your Python 3.12 binary (e.g., `py -3.12` or `python3.12`).

---

## 2. Environment Setup

### 2.1 Clone Repository
```bash
git clone https://github.com/himani-2-git/TeleRAG.git
cd TeleRAG
```

### 2.2 Create a Virtual Environment
Isolate dependencies by creating a dedicated virtual environment:

- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Windows (Command Prompt - cmd)**:
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2.3 Install Core & Dev Dependencies
Install the pinned dependencies from `requirements/base.txt` (and optionally `requirements/dev.txt` for evaluation tooling):

```bash
pip install -r requirements/base.txt -r requirements/dev.txt
```

> [!NOTE]
> On the first run, the embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded and cached automatically from Hugging Face. This step requires an active internet connection once; subsequent runs execute completely offline on local CPU.

---

## 3. API Credentials & Configuration

TeleRAG requires a Telegram Bot Token (from BotFather) and an OpenAI-compatible LLM endpoint (defaulting to local Ollama with zero external keys).

### 3.1 Create a Telegram Bot Token
1. Open the Telegram app and search for `@BotFather`.
2. Start a chat and send `/newbot`.
3. Choose a friendly name for your bot (e.g. `My PDF RAG Assistant`).
4. Choose a username ending in `bot` (e.g. `my_pdf_rag_bot`).
5. BotFather will provide an HTTP API token (format: `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ`).
6. Copy this token.

### 3.2 Configure the LLM

#### Option A: Local Ollama Server (Documented Default — No API key needed)
1. Install [Ollama](https://ollama.ai) on your system.
2. Pull a local model, such as Qwen 2.5:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Run `ollama list` to inspect your installed models.
4. Set `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=qwen2.5:3b` in `.env`.

#### Optional: use a hosted provider (Groq, Google AI Studio, or other endpoints)
To route requests to a hosted OpenAI-compatible endpoint instead of running locally:
1. Obtain an API key from your provider (e.g. Groq or Google AI Studio).
2. Set the base URL pattern to your provider's OpenAI-compatible endpoint. Choose an active model and verify rate limits and availability.
3. Set `LLM_BASE_URL=https://api.example.com/v1`, `LLM_API_KEY=your_api_key_here`, and `LLM_MODEL=your_chosen_model_name` in `.env`.

### 3.3 Configure `.env`
Copy the template configuration file:

- **Linux / macOS**:
  ```bash
  cp .env.example .env
  ```
- **Windows (PowerShell)**:
  ```powershell
  Copy-Item .env.example .env
  ```

Open `.env` in an editor and set at least `TELEGRAM_BOT_TOKEN`. Every setting, with its default and a
description, is listed in the [README](../README.md#configuration), and `.env.example` explains the
access settings (`ALLOWED_USER_IDS`, `ADMIN_USER_IDS`). The bot checks all settings at startup and
prints one list of every problem it finds.

---

## 4. Verification & Testing

Before launching the bot, verify your environment by running the unit test suite:

```bash
python -m unittest discover tests
```

Expected output (the number of tests grows over time):
```text
Ran N tests in S seconds

OK
```
This validates SSRF defense, multi-hop redirect checks, streaming limits, per-page PDF parsing, sliding-window chunk bounds, page citations and source footers, SQLite schema migrations, BM25 indexing, RRF rank fusion, cross-encoder reranking, YAML prompt loading, grounding checks, and cosine similarity threshold filtering without making external network calls.

### 4.1 Running the Retrieval Evaluation Quality Gate
To verify retrieval performance against the synthetic regression fixture:
```bash
python eval/run_eval.py --mode hybrid --pdf eval/fixtures/ci_fixture.pdf --golden eval/golden_ci.json --thresholds eval/thresholds.json
```
All quality gate assertions (`min_hit_at_5 >= 0.85`, `min_mrr >= 0.70`, `min_abstention_easy >= 0.90`) should pass.

### 4.2 Running the Grounding Evaluation Quality Gate
To verify citation grounding and rejection of ungrounded hallucinations:
```bash
python eval/run_grounding_gate.py --pdf eval/fixtures/ci_fixture.pdf --answers eval/fixtures/ci_answers.json --thresholds eval/thresholds.json
```
Verifies that mean reference answer citation coverage meets the threshold ($\ge 0.85$) and negative controls score $\le 0.40$.

### 4.3 Running Local Faithfulness Evaluation (Optional)
To run claim-level entailment evaluation with an LLM judge:
```bash
pip install -r requirements/eval.txt
python -m eval.run_faithfulness --n 20 --judge-model qwen2.5:3b --gen-model qwen2.5:3b --output eval/results/faithfulness.json
```
*(Note: A local 3B judge is noisy and intended for directional trend-checking, not a strict CI gate).*

---

## 5. Running the Application

### 5.1 Start the Bot
Run the main script from your activated virtual environment:

```bash
python main.py
```

At startup the bot checks the settings, then the LLM server and the model, then opens the database and
loads the embedding model. If something is wrong it stops with a message that says what to fix. When all
checks pass, the console shows a line like this:
```text
Bot is ready! Starting polling...
```

The application is now actively polling Telegram for incoming messages.

### 5.2 Stop the Bot
To stop the bot, press `Ctrl + C` in the console window. The bot stops taking new messages and finishes the answer it is writing first, so stopping can take up to a few minutes. To run the bot in Docker instead, see the [README](../README.md#setup-with-docker).

---

## 6. Telegram User Guide

Once the bot is running, open Telegram and search for your bot's username.

### 6.1 Available Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `/start` | Bot Introduction | Sends a welcome overview and basic instructions. |
| `/help` | Detailed Help | Explains document upload options and question syntax. |
| `/list` | Show Documents | Lists all active documents in your library with file sizes and chunk counts. |
| `/addurl <url>` | Ingest Web Page | Ingests a public web page with SSRF protection and extracts article content. |
| `/delete 2` or `/delete part-of-name` | Delete Specific File | Deletes a document, its database chunks, and its vector embeddings (precedence: exact name, list number, substring). |
| `/clear` | Wipe Library | Removes all uploaded documents and embeddings for your user account. |
| `/status` (or `/health`) | Check Services | Admins only (`ADMIN_USER_IDS`): shows whether the database, the embedding model and the LLM server work. |

### 6.2 Uploading Documents and Ingesting Content

#### Option A: Direct File Upload (PDF, Markdown, Plain Text)
1. Click the attachment icon (paperclip) in Telegram.
2. Select a document file (under 20 MB):
   - **PDF (`.pdf`)**: Extracted page by page, preserving page-level citations (`(p. X)` or `(pp. X-Y)`).
   - **Markdown (`.md`)**: Structure-aware parsing splits content along heading boundaries (`#`, `##`, `###`), preserving full section paths in citations (e.g., `guide.md > Installation`). Sections exceeding chunk size are split with 20% overlap.
   - **Plain Text (`.txt`)**: Clean sliding-window chunking, cited as chunk indices (`doc.txt (chunk 1)`).
3. Send it to the bot.
4. The bot responds:
   - *"Processing your document..."*
   - *"✅ Successfully processed <filename> | 📊 Created N chunks"*

#### Option B: Public PDF URL Upload
1. Send a direct link to any public PDF file in chat:
   ```text
   https://greenteapress.com/thinkpython2/thinkpython2.pdf
   ```
2. The bot validates the URL, downloads it with streaming size bounds, parses chunks, and saves embeddings.

#### Option C: Web Page Ingestion (`/addurl <url>`)
1. Send `/addurl <url>` with any public web article:
   ```text
   /addurl https://en.wikipedia.org/wiki/Raft_(algorithm)
   ```
2. **Content Extraction**: Strips non-content boilerplate tags (`<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`), extracts title and semantic body text, and chunks it cleanly.
3. **SSRF Protection**:
   - Only `http` and `https` schemes are accepted; dangerous protocols (`file://`, `ftp://`, `gopher://`, etc.) are rejected immediately.
   - Resolves hostnames and strictly blocks private, loopback, link-local, multicast, and cloud metadata addresses (e.g. `127.0.0.1`, `10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`, `169.254.169.254`).
   - Web pages are limited to 2 MB, 15 s per wait on the network and 45 s in total. PDF links are limited to `MAX_FILE_SIZE_MB`, 30 s per wait and 60 s in total.
   - Known gap: the address is checked and then looked up again when connecting (DNS rebinding). See the README for details.

---

## 7. Vector Backend Options

TeleRAG supports modular vector store backends configured via `VECTOR_BACKEND` in `.env`:

### 7.1 NumPy Vector Store (`VECTOR_BACKEND=numpy` — Default)
- **Zero Extra Dependencies**: Backed by the built-in SQLite metadata store (`data/documents.db`) and one `.npy` embedding file per document (`data/embeddings/`).
- **In-Process Similarity**: Uses standard NumPy cosine similarity with zero server overhead. Ideal for lightweight deployments, low-memory embedded machines, and standard CI environments.

### 7.2 ChromaDB Vector Store (`VECTOR_BACKEND=chroma` — Optional)
- **Persistent Vector Store**: Backed by ChromaDB with cosine distance metric (`hnsw:space: cosine`).
- **Per-User Scoping**: Every query filters chunks using Chroma metadata (`{"user_id": user_id}`), so a user only gets their own chunks.
- **Enabling Chroma**:
  1. Install the optional dependency:
     ```bash
     pip install -r requirements/chroma.txt
     ```
  2. Set environment variables in `.env`:
     ```bash
     VECTOR_BACKEND=chroma
     CHROMA_PATH=data/chroma
     ```
  3. If `VECTOR_BACKEND=chroma` is specified without `chromadb` installed, TeleRAG logs a warning and gracefully falls back to `NumPyVectorStore` without crashing.

---

## 8. Pipeline Orchestration Options

TeleRAG supports modular RAG orchestration configured via `ORCHESTRATOR` in `.env`:

### 8.1 Plain Orchestrator (`ORCHESTRATOR=plain` — Default)
- Executes the 4 pipeline steps sequentially in pure Python:
  1. `retrieve`: Queries vector store and BM25 for candidate chunks.
  2. `abstain_or_generate`: Checks similarity threshold; either abstains with refusal or invokes LLM.
  3. `grounding`: Compares each answer sentence with the retrieved chunks by cosine similarity and records how many sentences are supported.
  4. `format_reply`: Formats citations and the Sources footer.

### 8.2 LangGraph Orchestrator (`ORCHESTRATOR=langgraph` — Optional)
- Compiles the identical 4 nodes into an executable LangGraph `StateGraph`:
  ```
  [retrieve] ──> [abstain_or_generate] ──> [grounding] ──> [format_reply] ──> [END]
  ```
- **Parity**: Uses the same component methods as the plain orchestrator. Tests run both on the CI fixture and compare the answers, refusals and footers.
- **Enabling LangGraph**:
  1. Install the optional dependency:
     ```bash
     pip install -r requirements/graph.txt
     ```
  2. Set environment variable in `.env`:
     ```bash
     ORCHESTRATOR=langgraph
     ```
  3. If `ORCHESTRATOR=langgraph` is set without `langgraph` installed, TeleRAG logs a warning and gracefully falls back to `plain` execution without crashing.

---

### 6.3 Asking Questions
Once at least one document is uploaded, simply send natural language questions:
* *"What is recursion?"*
* *"Explain the difference between mutable and immutable types."*
* *"Summarize this document."* (uses 8 chunks spread over the whole document)

#### Grounding & Abstention Behavior
* **In-Context Queries**: If relevant chunks meet the similarity threshold (`>= 0.3`), the bot constructs a context prompt for the LLM and returns an answer citing retrieved page ranges. This relevance cutoff reduces answers not supported by the document; does not eliminate hallucination.
* **Out-of-Context Queries**: If no chunks meet the threshold (e.g. asking *"How to bake bread?"* on a Python textbook), the bot abstains without calling the LLM:
  > *"🤔 I couldn't find relevant information in your documents to answer this question."*

---

## 9. Troubleshooting & Common Issues

### 1. HTTP 406 "Not Acceptable" on URL Uploads
* **Symptom**: The bot reports a 406 Client Error when downloading a PDF from an institutional or hosting server.
* **Cause**: Apache Mod_Security rules block generic script user agents.
* **Resolution**: TeleRAG automatically sends a compliant `User-Agent: TeleRAG/1.0 (PDF Document Assistant)` and `Accept: application/pdf` header. Verify that the URL directly serves a PDF file and is publicly accessible without requiring a login cookie.

### 2. "User Safety: safe" Response from LLM
* **Symptom**: The bot replies *"User Safety: safe"* to all questions instead of answering.
* **Cause**: Generic router endpoints or guardrail classifiers return safety verdicts instead of generated answers.
* **Resolution**: Set `LLM_MODEL` to a specific instruction-tuned model (e.g. `qwen2:7b` on Ollama, or a specific model name on your provider).

### 3. HTTP 429 "Too Many Requests"
* **Symptom**: Error in logs `429 Too Many Requests`.
* **Cause**: Upstream provider rate limits on free-tier models during peak periods.
* **Resolution**: TeleRAG includes automatic retry logic (retrying at most once on 429). If a model remains congested, switch `LLM_MODEL` in `.env` to another model or run locally with Ollama.

### 4. PDF Text Extraction Returns 0 Chunks
* **Symptom**: *"❌ No text content could be extracted from PDF."*
* **Cause**: The PDF is a scanned image or raster graphic without an embedded OCR text layer.
* **Resolution**: TeleRAG uses `pypdf` text extraction; use an OCR tool (e.g. Adobe Acrobat OCR or ocrmypdf) to convert scanned PDFs into searchable text before uploading.
