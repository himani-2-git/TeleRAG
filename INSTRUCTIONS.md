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
Install the pinned dependencies from `requirements.txt` (and optionally `requirements-dev.txt` for evaluation tooling):

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

> [!NOTE]
> On the first run, the embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded and cached automatically from Hugging Face. This step requires an active internet connection once; subsequent runs execute completely offline on local CPU.

---

## 3. API Credentials & Configuration

TeleRAG requires two external services: a Telegram Bot and an OpenRouter API key.

### 3.1 Create a Telegram Bot Token
1. Open the Telegram app and search for `@BotFather`.
2. Start a chat and send `/newbot`.
3. Choose a friendly name for your bot (e.g. `My PDF RAG Assistant`).
4. Choose a username ending in `bot` (e.g. `my_pdf_rag_bot`).
5. BotFather will provide an HTTP API token (format: `123456789:ABCdefGHIjklMNOpqrSTUvwxYZ`).
6. Copy this token.

### 3.2 Obtain an OpenRouter API Key
1. Go to [OpenRouter](https://openrouter.ai/) and register an account.
2. Navigate to **Keys** (`openrouter.ai/keys`) and click **Create Key**.
3. Copy your API key (format: `sk-or-v1-...`).
4. *Note: free model names change often; pick a model marked free there and set OPENROUTER_MODEL: https://openrouter.ai/models*

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

Open `.env` in an editor and enter your credentials:

```ini
# ===================================================================
# Required Credentials
# ===================================================================
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# ===================================================================
# LLM Model Configuration
# ===================================================================
# free model names change often; pick a model marked free there and set OPENROUTER_MODEL: https://openrouter.ai/models
OPENROUTER_MODEL=qwen/qwen3.8-27b:free
MAX_RETRIES=3
RETRY_DELAY=1.0

# ===================================================================
# RAG Retrieval & Chunking Settings
# ===================================================================
RETRIEVAL_MODE=hybrid
MAX_FILE_SIZE_MB=20
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3

# ===================================================================
# Storage & Logging Paths
# ===================================================================
DATABASE_PATH=data/documents.db
EMBEDDINGS_PATH=data/embeddings
LOG_LEVEL=INFO
LOG_FILE=logs/bot.log
```

---

## 4. Verification & Testing

Before launching the bot, verify your environment by running the unit test suite:

```bash
python -m unittest discover tests
```

Expected output:
```text
Ran unit tests in ~10s

OK
```
This validates SSRF defense, multi-hop redirect checks, streaming limits, per-page PDF parsing, sliding-window chunk bounds, page citations and source footers, SQLite schema migrations, BM25 indexing, RRF rank fusion, cross-encoder reranking, and cosine similarity threshold filtering without making external network calls.

### 4.1 Running the Retrieval Evaluation Quality Gate
To verify retrieval performance against the synthetic regression fixture:
```bash
python eval/run_eval.py --mode hybrid --pdf eval/fixtures/ci_fixture.pdf --golden eval/golden_ci.json --thresholds eval/thresholds.json
```
All quality gate assertions (`min_hit_at_5 >= 0.85`, `min_mrr >= 0.70`, `min_abstention_easy >= 0.90`) should pass.

---

## 5. Running the Application

### 5.1 Start the Bot
Run the main script from your activated virtual environment:

```bash
python main.py
```

Console logs will confirm each initialization stage:
```text
YYYY-MM-DD HH:MM:SS - __main__ - INFO - Starting PDF RAG Chatbot...
YYYY-MM-DD HH:MM:SS - __main__ - INFO - Initializing components...
YYYY-MM-DD HH:MM:SS - src.storage.database - INFO - Database schema initialized successfully
YYYY-MM-DD HH:MM:SS - src.rag.embeddings - INFO - Loading embedding model: all-MiniLM-L6-v2
YYYY-MM-DD HH:MM:SS - src.rag.embeddings - INFO - Embedding model loaded successfully
YYYY-MM-DD HH:MM:SS - __main__ - INFO - Bot is ready! Starting polling...
YYYY-MM-DD HH:MM:SS - telegram.ext.Application - INFO - Application started
```

The application is now actively polling Telegram for incoming messages.

### 5.2 Stop the Bot
To terminate execution, press `Ctrl + C` in the console window.

---

## 6. Telegram User Guide

Once the bot is running, open Telegram and search for your bot's username.

### 6.1 Available Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `/start` | Bot Introduction | Sends a welcome overview and basic instructions. |
| `/help` | Detailed Help | Explains document upload options and question syntax. |
| `/list` | Show Documents | Lists all active documents in your library with file sizes and chunk counts. |
| `/delete <filename>` | Delete Specific File | Deletes a document, its database chunks, and its vector embeddings. |
| `/clear` | Wipe Library | Removes all uploaded documents and embeddings for your user account. |

### 6.2 Uploading Documents

#### Option A: Direct File Upload
1. Click the attachment icon (paperclip) in Telegram.
2. Select a text-based PDF file (under 20 MB).
3. Send it to the bot.
4. The bot responds:
   - *"Processing your PDF..."*
   - *"✅ Successfully processed <filename> | 📊 Created N chunks"*

#### Option B: Public URL Upload
1. Send a direct link to any public PDF file in chat:
   ```text
   https://greenteapress.com/thinkpython2/thinkpython2.pdf
   ```
2. The bot validates the URL, downloads it with streaming size bounds, parses chunks, and saves embeddings.

### 6.3 Asking Questions
Once at least one document is uploaded, simply send natural language questions:
* *"What is recursion?"*
* *"Explain the difference between mutable and immutable types."*
* *"Summarize chapter 3."*

#### Grounding & Abstention Behavior
* **In-Context Queries**: If relevant chunks meet the similarity threshold (`>= 0.3`), the bot constructs a context prompt for OpenRouter and returns an answer citing retrieved page ranges. This relevance cutoff reduces answers not supported by the document; does not eliminate hallucination.
* **Out-of-Context Queries**: If no chunks meet the threshold (e.g. asking *"How to bake bread?"* on a Python textbook), the bot abstains without calling the LLM:
  > *"🤔 I couldn't find relevant information in your documents to answer this question."*

---

## 7. Troubleshooting & Common Issues

### 1. HTTP 406 "Not Acceptable" on URL Uploads
* **Symptom**: The bot reports a 406 Client Error when downloading a PDF from an institutional or hosting server.
* **Cause**: Apache Mod_Security rules block generic script user agents.
* **Resolution**: TeleRAG automatically sends a compliant `User-Agent: TeleRAG/1.0 (PDF Document Assistant)` and `Accept: application/pdf` header. Verify that the URL directly serves a PDF file and is publicly accessible without requiring a login cookie.

### 2. "User Safety: safe" Response from LLM
* **Symptom**: The bot replies *"User Safety: safe"* to all questions instead of answering.
* **Cause**: Using `openrouter/free` routes requests nondeterministically across all free models, which includes guardrail safety classifiers (e.g. `nvidia/nemotron-3.5-content-safety:free`).
* **Resolution**: Keep `OPENROUTER_MODEL=qwen/qwen3.8-27b:free` configured in your `.env`.

### 3. OpenRouter HTTP 429 "Too Many Requests"
* **Symptom**: Error in logs `429 Too Many Requests`.
* **Cause**: Upstream provider rate limits on free-tier models during peak periods.
* **Resolution**: TeleRAG includes automatic exponential backoff retry logic. If a model remains congested, switch `OPENROUTER_MODEL` in `.env` to another free model listed at https://openrouter.ai/models.

### 4. PDF Text Extraction Returns 0 Chunks
* **Symptom**: *"❌ No text content could be extracted from PDF."*
* **Cause**: The PDF is a scanned image or raster graphic without an embedded OCR text layer.
* **Resolution**: TeleRAG uses `PyPDF2` text extraction; use an OCR tool (e.g. Adobe Acrobat OCR or ocrmypdf) to convert scanned PDFs into searchable text before uploading.
