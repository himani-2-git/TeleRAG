# TeleRAG

![CI](https://github.com/himani-2-git/TeleRAG/actions/workflows/ci.yml/badge.svg)

A lightweight Telegram bot for grounded question answering over PDFs, markdown, text files and static web pages, running on local embeddings with no external vector database required by default (Chroma is optional).

**Project status**: v1.1.0: versioned prompts, grounding gate, faithfulness evaluation (experimental, unvalidated judge), markdown/text/web ingestion, optional Chroma and LangGraph.

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Quality Gate](https://github.com/himani-2-git/TeleRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/himani-2-git/TeleRAG/actions/workflows/ci.yml)
![RAG Architecture](https://img.shields.io/badge/RAG-Hybrid%20%2B%20RRF%20%2B%20Reranker-orange)

<!-- demo video goes here -->

## What It Does

- Ingests multiple formats: PDF files, Markdown (`.md`), plain text (`.txt`), and static web pages via `/addurl <url>` with multi-hop SSRF defense.
- Retrieves passages using local bi-encoder embeddings (`all-MiniLM-L6-v2`) and BM25 lexical search merged with Reciprocal Rank Fusion ($k=60$), with optional cross-encoder reranking.
- Gated generation: evaluates dense cosine similarity threshold (`SIMILARITY_THRESHOLD=0.3`) and abstains when context is insufficient.
- Generates answers with an instruction-tuned LLM (local Ollama or hosted OpenAI-compatible provider) and appends a verified Sources citation footer.
- Supports modular storage (`VECTOR_BACKEND=numpy` default, `chroma` optional) and execution (`ORCHESTRATOR=plain` default, `langgraph` optional).

## Architecture

```
User (Telegram) ──> BotHandlers ──> RAGPipeline ──> [retrieve]
                                        │                 │
                                        ▼                 ▼
                              [abstain_or_generate] <── BM25 + Embeddings
                                        │               (NumPy / Chroma)
                                        ▼
                                   [grounding]
                                        │
                                        ▼
                                 [format_reply] ──> Telegram Answer + Sources
```

## Quickstart

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements/base.txt -r requirements/dev.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
```

Ensure [Ollama](https://ollama.ai) is running locally (`ollama pull qwen2.5:3b`) or set your OpenAI-compatible endpoint in `.env`.

### 3. Run the Bot

```bash
python main.py
```

## Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | - | Telegram Bot API token from @BotFather |
| `LLM_API_KEY` | - | Optional API key for remote LLM providers |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint URL |
| `LLM_MODEL` | `qwen2.5:3b` | Target LLM model name |
| `LLM_MAX_TOKENS` | `1024` | Maximum tokens for generated answers |
| `LLM_TIMEOUT_SECONDS` | `120` | Read timeout for LLM generation requests |
| `MAX_RETRIES` | `3` | Maximum retry attempts for LLM calls |
| `RETRY_DELAY` | `1.0` | Base retry delay in seconds |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload file size in megabytes |
| `MAX_CHUNKS_PER_DOCUMENT` | `5000` | Maximum chunk limit per document |
| `CHUNK_SIZE` | `1000` | Target character length per text chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name for dense retrieval |
| `TOP_K_RESULTS` | `2` | Number of context chunks retrieved for answer generation |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to attempt retrieval |
| `RETRIEVAL_MODE` | `hybrid` | Retrieval mode (`dense`, `hybrid`, `hybrid_rerank`) |
| `VECTOR_BACKEND` | `numpy` | Vector store backend (`numpy`, `chroma`) |
| `CHROMA_PATH` | `data/chroma` | Directory path for ChromaDB storage |
| `ORCHESTRATOR` | `plain` | Pipeline orchestrator (`plain`, `langgraph`) |
| `PROMPT_VERSION` | `v1` | Prompt version loaded from `prompts/<version>.yaml` |
| `GROUNDING_THRESHOLD` | `0.45` | NLI entailment threshold for grounding verification |
| `GROUNDING_ENFORCE` | `false` | When true, rejects generations failing grounding coverage |
| `GROUNDING_MIN_COVERAGE` | `0.5` | Ratio of supported sentences required to pass grounding |
| `DATABASE_PATH` | `data/documents.db` | SQLite database file path |
| `EMBEDDINGS_PATH` | `data/embeddings` | Directory path for cached NumPy embeddings |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FILE` | `logs/bot.log` | Log file path |

For full deployment and configuration options, see [docs/setup.md](docs/setup.md).

## Measured Retrieval Results

Retrieval evaluation reports two complementary metrics to evaluate physical page citations:
- **Span metric**: A retrieval is counted as a hit if the chunk's physical page interval `[page_start, page_end]` overlaps any target ground-truth page.
- **Strict metric**: A retrieval is counted as a hit only if the chunk's starting page `page_start` strictly matches a target ground-truth page.

### Real-Document Golden Set (*Think Python 2e*)
Evaluated on *Think Python (2nd Edition)* (`eval/fixtures/book.pdf`, 240 pages, 582 chunks of 1,000 characters with 200 overlap) across 70 queries: 50 answerable questions spanning all 19 chapters, 10 easy unanswerable off-topic questions, and 10 hard unanswerable Python questions absent from the book.

| Strategy | Mode | Hit@1 (Span) | Hit@1 (Strict) | Hit@5 (Span) | Hit@5 (Strict) | MRR (Span) | MRR (Strict) | Easy Abst. | Hard Abst. | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Dense** | `all-MiniLM-L6-v2` | 66.0% (33/50) | 46.0% (23/50) | **100.0%** (50/50) | 90.0% (45/50) | 0.7897 | 0.6243 | **100%** (10/10) | 0.0% (0/10) | **56.9 ms** |
| **Hybrid (Default)** | BM25 + Dense RRF ($k=60$) | 70.0% (35/50) | 52.0% (26/50) | 96.0% (48/50) | 84.0% (42/50) | 0.8073 | 0.6490 | **100%** (10/10) | 0.0% (0/10) | 57.8 ms |
| **Hybrid + Reranker** | Cross-Encoder MiniLM | **76.0%** (38/50) | **62.0%** (31/50) | 98.0% (49/50) | **92.0%** (46/50) | **0.8507** | **0.7323** | **100%** (10/10) | 0.0% (0/10) | 2,020.6 ms |

### CI Synthetic Fixture Gate
Evaluated on the synthetic distributed systems document (`eval/fixtures/ci_fixture.pdf`, 8 pages, 10 chunks) across 15 queries (`eval/golden_ci.json`): 12 answerable questions and 3 easy unanswerable off-topic questions (0 hard unanswerable).

| Strategy | Mode | Hit@1 (Span) | Hit@1 (Strict) | Hit@5 (Span) | Hit@5 (Strict) | MRR (Span) | MRR (Strict) | Easy Abst. | Hard Abst. | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (CI Gate)** | BM25 + Dense RRF ($k=60$) | 100.0% (12/12) | 16.7% (2/12) | 100.0% (12/12) | 83.3% (10/12) | 1.0000 | 0.4861 | 100% (3/3) | n/a (0/0) | 30.4 ms |

For detailed breakdown, latency benchmarks, ablation experiments, and faithfulness analysis, see [docs/technical-report.md](docs/technical-report.md).

## Running Tests and Evaluations

```bash
# Run unit tests
python -m unittest discover tests

# Lint and format checks
ruff check .
ruff format --check .

# Run CI retrieval evaluation quality gate
python eval/run_eval.py --mode hybrid --pdf eval/fixtures/ci_fixture.pdf --golden eval/golden_ci.json --thresholds eval/thresholds.json

# Run CI grounding gate
python eval/run_grounding_gate.py --pdf eval/fixtures/ci_fixture.pdf --answers eval/fixtures/ci_answers.json --thresholds eval/thresholds.json
```

## Documentation

- [Setup Guide](docs/setup.md): Complete installation, multi-format ingestion, SSRF protection, commands, and troubleshooting.
- [Technical Report](docs/technical-report.md): In-depth benchmarks, retrieval metrics, faithfulness evaluation, and ablation results.
