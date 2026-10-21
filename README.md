# TeleRAG

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
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint URL |
| `LLM_MODEL` | `qwen2.5:3b` | Target LLM model name |
| `RETRIEVAL_MODE` | `hybrid` | Retrieval mode (`dense`, `hybrid`, `hybrid_rerank`) |
| `VECTOR_BACKEND` | `numpy` | Vector store backend (`numpy`, `chroma`) |
| `ORCHESTRATOR` | `plain` | Pipeline orchestrator (`plain`, `langgraph`) |
| `PROMPT_VERSION` | `v1` | Prompt version loaded from `prompts/<version>.yaml` |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to attempt retrieval |
| `GROUNDING_ENFORCE` | `false` | When true, rejects generations failing grounding coverage |

For full deployment and configuration options, see [docs/setup.md](docs/setup.md).

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
