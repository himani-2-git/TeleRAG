# TeleRAG

A lightweight Telegram bot for grounded question answering over PDF documents, running on local embeddings without external vector databases.

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI Quality Gate](https://github.com/himani-2-git/TeleRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/himani-2-git/TeleRAG/actions/workflows/ci.yml)
![RAG Architecture](https://img.shields.io/badge/RAG-Hybrid%20%2B%20RRF%20%2B%20Reranker-orange)

---

## What It Does

• Ingests PDF documents via direct Telegram file uploads or public web links with multi-hop redirect verification.  
• Retrieves relevant passages using local bi-encoder embeddings and BM25 lexical search merged with Reciprocal Rank Fusion ($k=60$), with optional cross-encoder reranking.  
• Generates answers using an instruction-tuned LLM (local Ollama or hosted provider) and appends a verified Sources footer citing retrieved physical page numbers.

Retrieval was evaluated on a 70-question set over *Think Python 2e*, a smaller synthetic fixture set runs in CI to guard against regressions, and a Sources footer lists the retrieved pages for each answer.

---

<!-- demo video goes here -->

---

## Why I Built It & What I Learned

I built TeleRAG to understand how a production-style RAG pipeline behaves end-to-end without relying on managed vector database services or opaque framework abstractions. Building this project taught me that chunk boundaries frequently split physical pages—causing citation errors unless tracked as intervals—and that combining BM25 with dense retrieval via Reciprocal Rank Fusion provides a practical hedge against vocabulary mismatch. Measuring each change against a verified 70-question golden set showed that dense and hybrid differ by only 1–2 questions on a 50-question set (within noise), whereas the real precision jump came from cross-encoder reranking at the cost of CPU inference latency.

---

## Measured Retrieval Results

All benchmarks below are measured on an Intel i5 CPU using `eval/run_eval.py` against *Think Python 2e* (582 chunks, ~1000 characters, 200 character overlap) and a verified golden dataset of 70 queries: 50 answerable questions spanning every chapter, 10 easy unanswerable off-topic questions (`u01`–`u10`), and 10 hard unanswerable programming questions (`u11`–`u20`, verified 100% absent from the text).

Metrics are reported under both **Span** (retrieved chunk's `[page_start, page_end]` interval overlaps the target page) and **Strict** (chunk strictly starts on target page):

| Strategy | Mode | Hit@1 (Span) | Hit@5 (Span) | MRR (Span) | Hit@1 (Strict) | Hit@5 (Strict) | MRR (Strict) | Easy Abstention | Hard Abstention | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Dense** | `all-MiniLM-L6-v2` | 66.0% (33/50) | **100.0%** (50/50) | 0.7897 | 46.0% (23/50) | 90.0% (45/50) | 0.6243 | **100%** (10/10) | 0.0% (0/10) | **56.9 ms** |
| **Hybrid (Default)** | BM25 + Dense RRF ($k=60$) | 70.0% (35/50) | 96.0% (48/50) | 0.8073 | 52.0% (26/50) | 84.0% (42/50) | 0.6490 | **100%** (10/10) | 0.0% (0/10) | 57.8 ms |
| **Hybrid + Reranker** | Cross-Encoder MiniLM | **76.0%** (38/50) | 98.0% (49/50) | **0.8507** | **62.0%** (31/50) | **92.0%** (46/50) | **0.7323** | **100%** (10/10) | 0.0% (0/10) | 2,020.6 ms |

### Key Empirical Findings
1. **Hybrid vs. Dense Baseline**: Dense and hybrid differ by 1-2 questions on a 50-question set and are within noise; the measurable gain comes from the reranker (hit@1 66% -> 76%, MRR 0.79 -> 0.85, at ~2 s/query on CPU).
2. **Cross-Encoder Reranking (+10.0 pp Hit@1 Span, +16.0 pp Strict)**: Running `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-20 hybrid candidates yields the strongest retrieval ranking (MRR 0.8507), trading ~2s of CPU latency for joint query-passage attention.
3. **Semantic Invariant & Hard Negative Failure**: Bi-encoders score 100% abstention on completely off-topic queries (scores < 0.22), but fail on out-of-scope in-domain queries (e.g. `asyncio`, `dataclass` score 0.35–0.48). This motivated our stratified dev/test abstention experiment below.

### Dev/Test Abstention Experiment
Using a balanced, stratified split (`eval/split.json`, seed=42; 35 DEV, 35 TEST; 25 answerable, 5 easy unanswerable, 5 hard unanswerable each), we evaluated an optional Cross-Encoder top-1 score gate ($T = -2.80$, tuned strictly on DEV) versus the production dense cosine threshold (0.3):

| Gatekeeper Strategy | Split | Answerable Recall | Easy Abstention | Hard Abstention | Decision Rule Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dense Cosine (0.3)** | TEST (N=35) | **100.0%** (25/25) | **100.0%** (5/5) | 0.0% (0/5) | Production baseline |
| **Cross-Encoder ($T=-2.80$)**| TEST (N=35) | **100.0%** (25/25) | **100.0%** (5/5) | **40.0%** (2/5) | Rejects 2/5 hard negatives with 0/25 (0.0%) recall loss |

*Limitations of the experiment*: While the cross-encoder gate rejected 2/5 hard negatives without penalizing answerable recall on TEST, the dev-tuned threshold may be overfit to the dev sample, the test sample size is small (5 hard test queries), and cross-encoder inference adds ~2s query latency on CPU. Dense cosine gating remains the default gatekeeper, with hybrid and reranking configurable via `RETRIEVAL_MODE`.

> [!NOTE]
> **End-to-End Latency & Hosting**: The retrieval benchmarks above report indexing and retrieval search latency (57 ms for dense/hybrid, 2.0s with reranking). End-to-end answer time depends heavily on the model and host hardware; on a laptop CPU, local generation with a 7B parameter model (e.g. `qwen2:7b` via Ollama) is measured in minutes. TeleRAG keeps users informed with an immediate thinking notice, periodic typing indicators, and concurrency queueing via a worker thread. Additionally, because this is self-hosted on your machine, the Telegram bot only runs and responds while the host machine is awake and powered on. For hosted cloud providers, free model endpoints return faster but can be rate-limited or retired over time.

---

## System Architecture

```
User (Telegram Client)
         │
         │  (HTTPS Telegram Bot API)
         ▼
┌────────────────────────────────────────────────────────┐
│             BotHandlers (Telegram Interface)           │
│  - User session & tenant isolation                     │
│  - Multi-hop SSRF validation & streaming limits        │
│  - Command routing: /start, /list, /delete, /clear     │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
      [Document Upload]             [User Question]
               │                          │
               ▼                          ▼
┌──────────────────────────────┐ ┌─────────────────────────────────────────┐
│     Page-Aware Ingestion     │ │            Retrieval Pipeline           │
│  - PDFProcessor (per-page)   │ │  - EmbeddingGenerator (all-MiniLM-L6)   │
│  - TextChunker (1000/200)    │ │  - Sparse BM25Okapi Indexing            │
│  - Dense EmbeddingGenerator  │ │  - Reciprocal Rank Fusion (RRF k=60)    │
│  - SQLite (page_start/end)   │ │  - Cross-Encoder Reranker (optional)    │
└──────────────┬───────────────┘ └──────────────┬──────────────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────────┐ ┌─────────────────────────────────────────┐
│        DocumentStore         │ │               LLM Client                │
│  - SQLite (Metadata & chunks)│ │  - Page citation context assembly       │
│  - JSON per-document vectors │ │  - Grounded answer generation           │
└──────────────────────────────┘ └──────────────┬──────────────────────────┘
                                                │
                                                ▼
                                    Telegram Answer Output
                                    + Verified Sources Footer
```

---

## Features

### 1. Document Ingestion & URL Fetching
* **Direct File Upload**: Upload local PDF documents up to 20 MB directly through Telegram.
* **URL-Based Ingestion**: Supply remote PDF web links with manual multi-hop redirect validation (up to 3 hops).
* **SSRF Defense**: Inspects each redirect hop and blocks private RFC1918 subnets (`10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`), loopback (`127.0.0.1`), and cloud metadata addresses (`169.254.169.254`).
* **Streaming Byte Bounds**: Enforces `MAX_FILE_SIZE_MB` chunk-by-chunk during streaming to prevent chunked-transfer memory exhaustion.

### 2. Page-Aware Chunking & Citations
* **Boundary-Aware Text Extraction**: Extracts text per physical page and tracks chunk boundary intervals (`page_start`, `page_end`).
* **Sliding-Window Chunking**: Character chunk size (`CHUNK_SIZE=1000`) and overlap (`CHUNK_OVERLAP=200`) with backward-compatible SQLite schema migrations.
* **Page-Level In-Text Citations**: Context formatted as `[<filename>, p. <page>]` instructing the LLM to cite page locations.
* **Programmatic Source Verification**: Responses automatically append a verified Sources footer extracted directly from retrieved database chunks.

### 3. Multi-Stage Retrieval
* **Local Bi-Encoder Model**: Generates 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` locally on host CPU.
* **Lexical BM25 Indexing**: In-memory `rank-bm25` index built over chunk tokens for exact term matching.
* **Reciprocal Rank Fusion (RRF)**: Merges dense semantic and BM25 rank lists ($k=60$) without requiring score calibration.
* **Cross-Encoder Reranking**: Optional deep re-ordering using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Relevance Cutoff Gatekeeper**: Enforces a dense cosine threshold (`SIMILARITY_THRESHOLD=0.3`) before calling the LLM. If no chunk meets the threshold, the bot abstains. This reduces answers not supported by the document; does not eliminate hallucination.

### 4. LLM Generation via Local Ollama or Hosted Endpoints
* **Local Inference (Default)**: Preconfigured for local [Ollama](https://ollama.ai) at `http://localhost:11434/v1` using locally hosted models (e.g. `qwen2.5:3b` via `ollama list`) with zero external API dependencies or fees.
* **Hosted Alternative**: Can optionally route to remote OpenAI-compatible endpoints by supplying an API key and base URL.
* **Versioned Prompts**: Loads structured prompt configurations from `prompts/<PROMPT_VERSION>.yaml` (default `v1`) with system prompt, context template, and explicit refusal instructions.
* **Post-Generation Grounding Check**: Measures sentence-level semantic alignment against retrieved passages (`GROUNDING_THRESHOLD=0.45`). By default, runs in log-only mode (`GROUNDING_ENFORCE=false`). When enforcement is enabled (`GROUNDING_ENFORCE=true`), answers with citation coverage below `GROUNDING_MIN_COVERAGE=0.5` are safely refused.
* **Dual Verified Footers**: Successful responses append physical page citations (`Sources:\n• <doc> (p. X)`) followed by generation latency and model attribution (`Answered in <N> s with <model>`).

### 5. Multi-User Isolation & Session Control
* **Tenant Isolation**: SQLite metadata and chunk records are indexed and scoped by Telegram `user_id`. Users can only query and manage their own documents.
* **Foreign Key Cascades**: SQLite database enforces `PRAGMA foreign_keys = ON`. Deleting a document removes associated chunk vectors and disk files.

---

## Quick Start

### 1. Prerequisites & Environment
```bash
python --version   # Must be Python 3.10, 3.11, or 3.12
python -m venv .venv
.\.venv\Scripts\activate   # Windows (or source .venv/bin/activate on Linux)
pip install -r requirements.txt -r requirements-dev.txt
```

> [!NOTE]
> On the first run, the embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded and cached automatically from Hugging Face. This requires an active internet connection once; subsequent runs execute completely offline on local CPU.

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
By default, TeleRAG runs completely offline using a local Ollama server:
```ini
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_from_botfather
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:3b
RETRIEVAL_MODE=hybrid
```

#### Run fully local with Ollama (Default)
TeleRAG supports fully offline, local inference via [Ollama](https://ollama.ai)'s OpenAI-compatible endpoint. No API key is required when pointing to localhost:

1. Install Ollama and pull your chosen model (e.g., Qwen 2.5):
   ```bash
   ollama pull qwen2.5:3b
   ```
2. Verify available models with `ollama list` and specify your model name in `LLM_MODEL` (e.g. `qwen2.5:3b`).
3. Start Ollama (`ollama serve`) and run the bot. Requests execute completely locally with an unblocked worker thread and generous 300s timeout tailored for CPU inference.

#### Choosing a local model

Measured results using Ollama on a **Dell Inspiron i5 with 16 GB RAM (CPU only)** across the 12-question benchmark (`python -m eval.bench_llm` on `eval/fixtures/ci_fixture.pdf`, saved in `eval/results/llm_bench.json`):

| Model | Median Latency (Answered) | P95 Latency | Answerable Answered % | Unanswerable Refused % |
| :--- | :--- | :--- | :--- | :--- |
| `qwen2.5:3b` *(default)* | ~18 s | ~21 s | 100% | 100% |
| `llama3.2:3b` | ~18 s | ~21 s | 100% | 100% |

* **Refusals**: Take under 3 s because no LLM call is made when queries fail the similarity threshold.
* **Full-Document Scale**: On a full book (~600 chunks) in the Telegram bot, real answers took about 20 to 40 s with `qwen2.5:3b`.
* **Earlier 7B Baseline**: The earlier default `qwen2:7b` yielded ~3 tokens/s generation, taking 10 to 20 minutes per answer on this laptop CPU, and answered "RAG" wrongly.

> [!NOTE]
> This benchmark measures answered-vs-refused behaviour and latency, not answer correctness; faithfulness evaluation is reported under Faithfulness evaluation below.

#### Optional: use a hosted provider (OpenRouter, Groq, Google AI Studio)
To route to a hosted OpenAI-compatible endpoint instead of running locally:
1. Obtain an API key from your provider (e.g. [OpenRouter](https://openrouter.ai/keys), Groq, or Google AI Studio).
2. For OpenRouter, set the base URL pattern to `https://openrouter.ai/api/v1`. Choose an active model (e.g. check [https://openrouter.ai/models](https://openrouter.ai/models) for currently free models; verify rate limits and availability, as free model IDs rotate frequently).
3. Update `.env`:
   ```ini
   LLM_BASE_URL=https://openrouter.ai/api/v1
   LLM_API_KEY=your_api_key_here
   LLM_MODEL=your_chosen_model_name
   ```

### 3. Verify & Run Test Suite
```bash
# Execute unit test suite (all tests should pass)
python -m unittest discover tests

# Launch the bot
python main.py
```

---

## Telegram Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `/start` | Bot Introduction | Sends a welcome overview and basic instructions. |
| `/help` | Detailed Help | Explains document upload options and question syntax. |
| `/list` | Show Documents | Lists all active documents in your library with file sizes and chunk counts. |
| `/delete <filename>` | Delete Specific File | Deletes a document, its database chunks, and its vector embeddings. |
| `/clear` | Wipe Library | Removes all uploaded documents and embeddings for your user account. |

---

## Evaluation

TeleRAG includes a reproducible evaluation harness:

```bash
# Evaluate baseline dense retrieval on Think Python 2e (70 questions)
python eval/run_eval.py --mode dense --pdf eval/fixtures/book.pdf --golden eval/golden.json --out eval/results/baseline.json

# Evaluate hybrid BM25 + dense retrieval
python eval/run_eval.py --mode hybrid --pdf eval/fixtures/book.pdf --golden eval/golden.json --out eval/results/hybrid.json

# Evaluate hybrid retrieval + cross-encoder reranking
python eval/run_eval.py --mode hybrid_rerank --pdf eval/fixtures/book.pdf --golden eval/golden.json --out eval/results/hybrid_rerank.json

# Run CI retrieval evaluation quality gate on synthetic 8-page fixture
python eval/run_eval.py --mode hybrid --pdf eval/fixtures/ci_fixture.pdf --golden eval/golden_ci.json --thresholds eval/thresholds.json

# Run CI grounding quality gate (verifies reference citations & rejects hallucinations)
python eval/run_grounding_gate.py --pdf eval/fixtures/ci_fixture.pdf --answers eval/fixtures/ci_answers.json --thresholds eval/thresholds.json

# Validate NLI faithfulness judge against reference answers & negative controls (default)
python -m eval.run_faithfulness --validate-judge --judge-type nli

# Validate LLM faithfulness judge against reference answers & negative controls
python -m eval.run_faithfulness --validate-judge --judge-type llm --judge-model llama3.2:3b

# Run local faithfulness evaluation on Think Python 2e (NLI judge, 20 answerable questions)
# Note: Install optional evaluation dependencies via `pip install -r requirements-eval.txt`
python -m eval.run_faithfulness --judge-type nli --n 20 --output eval/results/faithfulness.json
```

> [!NOTE]
> **Faithfulness Evaluation & Judge Architectures**:
> * **NLI CrossEncoder Judge (Default)**: Uses a cross-encoder NLI model (`--nli-model`, default `cross-encoder/nli-deberta-v3-base`; lighter alternative: `cross-encoder/nli-deberta-v3-small`). The generated answer is split into sentences (ignoring citation and timing footers). To resolve premise dilution, chunks are segmented into sentences and evaluated across sliding premise windows (sizes 1, 2, and 3 consecutive sentences, `--premise-window "1,2,3"`). For each sentence, entailment probabilities are evaluated against all premise windows via batched softmax over the model's dynamically resolved label config. A sentence is marked supported if `max entailment >= --entail-threshold` AND `entailment > contradiction` for the best window. Factual faithfulness is computed as `supported_sentences / total_sentences`.
> * **LLM Judge (`--judge-type llm`)**: Prompts an instruction-tuned model (e.g. `llama3.2:3b`) to extract factual claims and assess entailment as structured JSON. TeleRAG warns if the generator and judge models match to alert against self-enhancement bias.
> * **Unscorable Handling**: Any answer where the judge extracts 0 checkable claims or 0 sentences is classified as `UNSCORABLE` (never scored 1.00 or 0.00) and excluded from faithfulness means.
> * **Judge Validation Rule (`--validate-judge`)**: Before trusting a judge, it is benchmarked against `eval/fixtures/ci_answers.json` (DEV set: 12 reference answers and 6 negative controls) and `eval/fixtures/ci_answers_heldout.json` (HELD-OUT set: 8 reference answers and 8 negative controls). Threshold calibration sweeps 0.30 to 0.90 on DEV only, selecting the optimal threshold with the largest pass margin. The held-out set is evaluated strictly once at that fixed threshold. The judge is marked `VALID` only if BOTH sets pass:
>   1. Mean reference answer faithfulness is $\ge 0.80$.
>   2. Every negative control scores $\le 0.50$ (false-accept rate = 0%).
>   3. Zero unscorable references (coverage requirement).
>   Otherwise, the runner outputs `JUDGE NOT TRUSTWORTHY` and flags the judge as `UNVALIDATED JUDGE: treat scores as directional`. Validation results are saved to `eval/results/judge_validation_<judge>.json`.

### Faithfulness Evaluation

#### Think Python 2e Benchmark Results

Empirical faithfulness evaluation was conducted on 20 answerable questions sampled from *Think Python 2e* (`eval/fixtures/book.pdf`, saved in `eval/results/faithfulness.json`):

| Parameter / Metric | Measured Value |
| :--- | :--- |
| Sample Size ($n$) | 20 answerable questions |
| Random Seed | 42 |
| Generator Model | `qwen2.5:3b` |
| Prompt Version | `v1` |
| Judge Model | `cross-encoder/nli-deberta-v3-base` (premise windows 1, 2, 3; threshold 0.30) |
| Mean Answer Faithfulness | **0.535** (53.5%) |
| Unscorable Answers | 0 |
| Judge Validation Verdict | **`JUDGE NOT TRUSTWORTHY`** (`UNVALIDATED JUDGE: treat scores as directional`) |

#### Judge Error Rates & Validation Status

The default judge model (`cross-encoder/nli-deberta-v3-base`) was benchmarked using `--validate-judge` against ground-truth reference answers and hand-written negative controls on both DEV and HELD-OUT sets (saved in `eval/results/judge_validation_cross-encoder_nli-deberta-v3-base.json`):

* **DEV Set Error Rates**:
  * **Reference Accept Rate**: **9 / 12 (75.0%)** (mean faithfulness: 0.750; failed the $\ge 0.800$ passing requirement).
  * **Negative-Control False-Accept Rate**: **0 / 6 (0.0%)** (mean faithfulness: 0.111, max negative control score: 0.333).
* **HELD-OUT Set Error Rates**:
  * **Reference Accept Rate**: **7 / 8 (87.5%)** (mean faithfulness: 0.875; passed $\ge 0.800$).
  * **Negative-Control False-Accept Rate**: **0 / 8 (0.0%)** (mean faithfulness: 0.250, max negative control score: 0.500).

Because the judge achieved a DEV reference accept rate of 75.0% (falling short of the required 80.0%), the judge did **NOT** pass the validation gate and is explicitly labelled **`UNVALIDATED JUDGE: treat scores as directional`** (`JUDGE NOT TRUSTWORTHY`). Reported faithfulness scores must be treated as conservative and directional.

#### Interpreting the 0.535 Score (Not a Hallucination Rate)

The measured mean answer faithfulness of **0.535** combines three distinct sources of unsupported sentences and **must NOT be interpreted as a 46.5% hallucination rate**:
1. **Strict Judge Over-Rejection**: The NLI cross-encoder model applies strict premise-sentence entailment and frequently assigns `neutral` to factually correct claims when phrasing deviates from the text or requires implicit bridging commonsense knowledge. This is demonstrated by the judge's 25.0% error rate on human-verified ground truth reference answers in the DEV benchmark (accepting only 9 of 12 references).
2. **Generator Unsupported Content**: The lightweight 3B parameter local generator (`qwen2.5:3b`) occasionally includes introductory framing, high-level summaries, or general programming explanations that, while plausible, are not directly asserted within the retrieved passage.
3. **Retrieval Misses**: Context retrieval is limited to the top-2 chunks (`TOP_K_RESULTS=2`). When a multi-faceted question requires context across multiple pages or when hybrid retrieval omits a supporting sentence, even an accurate generation cannot be entailed from the retrieved snippets alone (e.g. `ci_06`, where retrieval missed the key passage, resulting in a 0.00 score on a true reference answer).

#### Dataset Item Counts & Evaluation Filtering

* **DEV Set (`eval/fixtures/ci_answers.json`)**: 12 reference answers and 6 negative controls (18 scored items).
  * *Why the validation log displayed 21 items but only 18 were scored*: `ci_answers.json` contains 21 total fixture entries: 12 reference answers, 6 negative controls, and 3 unanswerable questions (`ci_u01`–`ci_u03`) where the expected answer is `"INSUFFICIENT_CONTEXT"`. In `eval/run_faithfulness.py`, `if ans_text.strip() == "INSUFFICIENT_CONTEXT": continue` bypasses unanswerable refusal items from NLI claim verification, leaving exactly 18 scored items.
* **HELD-OUT Set (`eval/fixtures/ci_answers_heldout.json`)**: 8 reference answers and 8 negative controls (16 scored items).

#### Judge Validation Limitations

1. **Whole-Answer vs Sentence-Level False-Accept Rate**: The negative-control false-accept rate is calculated at the whole-answer level (requiring aggregate answer faithfulness $> 0.50$ to register as a false accept). Partially supported negative controls (where 1 of 3 sentences or 1 of 2 sentences happens to be true) score 0.333 or 0.500; they count as non-false-accepts under the $\le 0.50$ rule despite individual sentences being classified as entailed.
2. **Boundary Sensitivity on Held-Out Negative Controls**: 4 of the 8 HELD-OUT negative controls (`heldout_neg_05`, `heldout_neg_06`, `heldout_neg_07`, and `heldout_neg_08`) scored exactly **0.500** (1 of 2 sentences supported). Consequently, the pass status on the held-out set strictly depends on the non-strict `<=` boundary criterion (`score <= 0.50`); a strict `< 0.50` rule would have failed all 4 items and resulted in a FAIL verdict on held-out.
3. **Post-Hoc Model Selection**: The `cross-encoder/nli-deberta-v3-base` judge model was selected post-hoc after the smaller `cross-encoder/nli-deberta-v3-small` failed the held-out evaluation due to entity-swap vulnerability on `heldout_neg_02`. Because model selection was guided by performance on the held-out set, the held-out validation is not fully independent.

### Judge Validation Findings

Validation scores reference answers against **retrieved context** (not raw ground-truth fixture text). Consequently, a retrieval miss counts against the reference. For example, on `ci_06` (*What RPCs are used by the Raft leader to replicate log entries across followers?*), the true reference answer was scored `0.00` across all judges because hybrid retrieval's top-2 chunks did not contain the sentence explaining `AppendEntries`, demonstrating that end-to-end judge validation tests the joint retrieval-and-judge pipeline.

#### Summary of Evaluated Judges

| Judge Architecture | Model | Premise Granularity | DEV Ref Mean | DEV Neg Max (False Accept) | HELD-OUT Ref Mean | HELD-OUT Neg Max (False Accept) | Validation Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LLM Judge | `llama3.2:3b` | Whole retrieved context | 0.917 | 1.000 (16.7%) | n/a | n/a | **`NOT TRUSTWORTHY`** |
| NLI CrossEncoder | `nli-deberta-v3-small` | Whole chunk | 0.167 | 0.333 (0.0%) | n/a | n/a | **`NOT TRUSTWORTHY`** |
| NLI CrossEncoder | `nli-deberta-v3-small` | Sentence windows (1,2,3) | 0.917 | 0.333 (0.0%) | 0.875 | 1.000 (12.5%) | **`NOT TRUSTWORTHY`** |
| NLI CrossEncoder (Default) | `nli-deberta-v3-base` | Sentence windows (1,2,3) | 0.750 | 0.333 (0.0%) | 0.875 | 0.500 (0.0%) | **`NOT TRUSTWORTHY`** |

#### Detailed Judge Analysis

* **`llama3.2:3b` (LLM Judge)**:
  * DEV Set: Mean Ref **0.917**, Mean Neg **0.222**, Max Neg **1.000** (`neg_06`, a subtly incorrect assertion, scored 2/2 entailed; false-accept rate 16.7%).
  * Coverage Failure: 4 of 12 reference answers (`ci_03`, `ci_04`, `ci_06`, `ci_08`) yielded zero extracted claims and were erroneously scored 1.00 before the zero-claim bug fix.
  * Verdict: **`JUDGE NOT TRUSTWORTHY`** (failed negative control ceiling and reference coverage).

* **`cross-encoder/nli-deberta-v3-small` (Whole Chunks)**:
  * DEV Set: Mean Ref **0.167** (failed $\ge 0.800$ requirement), Mean Neg **0.056**, Max Neg **0.333** (false-accept rate 0.0%).
  * Root Cause: Premise granularity mismatch. Passing multi-topic 200-word chunks as premises to a small MNLI model diluted attention, producing ~0.98 `neutral` probabilities even for verbatim claims.
  * Verdict: **`JUDGE NOT TRUSTWORTHY`**.

* **`cross-encoder/nli-deberta-v3-small` (Sentence Windows `1,2,3`, Calibrated Threshold `0.35`)**:
  * DEV Set: Mean Ref **0.917** (11/12 passed, accept rate 91.7%), Mean Neg **0.167**, Max Neg **0.333** (false-accept rate 0.0%, fully-accepted rate 0.0%). DEV Set Status: **PASS**.
  * HELD-OUT Set: Mean Ref **0.875** (7/8 passed, accept rate 87.5%), Mean Neg **0.375**, Max Neg **1.000** (false-accept rate 12.5%, fully-accepted rate 12.5%). HELD-OUT Set Status: **FAIL**.
  * Root Cause: Entity-swap blindness on `heldout_neg_02`.
  * Verdict: **`JUDGE NOT TRUSTWORTHY`**.

* **`cross-encoder/nli-deberta-v3-base` (Sentence Windows `1,2,3`, Calibrated Threshold `0.30`, Default)**:
  * DEV Set: Mean Ref **0.750** (9/12 passed, accept rate 75.0%; `ci_04`, `ci_06`, `ci_12` under-scored), Mean Neg **0.111**, Max Neg **0.333** (false-accept rate 0.0%, fully-accepted rate 0.0%). DEV Set Status: **FAIL**.
  * HELD-OUT Set: Mean Ref **0.875** (7/8 passed, accept rate 87.5%), Mean Neg **0.250**, Max Neg **0.500** (false-accept rate **0.0%**, fully-accepted rate **0.0%**). HELD-OUT Set Status: **PASS**.
  * Key Advantage: Unlike the small model, `base` was not fooled by `heldout_neg_02` (correctly rejecting it with score 0.00), achieving 0.0% false-accept rate across all 14 negative controls on both sets and passing the held-out set. However, because it under-scored on DEV (75.0% < 80.0%), it remains an unvalidated judge whose reports are explicitly flagged as directional.
  * Verdict: **`JUDGE NOT TRUSTWORTHY`**.

---

## Security

* **SSRF Protection**: URL ingestion validates schemes (`http`/`https`) and resolves target IP addresses via `socket.getaddrinfo()` to block private RFC1918 subnets, loopback (`127.0.0.1`), and cloud metadata (`169.254.169.254`). Each hop in a redirect chain (up to 3 hops) is validated separately with `allow_redirects=False`.
* **Streaming Byte Limits**: Downloads are consumed in 8 KB streaming chunks. Sockets are aborted immediately if transferred bytes exceed `MAX_FILE_SIZE_MB`, guarding against unbounded downloads where `Content-Length` is omitted.
* **Per-User Scoping**: All database queries filter on `WHERE documents.user_id = ?` to partition document and chunk access by Telegram user.
* **Cascade Deletion**: SQLite enforces `PRAGMA foreign_keys = ON` with `ON DELETE CASCADE`. Removing a document cleanly removes its associated chunk metadata and vector files.

---

## Limitations

1. **Hallucination Reduction, Not Elimination**: The relevance threshold ($\tau = 0.3$) and prompt instructions reduce unsupported answers, but an LLM can still misinterpret retrieved context or hallucinate nuances.
2. **Hard Negative Abstention**: While bi-encoder cosine gating stops 100% of off-topic queries, it does not stop in-domain unanswerable queries (e.g., asking about `asyncio` on a basic Python textbook), where chunks score 0.35–0.48. The cross-encoder threshold experiment rejected 2/5 on test, but was tuned on a small sample.
3. **Hardware-Dependent Generation Latency & Host Uptime**: Reranking improves Hit@1 by +10.0 pp but adds ~2s on CPU. Answer generation time depends heavily on model choice and available hardware; on a laptop CPU without GPU acceleration, generation with a 7B parameter model is measured in minutes. TeleRAG caps completion tokens (default `LLM_MAX_TOKENS=300`), prompts for concise answers, limits context chunks (`TOP_K_RESULTS=2`), serializes requests via an `asyncio.Semaphore(1)` with queue notifications, and streams typing heartbeats. Furthermore, because it runs locally, the bot is only active and reachable while the host machine is awake and running `python main.py`.
4. **Free Hosted Model Lifecycle Volatility**: Third-party free endpoints on hosted providers frequently rotate, become rate-limited, or transition to paid slugs (e.g. returning HTTP 404/429), or can be retired entirely. The startup diagnostic catches unavailable models early.
5. **Document-Level Summaries**: For general summary/overview queries, the bot samples the first ~8 chunks in document page order rather than performing hierarchical map-reduce summarization across very long books.
6. **SSRF DNS Rebinding**: URL validation checks resolved IPs before request dispatch, but standard `requests.get()` does not pin resolved IP addresses to the socket connection, leaving a theoretical window for DNS rebinding attacks.
7. **Text-Only PDF Extraction**: Scanned documents or image-only PDFs without an embedded OCR text stream cannot be extracted by `PyPDF2`.
8. **Local Single-Host Storage**: SQLite and JSON embedding files reside unencrypted on the local filesystem, suitable for personal or small-team use rather than untrusted multi-tenant host environments.
9. **NLI Entity-Swap Blind Spot**: Smaller cross-encoder NLI models (e.g. `cross-encoder/nli-deberta-v3-small`) can suffer from bag-of-words / entity co-occurrence bias. For example, on `heldout_neg_02` (*Which database systems favor availability and low latency according to the PACELC classification?*), the subtly inverted answer `Spanner and Bigtable prioritize PA/EL, choosing availability and low latency over consistency.` was assigned entailment `0.7545` (scored 1.000) against the PACELC premise window because all mentioned entities (Spanner, Bigtable, PA/EL, availability, low latency) co-occur in the same sentence window even though the passage assigns PA/EL to Dynamo/Cassandra and PC/EC to Bigtable/Spanner. Testing the larger `cross-encoder/nli-deberta-v3-base` resolved this specific entity-swap error (rejecting `heldout_neg_02` with score 0.00 and 0.0% false-accept rate on held-out), but under-scored certain dev references (`ci_04`, `ci_12`), leaving both NLI models marked as unvalidated judges.
10. **Faithfulness Evaluation & Judge Validation Caveats**: The Think Python 2e faithfulness score (mean 0.535) is directional and conservative. It conflates strict NLI judge rejections (DEV reference accept rate 75.0%), 3B generator unsupported phrasing, and top-2 retrieval context misses, and must not be interpreted as a hallucination rate. Furthermore, the judge's 0.0% negative-control false-accept rate is whole-answer level (> 0.50), half of the held-out negative controls scored right at the boundary (0.500), and the base judge was selected post-hoc.

---

## Project Structure

```
TeleRAG/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI quality gate
├── eval/
│   ├── fixtures/
│   │   ├── ci_fixture.pdf      # Synthetic 8-page regression document
│   │   ├── ci_answers.json     # Reference answers & negative controls for grounding gate
│   │   ├── build_golden.py     # Golden dataset generator
│   │   ├── make_fixture.py     # CI fixture builder
│   │   └── make_split.py       # Stratified split generator
│   ├── golden.json             # 70-question verified golden dataset (Think Python 2e)
│   ├── golden_ci.json          # 15-question synthetic CI golden dataset
│   ├── split.json              # Stratified DEV/TEST split (seed 42)
│   ├── thresholds.json         # Regression thresholds for CI quality & grounding gates
│   ├── run_eval.py             # Retrieval evaluation harness (dense, hybrid, hybrid_rerank)
│   ├── run_grounding_gate.py   # CI factual citation coverage & hallucination check
│   ├── run_faithfulness.py     # Local LLM-as-judge claim entailment evaluation
│   ├── run_abstention_experiment.py # Dev/Test abstention gate experiment
│   ├── bench_llm.py            # Local model latency and refusal benchmark
│   └── results/                # Measured benchmark outputs (JSON)
│       ├── baseline.json       # Dense baseline metrics
│       ├── hybrid.json         # BM25 + Dense RRF metrics
│       ├── hybrid_rerank.json  # Cross-encoder reranked metrics
│       ├── ci_eval.json        # Synthetic CI fixture metrics
│       ├── llm_bench.json      # Measured local LLM latency & refusal metrics
│       └── abstention_experiment.json # Stratified dev/test gate comparison
├── prompts/
│   └── v1.yaml                 # Versioned prompt configuration (system, user, refusal)
├── docs/
│   └── design-decisions.md     # Architecture decisions and trade-offs log
├── main.py                     # Bot entry point and lifecycle manager
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Development & CI fixture dependencies (fpdf2)
├── requirements-eval.txt       # Optional local evaluation dependencies (pydantic, ragas)
├── .env.example                # Configuration template
├── .gitignore                  # Git exclusion rules
├── INSTRUCTIONS.md             # Complete installation, configuration & user manual
├── LICENSE                     # Standard MIT License
├── README.md                   # Repository features & empirical evaluation
├── TECHNICAL_REPORT.md         # Detailed technical engineering report
│
├── src/
│   ├── config.py               # Central environment variable configuration
│   ├── bot/
│   │   ├── handlers.py         # Telegram handlers, queueing, grounding check & footers
│   │   └── __init__.py
│   ├── processors/
│   │   ├── pdf_processor.py    # Page-aware extraction, validation
│   │   ├── text_chunker.py     # Sliding-window chunker with page bounds
│   │   └── __init__.py
│   ├── rag/
│   │   ├── embeddings.py       # SentenceTransformers embedding generation
│   │   ├── grounding.py        # Post-generation sentence grounding report
│   │   ├── hybrid.py           # BM25 + dense Reciprocal Rank Fusion
│   │   ├── llm_client.py       # OpenAI-compatible LLM client & response parsing
│   │   ├── prompts.py          # Versioned YAML prompt loader
│   │   ├── reranker.py         # CrossEncoder candidate reranking
│   │   ├── retriever.py        # Routing retriever & page citation context formatting
│   │   └── __init__.py
│   └── storage/
│       ├── database.py         # SQLite schema, page columns & migrations
│       ├── document_store.py   # Document ingestion, vector storage & retrieval
│       └── __init__.py
│
└── tests/                      # Automated unit test suite
    ├── test_bench_llm.py       # Tests for benchmark loop, abstention & recovery
    ├── test_citations.py       # Tests for page citations & source footers
    ├── test_faithfulness.py    # Tests for LLM-as-judge claim verification
    ├── test_grounding.py       # Tests for sentence grounding & footer stripping
    ├── test_handlers.py        # Tests for SSRF defense, redirects, queueing & answers
    ├── test_hybrid.py          # Tests for BM25, RRF fusion, and abstention invariant
    ├── test_pdf_processor.py   # Tests for extraction, chunking, and page boundaries
    ├── test_prompts.py         # Tests for YAML prompt loading & validation
    ├── test_rag.py             # Tests for embeddings, cosine search, 404 handling, thresholds
    ├── test_reranker.py        # Tests for cross-encoder reranking order and scores
    └── test_storage.py         # Tests for database CRUD, page migrations, and cascades
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
