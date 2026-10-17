# TeleRAG: Architecture Decisions & Trade-Offs

A summary of technical choices, empirical trade-offs, and design rationales made while building TeleRAG.

---

## 1. Local Dense Embeddings vs. External Vector DB

* **Decision**: Generate 384-dimensional dense vectors locally using `sentence-transformers/all-MiniLM-L6-v2` and compute pairwise cosine similarity in-memory using pure NumPy.
* **Why**: Managed vector databases (Pinecone, Milvus, Qdrant) add network hops, hosting costs, and credential management that are unnecessary for personal and small-group document collections. For libraries under 5,000 chunks, in-memory dot products take $< 5\text{ ms}$ on CPU.
* **Trade-off**: Pairwise linear search scales $\mathcal{O}(N \cdot D)$. If corpus sizes exceed tens of thousands of chunks, an approximate nearest neighbor index (e.g., `sqlite-vec` or FAISS) will be required.

---

## 2. Page-Aware Chunking & Boundary Intervals

* **Decision**: Extract text page-by-page from PDFs and track `[page_start, page_end]` intervals for every sliding-window text chunk (1,000 characters, 200 overlap), rather than naively assigning the start page.
* **Why**: In baseline evaluation error analysis, 5 out of 7 initial top-5 misses were off by exactly one page because the retrieved chunk started near the bottom of page $P$ and continued onto page $P+1$ where the target concept resided. Tracking intervals eliminated false negatives in evaluation and provides accurate page citations.
* **Trade-off**: Requires database schema migrations to store start and end page columns alongside backwards-compatible fallbacks.

---

## 3. Sparse-Dense Hybrid Fusion (BM25 + RRF)

* **Decision**: Combine BM25 lexical search (`rank-bm25`) with dense cosine retrieval using Reciprocal Rank Fusion ($k=60$).
* **Why**: Pure semantic search struggles on exact programming symbols, syntax keywords, and function identifiers (e.g. `tuple assignment`, `in operator`). BM25 brings exact lexical matching. RRF merges rank positions without requiring score normalization across disparate scales.
* **Reality check**: On a 50-question answerable evaluation set, hybrid retrieval improves top-1 hit by 2 questions over dense baseline (70.0% vs. 66.0%), which is within noise. The main architectural value of hybrid search is robustness across both keyword-heavy queries and conceptual paraphrases.

---

## 4. Cross-Encoder Reranking

* **Decision**: Support an optional two-stage pipeline where top-20 candidates from hybrid retrieval are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Why**: Joint attention across query and passage tokens significantly improves ranking accuracy over independent bi-encoder representations (Hit@1 increases from 66.0% to 76.0%, and MRR increases from 0.79 to 0.85).
* **Trade-off**: Latency. Cross-encoder scoring takes ~2 seconds per query on CPU versus ~57 ms for dense or hybrid retrieval. For real-time chat, hybrid without reranking is the faster default, but reranking is available when precision is prioritized (`RETRIEVAL_MODE=hybrid_rerank`).

---

## 5. Abstention Gating & Hard Negative Handling

* **Decision**: Keep dense cosine similarity thresholding ($\tau = 0.3$) as the default gatekeeper before calling the LLM.
* **Why**: Easy off-topic queries (cooking, sports) score $< 0.22$ and are filtered with 100% accuracy, saving API tokens and preventing unrelated answers.
* **Observation & Experiment**: Dense cosine similarity fails on hard unanswerable queries within the broader topic (e.g., asking about `asyncio` or `dataclass` on a beginner Python textbook that does not cover them). These queries score $0.35$–$0.48$ due to topic similarity and pass the cosine gate.
* In a stratified dev/test experiment ($N=35$ DEV, $N=35$ TEST), tuning a cross-encoder score gate on DEV ($T = -2.80$) rejected 2/5 hard negatives on the unseen TEST set without hurting answerable recall (25/25 retained). However, the threshold was tuned on a small sample and may be overfit. I kept the dense cosine gate as the default and documented this limitation honestly.

---

## 6. Defensive URL Ingestion & Multi-Hop SSRF Protection

* **Decision**: Follow HTTP redirects manually (max 3 hops) with `allow_redirects=False`. Validate URL scheme and resolve target IP addresses via `socket.getaddrinfo()` at every hop before opening connections. Stream downloads with a strict byte counter.
* **Why**: Standard `requests.get(url)` follows redirects automatically, allowing an attacker to supply a public URL that redirects to `127.0.0.1` or `169.254.169.254` (cloud metadata). Streaming byte enforcement prevents memory exhaustion via chunked transfer encoding where `Content-Length` is omitted or spoofed.
* **Trade-off**: Adds slight latency to URL downloads due to manual hop inspection, and does not defend against time-of-check-to-time-of-use DNS rebinding between resolution and fetch.
