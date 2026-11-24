# Interview notes

These notes are for explaining TeleRAG in an interview. For each main part they give the decision, the alternatives that were considered, the trade-offs, and questions you are likely to get, with answers in plain words. The numbers come from the files in `eval/results/`. If you quote a number, say how it was measured (one book, 50 questions, one laptop).

**How to use this page.** Do not memorize the answers. Understand the reason behind each decision, then say it in your own words. Interviewers care most that you know the weaknesses of your own project. Each section says where the weak spots are.

## Contents

1. [The 60-second pitch](#the-60-second-pitch)
2. [Numbers worth remembering](#numbers-worth-remembering)
3. [The cross-user leak bug](#the-cross-user-leak-bug)
4. [Chunking](#1-chunking)
5. [Embeddings and storage](#2-embeddings-and-storage)
6. [Hybrid retrieval: dense search, BM25 and RRF](#3-hybrid-retrieval-dense-search-bm25-and-rrf)
7. [Reranking](#4-reranking)
8. [Refusing questions that are not in the documents](#5-refusing-questions-that-are-not-in-the-documents)
9. [Prompt and refusal handling](#6-prompt-and-refusal-handling)
10. [The grounding check](#7-the-grounding-check)
11. [The LLM client: timeouts and retries](#8-the-llm-client-timeouts-and-retries)
12. [Concurrency](#9-concurrency)
13. [Separating users](#10-separating-users)
14. [Access control and rate limits](#11-access-control-and-rate-limits)
15. [Safe downloads (SSRF)](#12-safe-downloads-ssrf)
16. [Configuration and startup checks](#13-configuration-and-startup-checks)
17. [Logging](#14-logging)
18. [Docker](#15-docker)
19. [Evaluation](#16-evaluation)
20. [Testing](#17-testing)
21. [What I would do next, and how to talk about weaknesses](#what-i-would-do-next-and-how-to-talk-about-weaknesses)

## The 60-second pitch

"TeleRAG is a Telegram bot that answers questions about documents you send it. Under the hood it is retrieval-augmented generation: it cuts documents into chunks, finds the chunks closest to the question, and gives only those to a small language model that runs on my laptop CPU. It cites the pages it used and refuses when the answer is not in the documents. I measured it: on a 240-page book the right page is in the top 5 results 96 to 100 percent of the time. I also measured where it is weak: it does not refuse questions that are on the same topic but not in the book. I also fixed a real bug where one user could see another user's text, and I added a test so it cannot come back."

## Numbers worth remembering

| Fact | Value | Where it comes from |
| :--- | :--- | :--- |
| Retrieval test set | 50 answerable and 20 unanswerable questions on one book (582 chunks) | `eval/golden.json`, `eval/results/` |
| Hit@1 dense / hybrid / hybrid + reranker | 66% / 70% / 76% | `eval/results/baseline.json`, `hybrid.json`, `hybrid_rerank.json` |
| Hit@5 | 100% / 96% / 98% | same files |
| Search time | 57 ms, 58 ms, 2,021 ms (with reranker) | `mean_retrieval_ms` in the same files |
| Off-topic questions refused | 10 of 10 | same files |
| In-topic questions not in the book refused | 0 of 10 | same files |
| LLM answer time on the CI fixture | median 13.8 s, 95th percentile 21.1 s (`qwen2.5:3b`) | `eval/results/llm_bench.json` |
| Hand-checked answers | 8 correct, 3 partly, 4 wrong or empty, 1 unchecked (16 questions) | `docs/evaluation.md` |
| Tests | more than 350 | `python -m unittest discover tests` |

## The cross-user leak bug

This is the best story in the project, because it shows how a real bug happens, how it was found, and how it was prevented from coming back. Tell it honestly: it was found by reading code, I have no evidence it ever hit a real user, and the cause was a design mistake, not a typo.

**What was wrong.** Search used a long-lived `HybridRetriever` object. To avoid rebuilding the keyword (BM25) index for every question, the object stored the last index and the last list of chunks in its own fields (`self.bm25`, `self.chunks`). It also kept a cache of indexes per user. In `search()`, the code rebuilt the index only when the cache did not already contain the current user. When the cache did contain the user, it skipped the rebuild and then searched `self.chunks`, which still held the chunks of whoever had searched last.

**How it played out.**

1. User 1 asks a question. The retriever indexes user 1's chunks and caches them.
2. User 2 asks a question. The retriever indexes user 2's chunks. Now `self.chunks` belongs to user 2.
3. User 1 asks again. The cache says "user 1 is already indexed", so the rebuild is skipped, and the search runs over user 2's chunks. User 1 receives text from user 2's documents.

**A second problem made it worse.** The chunks read from the database had no `user_id` field and the bot never passed the user ID to the retriever. So every cache key had `None` where the user should be, and "invalidate the cache for this user" could never match anything. Nothing called it anyway.

**How it was found.** While checking the project before changing anything, I read `search()` and noticed that it reads shared fields after a conditional re-index. I then wrote a 15-line script that did the three steps above, and the third call printed the other user's text. That reproduction became the test.

**The fix.**

- `search()` no longer remembers any "current user". It works only with the chunks it is given and an index looked up by user and corpus. Nothing about one request can be left over for the next one.
- The cache is a small LRU (8 entries) guarded by a lock. The key contains the user ID and a hash of the chunk IDs and text.
- The pipeline now passes the user ID down to the retriever.
- When a user adds, deletes or clears documents, the document store tells the retriever to drop that user's cached index.

**How the test prevents it from coming back.** `tests/test_hybrid.py` has a class `TestUserIsolation`:

- `test_hybrid_search_does_not_leak_after_cache_hit` does the user 1, user 2, user 1 sequence and checks that the third answer is user 1's own text. It failed on the old code with `'bravo private note for user two' != 'alpha private note for user one'`.
- `test_document_retriever_does_not_leak_when_user_id_is_missing` does the same through the real entry point without a user ID, as the old bot called it.
- A threaded test runs two users in parallel with a cache of size 1, which forces constant eviction, and checks every single result.
- Other tests check that the cache is bounded, that invalidating one user leaves the others alone, and that the pipeline passes the user ID.
- `tests/test_bot_runtime.py` runs two users at the same time through the real handlers and a real database and checks that neither answer contains the other's text.

**Why the old tests missed it.** They used one user at a time. A bug that needs the pattern A, B, A cannot show up when only A ever talks.

**Lessons you can state.** (1) Shared mutable state in a long-lived object is dangerous in anything with more than one user. Prefer functions that work only on what they are given. (2) A cache key must contain everything that decides whose data it is. (3) Test the interleaving, not only the single path. (4) When two things go wrong at once (a stale field and a key that is always `None`), fixing only one hides the other.

**Likely questions**

- *How could this have happened?* "I optimized for speed by remembering the last index on the object, and I made the cache hit path skip the step that set that field. It worked in every single-user test."
- *Was any data really leaked?* "I found it by reading the code and reproducing it with a script. I have no evidence it happened to a real user. It needs two users to talk to the bot, and the bot was used by me."
- *How do you know it is fixed and will stay fixed?* "The test that reproduces it failed before and passes after. There is also a concurrent test, and a handler-level test with a real database. I also checked that the concurrency test can fail, by forcing blocking work onto the event loop and watching it fail."

## 1. Chunking

**Decision.** Cut text into pieces of about 1,000 characters that overlap by 200. Break at a sentence end if possible, then at a paragraph, then at a word. For PDFs, extract page by page and store the page range (`page_start`, `page_end`) of every chunk. Markdown is split at headings and each chunk is labeled with its heading path.

**Why.** The embedding model reads at most 256 word pieces. The chunk-size test showed that 1,500-character chunks (71% longer than the window) get cut off, while 1,000-character chunks (0% too long) fit. Overlap keeps a sentence that falls on a boundary visible in two chunks. The page range makes the page citations correct: in an early test, 5 of 7 misses were off by exactly one page because a chunk started at the bottom of one page and continued onto the next.

**Alternatives.** Smaller chunks (500) give more precise matches but double the number of chunks. In the small ablation, 500 and 1,000 characters tied on the "overlaps the right page" metric and 500 was better on the strict metric, so the honest reading is: 1,000 was kept as a reasonable default, not proven best. Other options are splitting by sentences or paragraphs only, or using a model-based splitter.

**Trade-offs.** Character-based chunks ignore meaning: a table or a code block can be cut in the middle. Chunk size is a tuning knob that I only tested on a tiny fixture.

**Questions**

- *Why overlap?* "So an answer that sits across a boundary still appears whole in at least one chunk."
- *Why track page ranges instead of the start page?* "A chunk can cover two pages. Using only the start page made correct retrievals look like misses and gave wrong citations."
- *How did you pick 1,000?* "It fits the embedding model's window and gives fewer chunks than 500. My ablation was small and the retrieval scores tied, so I would re-check it on a bigger set before claiming it is best."

## 2. Embeddings and storage

**Decision.** Use `all-MiniLM-L6-v2` (384 numbers per chunk) through `sentence-transformers`, running on the CPU. Compute cosine similarity with NumPy over all of a user's chunks. Store metadata in SQLite and the vectors of each document in one `.npy` file.

**Why.** It is small (about 90 MB), fast on a CPU, needs no server and no API key. For a personal library of a few thousand chunks, comparing a question with every chunk takes milliseconds (57 ms for the whole search on 582 chunks, including BM25).

**Alternatives.** A vector database (Chroma, Qdrant, Pinecone) or an index library (FAISS). A larger embedding model would be more accurate but slower and bigger. There is an optional Chroma backend in the code behind `VECTOR_BACKEND=chroma`.

**Trade-offs.** Brute-force search grows linearly with the number of chunks. At tens of thousands of chunks per user it needs an approximate index. Loading the `.npy` files for each question is simple but wasteful. SQLite is a single file with no setup, but writes from many processes would contend, and the data is not encrypted.

**Questions**

- *What is an embedding?* "A list of numbers that represents the meaning of a text, so that texts with similar meaning have vectors that point in similar directions."
- *What is cosine similarity?* "The angle between two vectors: 1 means the same direction, 0 means unrelated."
- *Why not a vector database?* "Not needed at this size, and it would add a server to run and secure. If a user had hundreds of thousands of chunks I would add one."

## 3. Hybrid retrieval: dense search, BM25 and RRF

**Decision.** Run two searches and merge them. Dense search finds chunks with similar meaning. BM25 finds chunks that contain the same words. Reciprocal Rank Fusion (RRF) merges the two rankings: a chunk's score is the sum of `1 / (60 + rank)` over the lists it appears in.

**Why.** Dense search can miss exact terms such as function names, and BM25 misses paraphrases. RRF needs no score normalization, which matters because the two scores are on different scales.

**Alternatives.** Dense only (simpler), BM25 only, a weighted sum of normalized scores (needs tuning).

**What the measurements say, honestly.** On 50 questions, hybrid got Hit@1 70% against 66% for dense only, and Hit@5 96% against 100%. That is a difference of one or two questions, which is within noise. I keep hybrid as the default because it should help on keyword-heavy questions, but I did not prove a gain. The clear improvement came from the reranker.

**Trade-offs.** More code and a BM25 index to build and cache. The index is cached per user in an LRU of 8 and dropped when the user's documents change. Both are the places where the leak bug lived.

**Questions**

- *What is BM25?* "A classic keyword ranking function. It scores a chunk higher when it contains the query words, more so for rare words, and it corrects for chunk length."
- *Why RRF and not just add the scores?* "The scores have different scales. Ranks can be compared directly, and the constant 60 stops the top few positions from dominating."
- *Did hybrid search help?* "Not measurably on my test set: one or two questions out of 50. I say so in the README."

## 4. Reranking

**Decision.** Optional second stage (`RETRIEVAL_MODE=hybrid_rerank`): take the top 20 chunks from hybrid search and re-score them with a cross-encoder (`ms-marco-MiniLM-L-6-v2`), then keep the best few.

**Why.** A bi-encoder embeds the question and each chunk separately. A cross-encoder reads the question and chunk together, so it is more accurate but too slow to run on every chunk. Running it only on 20 candidates is a common compromise. In my test, Hit@1 went from 66% to 76% and MRR from 0.79 to 0.85.

**Trade-offs.** About 2 seconds more per question on this CPU (2,021 ms against 58 ms), so it is off by default. A second model to load, with a lock around it because several users run in parallel threads.

**Questions**

- *Bi-encoder versus cross-encoder?* "Bi-encoder: embed separately, compare vectors, fast and good for finding candidates. Cross-encoder: read both texts together, slower and better for ordering a short list."
- *Why is it not the default?* "The extra 2 seconds on a CPU, in a chat bot where the language model already takes 10 to 20 seconds. I made it a setting."
- *What is MRR?* "Mean reciprocal rank: the average of 1 divided by the position of the first correct result. 1.0 means always first."

## 5. Refusing questions that are not in the documents

**Decision.** Before calling the LLM, look at the best cosine similarity between the question and any chunk. If it is below 0.3, refuse and do not call the model.

**Why.** It saves 10 or more seconds of CPU time and avoids an answer made up from unrelated text. Off-topic questions score below 0.22 in my test, so all 10 were refused.

**The weakness, stated plainly.** Questions on the same subject that the book does not cover (for example `asyncio` in a beginner Python book) score 0.35 to 0.48, above the threshold. None of my 10 such questions was refused. I tried a cross-encoder score gate tuned on half of the data; on the other half it refused 2 of 5 without losing any answerable question, but 5 questions are too few to trust, so I did not switch it on.

**Alternatives.** A higher threshold (loses real questions), the cross-encoder gate, or asking the LLM to decide (slow and unreliable with a 3B model; the prompt does this as a second line of defense).

**Questions**

- *How did you choose 0.3?* "I looked at the scores of answerable and off-topic questions. It is a compromise, not a tuned optimum, and the evaluation shows it cannot separate in-topic questions."
- *What happens with a question that is on topic but not covered?* "The model gets the closest chunks and is told to say INSUFFICIENT_CONTEXT if the text does not contain the answer. That sometimes works. My evaluation counts it as not refused at the retrieval step."

## 6. Prompt and refusal handling

**Decision.** The system prompt (versioned in `prompts/v1.yaml`) tells the model to answer only from the context, cite `[file, p. N]`, stay under 150 words and answer exactly `INSUFFICIENT_CONTEXT` when the context does not contain the answer. The code then reads the reply sentence by sentence.

**Why sentence by sentence.** Earlier code refused the whole answer if a phrase like "the context does not mention" appeared anywhere. That threw away useful partial answers such as "The fee is five dollars. The context does not mention refunds." Now:

- only refusal sentences: refuse;
- real sentences plus a refusal sentence: keep the answer, hide the raw marker, but only if at least one real sentence is supported by the retrieved text (the grounding check decides);
- fewer than 3 words: retry once.

I found one more gap by running the real model: it often says "the **document** does not provide..." instead of "the **context** does not...". Those phrases are now covered too, with a test.

**Summary requests.** "Summarize this document" is detected with strict patterns (the whole question must be a summary request). Before, any question containing "summary" skipped retrieval, so "What is the summary of chapter 3?" got the first 8 chunks of the book. Now a real summary request uses 8 chunks spread evenly over the whole document.

**Trade-offs.** Matching refusal phrases is fragile: a model can word a refusal in a way the list does not know. That is why the grounding check is the second safeguard. Prompt versions let me change the prompt without changing code, and compare versions.

**Questions**

- *Why a marker word instead of free text?* "A fixed token is easy to detect. I still handle paraphrases because small models do not always follow the instruction exactly."
- *What is prompt injection and how do you handle it?* "A document can contain instructions that try to steer the model. I tell the model to use the text only as information, and two injection attempts in my manual test were ignored, but that is not a guarantee."

## 7. The grounding check

**Decision.** After the answer, split it into sentences, embed each one, and compute the cosine similarity to each retrieved chunk. A sentence is "supported" if its best similarity is at least 0.45. The Sources footer then lists only chunks that supported something, and an answer where nothing is supported gets a note "Could not verify this answer against the cited pages" instead of sources.

**Why this and not something stronger.** It reuses the embedding model that is already loaded, so it costs milliseconds and no extra download.

**What it cannot do.** Similarity is not truth. A wrong sentence about the same topic can score high, and a correct sentence in different words can score low. I say clearly that it reduces unsupported answers but does not remove them.

**Alternatives I tried.** An NLI (entailment) model as a judge in the offline evaluation script. I validated each judge against known-good and known-bad answers. The default judge rejected 3 of 12 correct reference answers on the development set (75% accepted). The smaller NLI judge and a 3B LLM judge each accepted a subtly wrong answer. None passed, so the script marks its own verdict "NOT TRUSTWORTHY". The mean score of 0.535 from that script is therefore not a hallucination rate. I did not put an NLI model in the bot because of this and because of the extra CPU time.

**Questions**

- *How do you know the answer is not made up?* "I cannot be sure. I reduce the risk: refuse when retrieval is weak, instruct the model, verify each sentence against the chunks, and show sources so the user can check."
- *Why is your faithfulness number only 0.535?* "It mixes a strict and imperfect judge, a small generator model and top-2 retrieval misses. My own validation says the judge is not trustworthy, so it is a trend, not a rate."
- *Why not use an LLM as the judge?* "I tried one. A 3B model as a judge was also unreliable, and it would double the response time."

## 8. The LLM client: timeouts and retries

**Decision.** A small client for any OpenAI-compatible `/chat/completions` endpoint (Ollama by default). Per-attempt timeouts of 5 s to connect and 120 s to read. Retries with growing pauses (1 s, 2 s, 4 s). An HTTP 429 is retried once, a 404 (model not found) is not retried. All attempts for one question together are capped by `LLM_MAX_TOTAL_SECONDS` (180 s).

**Why the cap.** Before it, three attempts of 120 s plus pauses meant a single question could take about 6 minutes. Now each attempt only gets the time that is left, and a pause that would pass the limit is skipped. The tests use a fake clock so they do not have to wait.

**Startup check.** The bot sends a tiny request at startup and stops with a clear message if the server is unreachable, the model is missing or the key is wrong. `/status` uses a quick call to `/models` instead, so checking health does not generate text.

**Questions**

- *What is exponential backoff?* "Wait longer after each failure (1, 2, 4 seconds) so a struggling server is not hammered."
- *Why not retry a 404?* "It means the model does not exist. Retrying cannot fix it, and the message tells the user to run `ollama pull`."
- *Why total time limit and not only per-request timeouts?* "Timeouts apply to one attempt. Retries multiply them. The total limit is what the user actually experiences."

## 9. Concurrency

**Decision.** The bot is an asyncio program (python-telegram-bot). `concurrent_updates` lets many updates run at the same time. Slow, blocking work (embedding, PDF parsing, URL checks, the LLM call) runs in worker threads with `asyncio.to_thread`, so the event loop stays free. A semaphore lets only one LLM answer run at a time. The embedding model and the reranker are protected by locks.

**Why.** Before, updates were processed one by one, so a single slow answer froze `/list`, uploads and every other user. A test starts a slow answer, then checks that another user's `/list` completes while it is still running. I also checked the test can fail: when I made the work run on the event loop, it failed with "the event loop was blocked".

**Why one answer at a time.** A CPU running a 3B model is already fully used by one answer. Two at once would each take twice as long. Users who wait see "Queued: N ahead".

**Why locks on the models.** Hugging Face tokenizers are not safe to share between threads, and the new design uses several threads. A test with six threads shows all six entered the model at once without the lock and one at a time with it.

**Request IDs across threads.** A context variable holds the request ID. `asyncio.to_thread` copies the context, so log lines written in the worker thread carry the same ID. A test checks this.

**Shutdown.** python-telegram-bot stops taking new messages and waits for running ones. Questions still waiting for the LLM when the bot stops get "the bot is restarting" instead of starting a long answer.

**Questions**

- *Threads versus async?* "Async is for waiting on the network; one thread can juggle many waits. Embedding and the LLM call use the CPU or block, so they go to threads. If I ran them on the event loop, everything would freeze."
- *What is a race condition? Did you have one?* "Two threads touching shared data in an unlucky order. The cache bug was this kind of problem in a single-thread form (stale shared state). With threads I added locks around the model and the cache."
- *Why a semaphore?* "To limit how many LLM calls run at once to one."

## 10. Separating users

**Decision.** Every query the bot uses to read documents or chunks filters by the Telegram user ID. File names are unique per user, not globally. Chroma queries filter by user ID metadata. Search caches include the user ID in the key. Deleting a document cascades to its chunks through a foreign key.

**Trade-offs.** Isolation relies on every code path remembering to filter. That is why the leak bug mattered, and why the tests run two users through the real code. A stronger design would be one database or one collection per user, at the cost of more files and complexity.

**Questions**

- *How do you make sure users cannot see each other's data?* "Filtering by user ID in every query, a cache keyed by user, and tests that run two users through the real components, including at the same time."
- *What is the weakest point?* "That it depends on discipline: if a new query forgets the filter, there is a leak. There is even an unused helper, `get_document_chunks`, that takes a document ID with no user filter. Nothing calls it, but it is the kind of thing a repository layer that always applies the user ID would remove."

## 11. Access control and rate limits

**Decision.** An optional allowlist of Telegram IDs. The check is a handler in an early handler group, so it runs before all others, and a test sends real update objects through a real application to prove a stranger never reaches a command. Rate limits use a sliding window per user (default 5 questions and 3 uploads per minute), and a cap of 20 stored documents.

**Why a sliding window.** A fixed window lets a user send a burst at the end of one window and another at the start of the next. A sliding window counts the last N seconds exactly. Refused attempts are not counted, so spamming does not extend the wait. Memory is bounded: idle users are forgotten once many are tracked.

**Why the limits matter here.** Each question uses a lot of CPU, and each upload runs the embedding model, so one user could make the bot unusable for everyone.

**Trade-offs.** Counters live in memory and reset when the bot restarts. The document limit is checked before the work starts, so two simultaneous uploads could exceed it by one.

**Questions**

- *Fixed window versus sliding window?* "The sliding window cannot be gamed at the boundary of a window, at the cost of storing timestamps."
- *Is it thread-safe?* "It is not locked, on purpose: it is only called from the event loop, which runs one step at a time. I wrote that in the class documentation."

## 12. Safe downloads (SSRF)

**Decision.** The bot downloads URLs that users send. An attacker could send a URL that points at a service inside my network (`http://127.0.0.1`, a router, the cloud metadata address), which is called SSRF (server-side request forgery). Defenses: only `http` and `https`; no credentials in the URL; the host is resolved and every resulting address is checked against private, loopback, link-local, multicast and reserved ranges (including odd forms such as `0x7f000001` and `2130706433`); redirects are followed manually, at most 3, and each hop is checked again; downloads are streamed with a size limit and a total time limit.

**The known gap.** The address is checked, and then `requests` resolves the name again when it connects. An attacker who controls a DNS server could answer with a public address the first time and a private one the second time (DNS rebinding). The fix is to connect to the address that was checked (IP pinning). That is hard to do correctly with HTTPS because the certificate check depends on the host name. I documented the gap in the code, the README and the tests instead of claiming it is solved.

**Questions**

- *What is SSRF?* "Tricking a server into making requests for the attacker, usually to reach internal systems."
- *Why check every redirect?* "A public URL can redirect to an internal one. Checking only the first URL would miss it."
- *Is your protection complete?* "No. DNS rebinding is still possible. Keeping the allowlist on limits who can send URLs, and I list it as a limitation."

## 13. Configuration and startup checks

**Decision.** All settings come from environment variables (`.env`). One reader collects every problem (wrong type, out of range, unknown choice, missing token) and the bot stops with one message that lists all of them. After that, it checks the LLM server with a real request and gives specific advice for each failure: unreachable, model missing, bad key. A remote service needs an API key; a local or internal host such as `ollama` in Docker does not.

**Why fail fast.** A wrong value found at 3 a.m. during the first question is worse than a clear error at start. It also means the code after startup can trust its settings.

**Trade-offs.** More code than `int(os.getenv(...))`. The benefit shows when a setting is wrong: the old code crashed with `invalid literal for int()` without naming the setting.

**Questions**

- *Why environment variables?* "They work the same on a laptop and in Docker, and secrets stay out of the code. The `.env` file is ignored by Git and by Docker."
- *How do you test configuration?* "Tests set the environment, call the loader and check the messages. A test also checks that every setting is listed in `.env.example` and the README, so the documentation cannot drift."

## 14. Logging

**Decision.** One JSON object per line in a rotating file (5 MB, 3 backups), readable text on the console. Every question and upload gets a short request ID that appears on every log line it causes, also inside worker threads. Secrets are removed from each finished line: the bot token (matched by shape), the LLM key and bearer tokens. URLs are logged without query strings. Document text, questions and answers are never logged, only their length.

**Why.** To follow one request through the logs you need an ID. Because logs are copied into tickets and chat, they must not hold private data. Redacting the final line also covers tracebacks.

**Real bug caught by a test.** My first token pattern needed a word boundary before the token, but a real Telegram URL looks like `.../bot123456:TOKEN/getUpdates`, with no boundary between `bot` and the digits. The test used a realistic URL and failed, so the pattern was fixed before it ever shipped.

**Questions**

- *What would you not log?* "Anything private: message text, document content, tokens, full URLs with query strings."
- *Why JSON logs?* "Tools can filter them, for example by request ID, without writing regular expressions."

## 15. Docker

**Decision.** `docker compose up` starts three services: Ollama, a one-time job that downloads the model, and the bot. The bot image uses `python:3.12-slim`, installs the CPU-only build of PyTorch (the default Linux build pulls several GB of CUDA files this project never uses), pins the library versions that the tests were run with, downloads the embedding model during the build, and runs as a normal user. Data, logs and models are in named volumes. The bot waits for Ollama to be healthy. No port is published. `.env` is never copied into the image.

**Honest status.** Docker could not be run when this was written, so the files are checked by tests (non-root user, CPU-only, no `.env` copy, no published ports, volumes, health dependency) and by `docker compose config`, not by a real start. Say that if asked, and run it before the interview.

**Questions**

- *Why a non-root user?* "If someone breaks into the bot, they should not be root inside the container."
- *Why pin versions?* "Builds should be repeatable. The pins are the versions the test suite passed with."
- *Why a separate model-download job?* "The bot should not start before the model exists, and the download is a one-time job. The volume keeps the model between restarts."

## 16. Evaluation

**Decision.** Build a test set before trusting the system: 70 questions on a real book (50 answerable with the correct pages, 10 off-topic, 10 on-topic but absent). Report two ways of judging a hit: "span" (the chunk's page range overlaps the target page) and "strict" (the chunk starts on the target page). Run the same questions through dense, hybrid and hybrid with reranking. Keep a small synthetic document as a regression test in CI.

**Things I did that interviewers like to hear.**

- I did not believe my first perfect faithfulness score, tested the judge with wrong answers, and found it was too lenient.
- I split the data into development and test halves before tuning the abstention gate, and I report that the result is based on 5 questions.
- I report negative results: hybrid is not clearly better than dense, hard questions are not refused.
- I kept a hand-checked list of 16 answers from the real bot instead of only automatic metrics.

**Weak points to admit.** One book, 50 questions, one laptop, an informal manual check on an older version, and a judge I do not trust.

**Questions**

- *What is Hit@5?* "The share of questions where a correct page is among the top 5 results."
- *How would you improve the evaluation?* "More documents and question types, more hand-labelled answers, and a trustworthy way to judge whether an answer is supported."

## 17. Testing

**Decision.** More than 350 tests that run without Ollama or Telegram. Fake models and fake Telegram objects are used where the real ones would be slow or need a network. Bugs get a failing test first. Some tests use real components (a real SQLite file, real retrieval) with only the language model faked, so they check how the pieces work together.

**Techniques worth naming.**

- A fake clock so timeout and rate-limit tests are instant and exact.
- Concurrency tests with threads and events to force an overlap.
- A mutation check: break the code on purpose (run blocking work on the event loop) and confirm the test fails, which proves the test can detect the problem.
- End-to-end tests that send real `Update` objects through a real `Application`.
- Documentation tests: every setting is documented, links work, known-false statements do not come back, and the Docker and CI files keep their important properties.

**Questions**

- *What is the difference between a unit test and an integration test?* "A unit test checks one piece in isolation. An integration test uses several real pieces together. I use both, and the leak bug showed why the second kind matters."
- *How do you test code that waits or uses time?* "Inject a clock so the test controls time instead of sleeping."
- *What is test-driven development in your words?* "For a bug, I first write a test that fails because of the bug, then fix the code until it passes. The test then protects against it returning."

## What I would do next, and how to talk about weaknesses

Say weaknesses before you are asked. It shows judgment.

**Next steps, in order of value**

1. Run the Docker setup end to end and fix what it finds.
2. Evaluate on more documents, and hand-check more answers.
3. Find a way to refuse in-topic questions that the documents do not cover (the 0 of 10 result).
4. Pin the connection to the checked IP address to close the DNS rebinding gap.
5. A hard time limit for a whole question, not only for the LLM call.
6. Add OCR for scanned PDFs and better summaries for long documents.
7. A repository layer that always applies the user filter, so isolation does not depend on every query remembering it.

**Questions about the project as a whole**

- *What was the hardest bug?* The cross-user leak: the cause was design, not syntax, and tests with a single user hid it.
- *What would break first if 100 people used it?* "The LLM answers one question at a time, so people would wait minutes. The first fix is a faster machine or a hosted model, since the bot talks to any OpenAI-compatible endpoint."
- *Why a 3B model?* "A 7B model took 10 to 20 minutes per answer on this laptop. A 3B model takes about 14 seconds in my benchmark. The price is more wrong answers, which I measured by hand."
- *Is this production ready?* "It is a portfolio project that I hardened: input checks, limits, safe logging, tests. I would not put it in front of the public. It is single machine, unencrypted at rest, and the language model is small."
- *What did you learn?* "That measuring is harder than building, and that shared state is where multi-user bugs live."
