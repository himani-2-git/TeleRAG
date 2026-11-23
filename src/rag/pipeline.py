"""RAG execution pipeline with Plain and LangGraph orchestrators."""

import importlib.util
import logging
import re
from typing import Any, TypedDict

from .grounding import (
    grounding_report,
    is_refusal_sentence,
    split_sentences,
)
from .llm_client import LLMClient, LLMUnavailableError
from .prompts import PromptConfig, load_prompt_config
from .retriever import DocumentRetriever

logger = logging.getLogger(__name__)

# Only a few words can follow a summary request. If the question names a topic ("summary of
# chapter 3", "summarize the Raft algorithm") it is a normal question and goes through retrieval.
_DOCUMENT_WORDS = r"(?:document|doc|pdf|book|file|paper|article|page|report|text)"
_THIS_DOCUMENT = rf"(?:this|the|my|that)\s+(?:(?:whole|entire|full)\s+)?{_DOCUMENT_WORDS}"
_WHOLE_THING = (
    rf"(?:it|everything|(?:the\s+)?(?:topics|contents?|chapters|sections)|{_THIS_DOCUMENT})"
)
_POLITE = r"(?:(?:please|kindly|can you|could you|would you|can i get|could i get|can i have)\s+)*"
# Harmless words that may end the request: "summarize this document for me, please"
_END = r"(?:\s+(?:for me|please|briefly|in brief))*$"

SUMMARY_REQUEST_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        # "summarize", "summarise this document", "please summarize the entire book"
        rf"^{_POLITE}(?:summari[sz]e|sum up)(?:\s+{_WHOLE_THING})?{_END}",
        # "give me a summary", "an overview of the book", "what is the overview of this paper"
        rf"^{_POLITE}(?:(?:give me|provide|write|show me|i need|i want|what is|what's)\s+)?"
        rf"(?:(?:a|an|the)\s+)?(?:(?:brief|short|quick|full)\s+)?"
        rf"(?:summary|overview|tl;?dr)(?:\s+of\s+{_WHOLE_THING})?{_END}",
        # "what is this book about"
        rf"^(?:what is|what's)\s+{_THIS_DOCUMENT}\s+(?:about|on){_END}",
        # "tell me about this document", "explain this document"
        rf"^(?:tell me about|explain|describe)\s+{_THIS_DOCUMENT}{_END}",
        # "what are the main points", "key takeaways of this document"
        rf"^(?:(?:what are|list|give me)\s+)?(?:the\s+)?"
        rf"(?:main points|key points|key takeaways|main ideas)"
        rf"(?:\s+(?:of|in|from)\s+{_WHOLE_THING})?{_END}",
    )
)

# A summary is built from this many chunks, spread evenly over the whole document.
SUMMARY_MAX_CHUNKS = 8


def is_summary_query(query: str) -> bool:
    """Return True only if the whole question is a request to summarise the document."""
    normalized = query.lower().replace("\u2019", "'").replace(",", "")
    normalized = " ".join(normalized.split()).strip(" .!?")
    return any(pattern.match(normalized) for pattern in SUMMARY_REQUEST_PATTERNS)


def _reading_order(chunk: dict[str, Any]) -> tuple[int, int, int]:
    """Sort key that puts chunks in document order: upload order, then page, then position."""
    document_id = chunk.get("document_id")
    if not isinstance(document_id, int):
        document_id = 0
    page = chunk.get("page_start")
    if page is None:
        page = chunk.get("page") or 0
    return (document_id, page, chunk.get("chunk_index", 0))


def select_summary_chunks(
    chunks: list[dict[str, Any]], max_chunks: int = SUMMARY_MAX_CHUNKS
) -> list[dict[str, Any]]:
    """Pick up to max_chunks chunks spread evenly over the whole document, in reading order."""
    ordered = sorted(chunks, key=_reading_order)
    if len(ordered) <= max_chunks:
        return ordered
    if max_chunks == 1:
        return ordered[:1]

    # Integer maths: the first pick is chunk 0, the last is the final chunk, and because the
    # step is larger than 1 no chunk is picked twice.
    last = len(ordered) - 1
    return [ordered[i * last // (max_chunks - 1)] for i in range(max_chunks)]


def is_insufficient_context(answer: str, marker: str = "INSUFFICIENT_CONTEXT") -> bool:
    """Return True if the answer is ONLY a refusal.

    An answer that mixes a useful sentence with "the context does not mention X" is a partial
    answer, not a refusal. Whether its useful sentences are grounded is decided later, by the
    grounding check.
    """
    if not answer or not marker:
        return False
    sentences = split_sentences(answer)
    if not sentences:
        return False
    return all(is_refusal_sentence(sentence, marker) for sentence in sentences)


def contains_refusal_sentence(answer: str, marker: str = "INSUFFICIENT_CONTEXT") -> bool:
    """Return True if at least one sentence of the answer is a refusal."""
    return any(is_refusal_sentence(sentence, marker) for sentence in split_sentences(answer))


def remove_marker_sentences(answer: str, marker: str) -> str:
    """Remove sentences that contain the raw refusal marker so users never see it."""
    pattern = rf"[^.!?\n]*{re.escape(marker)}[^.!?\n]*[.!?]?"
    cleaned = re.sub(pattern, "", answer, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def format_source_footer(chunks: list[dict[str, Any]]) -> str:
    """Format the citation footer block for retrieved chunks."""
    if not chunks or not isinstance(chunks, (list, tuple)):
        return ""

    sources = []
    seen = set()
    for chunk in chunks:
        filename = chunk.get("filename", "document.pdf")
        source_type = chunk.get("source_type", "pdf")
        section_label = chunk.get("section_label")
        p_start = chunk.get("page_start")
        p_end = chunk.get("page_end")
        page = chunk.get("page")

        if source_type == "web":
            entry = section_label if section_label else filename
        elif source_type == "markdown" and section_label:
            entry = f"{filename} ({section_label})"
        elif p_start is not None and p_end is not None:
            page_str = f"p. {p_start}" if p_start == p_end else f"p. {p_start}-{p_end}"
            entry = f"{filename} ({page_str})"
        elif page is not None:
            entry = f"{filename} (p. {page})"
        elif section_label:
            entry = f"{filename} ({section_label})"
        elif source_type == "text":
            entry = f"{filename}"
        else:
            entry = f"{filename} (p. 1)"

        if entry not in seen:
            seen.add(entry)
            sources.append(entry)

    if not sources:
        return ""

    footer_lines = ["\n\nSources:"] + [f"• {s}" for s in sources]
    return "\n".join(footer_lines)


def count_substantive_words(text: str) -> int:
    """Count words after removing bracketed citations, Sources lines, and whitespace."""
    if not text or not isinstance(text, str):
        return 0
    # Remove bracketed citations like [aipython.pdf, p. 11]
    cleaned = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)
    # Remove Sources: blocks to end of text
    cleaned = re.sub(r"(?:\n\s*|\A)sources:\s*.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Remove any individual line starting with sources
    lines = [
        line
        for line in cleaned.splitlines()
        if not re.match(r"^\s*sources\b", line, flags=re.IGNORECASE)
    ]
    cleaned = " ".join(lines).strip()
    return len(cleaned.split())


class RAGState(TypedDict, total=False):
    """Pipeline state passed across graph nodes."""

    question: str
    user_id: int
    user_chunks: list[dict[str, Any]]
    query_embedding: list[float] | None
    retrieved_chunks: list[dict[str, Any]]
    raw_answer: str | None
    answer: str
    refused: bool
    refusal_reason: str | None
    partial_answer: bool
    grounding_report: Any
    final_response: str


class RAGPipeline:
    """Encapsulates RAG components and execution nodes."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: LLMClient,
        embedding_generator: Any,
        prompt_config: PromptConfig | None = None,
        grounding_threshold: float = 0.45,
        grounding_enforce: bool = False,
        grounding_min_coverage: float = 0.5,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.prompt_config = prompt_config or load_prompt_config("v1")
        self.grounding_threshold = grounding_threshold
        self.grounding_enforce = grounding_enforce
        self.grounding_min_coverage = grounding_min_coverage

    def _is_insufficient_context(self, answer: str) -> bool:
        marker = self.prompt_config.insufficient_context_marker
        return is_insufficient_context(answer, marker)

    def node_retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks, or sample the whole document for summary requests."""
        question = state["question"]
        user_chunks = state.get("user_chunks", [])

        if not user_chunks:
            return {
                **state,
                "retrieved_chunks": [],
                "refused": True,
                "refusal_reason": "NO_CHUNKS",
                "answer": "❌ No document content available.",
            }

        if is_summary_query(question):
            return {**state, "retrieved_chunks": select_summary_chunks(user_chunks)}

        query_embedding = state.get("query_embedding")
        if query_embedding is None and self.embedding_generator is not None:
            query_embedding = self.embedding_generator.generate_query_embedding(question)

        retrieved = self.retriever.retrieve(
            query=question,
            query_embedding=query_embedding or [],
            chunks=user_chunks,
            user_id=state.get("user_id"),
        )
        return {**state, "query_embedding": query_embedding, "retrieved_chunks": retrieved}

    def node_abstain_or_generate(self, state: RAGState) -> RAGState:
        """Check threshold; either abstain with refusal or invoke LLM."""
        if state.get("refused"):
            return state

        retrieved = state.get("retrieved_chunks", [])
        if not retrieved:
            refusal_msg = self.prompt_config.refusal_message
            return {
                **state,
                "answer": refusal_msg,
                "refused": True,
                "refusal_reason": "NO_RELEVANT",
            }

        question = state["question"]
        context_text = self.retriever.format_context(retrieved)

        try:
            raw_ans = self.llm_client.generate_answer(question, context_text)
        except LLMUnavailableError:
            return {
                **state,
                "raw_answer": None,
                "answer": "❌ The model is currently unavailable. Please try again shortly.",
                "refused": True,
                "refusal_reason": "LLM_UNAVAILABLE",
            }

        answer = self.llm_client.validate_response(raw_ans) if raw_ans else ""

        if self._is_insufficient_context(answer):
            refusal_msg = self.prompt_config.refusal_message
            return {
                **state,
                "raw_answer": raw_ans,
                "answer": refusal_msg,
                "refused": True,
                "refusal_reason": "INSUFFICIENT_CONTEXT",
            }

        if count_substantive_words(answer) < 3:
            logger.info("Generated answer has fewer than 3 substantive words; retrying once.")
            try:
                retry_raw = self.llm_client.generate_answer(question, context_text)
            except LLMUnavailableError:
                retry_raw = None

            if retry_raw:
                raw_ans = retry_raw
                answer = self.llm_client.validate_response(raw_ans)
                if self._is_insufficient_context(answer):
                    return {
                        **state,
                        "raw_answer": raw_ans,
                        "answer": self.prompt_config.refusal_message,
                        "refused": True,
                        "refusal_reason": "INSUFFICIENT_CONTEXT",
                    }

            if count_substantive_words(answer) < 3:
                return {
                    **state,
                    "raw_answer": raw_ans,
                    "answer": "I couldn't produce a usable answer; try rephrasing the question.",
                    "refused": True,
                    "refusal_reason": "UNUSABLE_ANSWER",
                }

        # Keep a partial answer, but remember it contained a refusal sentence: the grounding
        # step must confirm that the remaining sentences are really supported by the context.
        marker = self.prompt_config.insufficient_context_marker
        is_partial = contains_refusal_sentence(answer, marker)
        if is_partial:
            answer = remove_marker_sentences(answer, marker)

        return {
            **state,
            "raw_answer": raw_ans,
            "answer": answer,
            "refused": False,
            "partial_answer": is_partial,
        }

    def node_grounding(self, state: RAGState) -> RAGState:
        """Evaluate semantic grounding against retrieved passages."""
        if state.get("refused"):
            return state

        answer = state.get("answer", "")
        retrieved = state.get("retrieved_chunks", [])
        is_partial = state.get("partial_answer", False)
        refusal_msg = self.prompt_config.refusal_message

        try:
            report = grounding_report(
                answer=answer,
                chunks=retrieved,
                embedder=self.embedding_generator,
                threshold=self.grounding_threshold,
                marker=self.prompt_config.insufficient_context_marker,
            )
        except Exception as e:
            # Always log first, whatever we decide to do next.
            logger.exception("Grounding check failed: %s", e)
            if self.grounding_enforce or is_partial:
                return {
                    **state,
                    "answer": refusal_msg,
                    "refused": True,
                    "refusal_reason": "GROUNDING_ERROR",
                }
            return state

        state_update: RAGState = {**state, "grounding_report": report}

        if is_partial and report.supported_sentences == 0:
            # The answer had a refusal sentence and nothing else in it is backed by the context.
            state_update["answer"] = refusal_msg
            state_update["refused"] = True
            state_update["refusal_reason"] = "INSUFFICIENT_CONTEXT"
        elif (
            self.grounding_enforce
            and report.applicable
            and report.citation_coverage < self.grounding_min_coverage
        ):
            state_update["answer"] = refusal_msg
            state_update["refused"] = True
            state_update["refusal_reason"] = "GROUNDING_UNSUPPORTED"

        return state_update

    def node_format_reply(self, state: RAGState) -> RAGState:
        """Format final reply, appending Sources footer if applicable."""
        if state.get("refused"):
            return {
                **state,
                "final_response": state.get("answer", self.prompt_config.refusal_message),
            }

        answer = state.get("answer", "")
        retrieved = state.get("retrieved_chunks", [])
        report = state.get("grounding_report")

        if report is not None and getattr(report, "applicable", False):
            supporting_indices = getattr(report, "supporting_chunk_indices", None)
            if supporting_indices is not None:
                if supporting_indices:
                    supporting_chunks = [
                        retrieved[i] for i in supporting_indices if i < len(retrieved)
                    ]
                    footer = format_source_footer(supporting_chunks)
                    final_text = f"{answer}{footer}" if footer else answer
                else:
                    final_text = f"{answer.rstrip()}\n\n(Could not verify this answer against the cited pages.)"
                return {**state, "final_response": final_text}

        # If grounding data is unavailable, keep the current behavior
        footer = format_source_footer(retrieved)
        final_text = f"{answer}{footer}" if footer else answer

        return {**state, "final_response": final_text}


class PlainOrchestrator:
    """Sequential direct execution of the 4 RAG pipeline nodes."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(
        self,
        question: str,
        user_id: int,
        user_chunks: list[dict[str, Any]],
        query_embedding: list[float] | None = None,
    ) -> RAGState:
        state: RAGState = {
            "question": question,
            "user_id": user_id,
            "user_chunks": user_chunks,
            "query_embedding": query_embedding,
            "refused": False,
        }
        state = self.pipeline.node_retrieve(state)
        state = self.pipeline.node_abstain_or_generate(state)
        state = self.pipeline.node_grounding(state)
        state = self.pipeline.node_format_reply(state)
        return state


class LangGraphOrchestrator:
    """Compiled StateGraph orchestration of the 4 RAG pipeline nodes."""

    def __init__(self, pipeline: RAGPipeline):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as e:
            raise ImportError(
                "langgraph is required for LangGraphOrchestrator. "
                "Install it using: pip install -r requirements/graph.txt"
            ) from e

        self.pipeline = pipeline

        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self.pipeline.node_retrieve)
        builder.add_node("abstain_or_generate", self.pipeline.node_abstain_or_generate)
        builder.add_node("grounding", self.pipeline.node_grounding)
        builder.add_node("format_reply", self.pipeline.node_format_reply)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "abstain_or_generate")
        builder.add_edge("abstain_or_generate", "grounding")
        builder.add_edge("grounding", "format_reply")
        builder.add_edge("format_reply", END)

        self.graph = builder.compile()

    def run(
        self,
        question: str,
        user_id: int,
        user_chunks: list[dict[str, Any]],
        query_embedding: list[float] | None = None,
    ) -> RAGState:
        initial_state: RAGState = {
            "question": question,
            "user_id": user_id,
            "user_chunks": user_chunks,
            "query_embedding": query_embedding,
            "refused": False,
        }
        return self.graph.invoke(initial_state)


def get_orchestrator(pipeline: RAGPipeline, orchestrator: str = "plain") -> Any:
    """Factory to get the appropriate pipeline orchestrator."""
    mode = (orchestrator or "plain").strip().lower()
    if mode == "langgraph":
        if importlib.util.find_spec("langgraph") is not None:
            return LangGraphOrchestrator(pipeline)
        logger.warning(
            "langgraph is requested (ORCHESTRATOR=langgraph) but not installed. "
            "Falling back to plain orchestrator."
        )
        return PlainOrchestrator(pipeline)
    return PlainOrchestrator(pipeline)
