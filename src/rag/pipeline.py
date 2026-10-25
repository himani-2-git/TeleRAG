"""RAG execution pipeline with Plain and LangGraph orchestrators."""

import importlib.util
import logging
import re
from typing import Any, TypedDict

from .grounding import grounding_report
from .llm_client import LLMClient, LLMUnavailableError
from .prompts import PromptConfig, load_prompt_config
from .retriever import DocumentRetriever

logger = logging.getLogger(__name__)

SUMMARY_CUES = (
    "summarize",
    "summary",
    "overview",
    "main points",
    "key takeaways",
    "what is this document about",
    "what is this book about",
    "what is the document about",
    "what is the book about",
    "what is this pdf about",
    "what is the pdf about",
    "tell me about this document",
    "tell me about this pdf",
    "explain this document",
    "brief overview",
)


def is_summary_query(query: str) -> bool:
    """Return True if query is asking for a document summary or overview."""
    q = query.lower().strip()
    return any(cue in q for cue in SUMMARY_CUES)


def is_insufficient_context(answer: str, marker: str = "INSUFFICIENT_CONTEXT") -> bool:
    """Check if the answer contains the insufficient context marker (case-insensitive)."""
    if not answer or not marker or not isinstance(answer, str) or not isinstance(marker, str):
        return False
    return marker.upper() in answer.upper()


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
        marker = getattr(self.prompt_config, "insufficient_context_marker", "INSUFFICIENT_CONTEXT")
        return is_insufficient_context(answer, marker)

    def node_retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks or first chunks for summary queries."""
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
            sorted_chunks = sorted(
                user_chunks,
                key=lambda c: (
                    c.get("page_start")
                    if c.get("page_start") is not None
                    else (c.get("page") or 0),
                    c.get("chunk_index", 0),
                ),
            )
            return {**state, "retrieved_chunks": sorted_chunks[:8]}

        query_embedding = state.get("query_embedding")
        if query_embedding is None and self.embedding_generator is not None:
            query_embedding = self.embedding_generator.generate_query_embedding(question)

        retrieved = self.retriever.retrieve(
            query=question, query_embedding=query_embedding or [], chunks=user_chunks
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

        return {**state, "raw_answer": raw_ans, "answer": answer, "refused": False}

    def node_grounding(self, state: RAGState) -> RAGState:
        """Evaluate semantic grounding against retrieved passages."""
        if state.get("refused"):
            return state

        answer = state.get("answer", "")
        retrieved = state.get("retrieved_chunks", [])

        try:
            report = grounding_report(
                answer=answer,
                chunks=retrieved,
                embedder=self.embedding_generator,
                threshold=self.grounding_threshold,
            )
            state_update: RAGState = {**state, "grounding_report": report}

            if (
                self.grounding_enforce
                and report.applicable
                and report.citation_coverage < self.grounding_min_coverage
            ):
                refusal_msg = self.prompt_config.refusal_message
                state_update["answer"] = refusal_msg
                state_update["refused"] = True
                state_update["refusal_reason"] = "GROUNDING_UNSUPPORTED"

            return state_update
        except Exception as e:
            if self.grounding_enforce:
                refusal_msg = self.prompt_config.refusal_message
                return {
                    **state,
                    "answer": refusal_msg,
                    "refused": True,
                    "refusal_reason": "GROUNDING_ERROR",
                }
            logger.exception("Grounding check failed: %s", e)
            return state

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
