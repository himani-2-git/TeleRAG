"""RAG execution pipeline with Plain and LangGraph orchestrators."""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .retriever import DocumentRetriever
from .llm_client import LLMClient
from .prompts import PromptConfig, load_prompt_config
from .grounding import grounding_report

logger = logging.getLogger(__name__)


def format_source_footer(chunks: List[Dict[str, Any]]) -> str:
    """Format the sources / citation footer block for retrieved chunks.

    Args:
        chunks: List of chunk metadata dictionaries.

    Returns:
        Formatted Sources string or empty string if no citations.
    """
    if not chunks:
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


class RAGState(TypedDict, total=False):
    """Pipeline state passed across graph nodes."""
    question: str
    user_id: int
    user_chunks: List[Dict[str, Any]]
    query_embedding: Optional[List[float]]
    retrieved_chunks: List[Dict[str, Any]]
    raw_answer: Optional[str]
    answer: str
    refused: bool
    refusal_reason: Optional[str]
    grounding_report: Any
    final_response: str


class RAGPipeline:
    """Encapsulates RAG components and execution nodes."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: LLMClient,
        embedding_generator: Any,
        prompt_config: Optional[PromptConfig] = None,
        grounding_threshold: float = 0.45,
        grounding_enforce: bool = False,
        grounding_min_coverage: float = 0.5
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.prompt_config = prompt_config or load_prompt_config("v1")
        self.grounding_threshold = grounding_threshold
        self.grounding_enforce = grounding_enforce
        self.grounding_min_coverage = grounding_min_coverage

    def _is_insufficient_context(self, answer: str) -> bool:
        if not answer:
            return False
        clean = answer.strip().upper()
        marker = getattr(self.prompt_config, "insufficient_context_marker", "INSUFFICIENT_CONTEXT").upper()
        return clean.startswith(marker) or marker in clean

    # --- Node 1: retrieve ---
    def node_retrieve(self, state: RAGState) -> RAGState:
        """Node 1: Retrieve relevant chunks using configured retriever."""
        question = state["question"]
        user_chunks = state.get("user_chunks", [])

        if not user_chunks:
            return {
                **state,
                "retrieved_chunks": [],
                "refused": True,
                "refusal_reason": "NO_CHUNKS"
            }

        query_embedding = state.get("query_embedding")
        if query_embedding is None and self.embedding_generator is not None:
            query_embedding = self.embedding_generator.generate_query_embedding(question)

        retrieved = self.retriever.retrieve(
            query=question,
            query_embedding=query_embedding or [],
            chunks=user_chunks
        )
        return {
            **state,
            "query_embedding": query_embedding,
            "retrieved_chunks": retrieved
        }

    # --- Node 2: abstain_or_generate ---
    def node_abstain_or_generate(self, state: RAGState) -> RAGState:
        """Node 2: Checks threshold, either abstains with refusal or calls LLM."""
        if state.get("refused"):
            return state

        retrieved = state.get("retrieved_chunks", [])
        if not retrieved:
            refusal_msg = self.prompt_config.refusal_message
            return {
                **state,
                "answer": refusal_msg,
                "refused": True,
                "refusal_reason": "NO_RELEVANT"
            }

        question = state["question"]
        context_text = self.retriever.format_context(retrieved)
        raw_ans = self.llm_client.generate_answer(question, context_text)
        answer = self.llm_client.validate_response(raw_ans)

        if self._is_insufficient_context(answer):
            refusal_msg = self.prompt_config.refusal_message
            return {
                **state,
                "raw_answer": raw_ans,
                "answer": refusal_msg,
                "refused": True,
                "refusal_reason": "INSUFFICIENT_CONTEXT"
            }

        return {
            **state,
            "raw_answer": raw_ans,
            "answer": answer,
            "refused": False
        }

    # --- Node 3: grounding ---
    def node_grounding(self, state: RAGState) -> RAGState:
        """Node 3: Evaluates semantic grounding against retrieved chunks."""
        if state.get("refused"):
            return state

        answer = state.get("answer", "")
        retrieved = state.get("retrieved_chunks", [])

        try:
            report = grounding_report(
                answer=answer,
                chunks=retrieved,
                embedder=self.embedding_generator,
                threshold=self.grounding_threshold
            )
            state_update: RAGState = {**state, "grounding_report": report}

            if (self.grounding_enforce and 
                report.applicable and 
                report.citation_coverage < self.grounding_min_coverage):
                refusal_msg = self.prompt_config.refusal_message
                state_update["answer"] = refusal_msg
                state_update["refused"] = True
                state_update["refusal_reason"] = "GROUNDING_UNSUPPORTED"

            return state_update
        except Exception as e:
            logger.warning("Grounding check failed: %s", e)
            return state

    # --- Node 4: format_reply ---
    def node_format_reply(self, state: RAGState) -> RAGState:
        """Node 4: Formats final reply, appending Sources footer if applicable."""
        if state.get("refused"):
            return {
                **state,
                "final_response": state.get("answer", self.prompt_config.refusal_message)
            }

        answer = state.get("answer", "")
        retrieved = state.get("retrieved_chunks", [])
        footer = format_source_footer(retrieved)
        final_text = f"{answer}{footer}" if footer else answer

        return {
            **state,
            "final_response": final_text
        }


class PlainOrchestrator:
    """Sequential direct execution of the 4 RAG pipeline nodes."""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(
        self,
        question: str,
        user_id: int,
        user_chunks: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None
    ) -> RAGState:
        state: RAGState = {
            "question": question,
            "user_id": user_id,
            "user_chunks": user_chunks,
            "query_embedding": query_embedding,
            "refused": False
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
            from langgraph.graph import StateGraph, END
        except ImportError as e:
            raise ImportError(
                "langgraph is required for LangGraphOrchestrator. "
                "Install it using: pip install -r requirements-graph.txt"
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
        user_chunks: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None
    ) -> RAGState:
        initial_state: RAGState = {
            "question": question,
            "user_id": user_id,
            "user_chunks": user_chunks,
            "query_embedding": query_embedding,
            "refused": False
        }
        return self.graph.invoke(initial_state)


def get_orchestrator(
    pipeline: RAGPipeline,
    orchestrator: str = "plain"
) -> Any:
    """Factory to get the appropriate pipeline orchestrator.

    Args:
        pipeline: RAGPipeline component instance.
        orchestrator: 'plain' or 'langgraph'.

    Returns:
        PlainOrchestrator or LangGraphOrchestrator instance.
    """
    mode = (orchestrator or "plain").strip().lower()
    if mode == "langgraph":
        try:
            import langgraph
            return LangGraphOrchestrator(pipeline)
        except ImportError:
            logger.warning(
                "langgraph is requested (ORCHESTRATOR=langgraph) but not installed. "
                "Falling back to plain orchestrator."
            )
            return PlainOrchestrator(pipeline)
    return PlainOrchestrator(pipeline)
