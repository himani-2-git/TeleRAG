"""Tests for Plain and LangGraph pipeline orchestration."""

import importlib.util
import os
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from src.rag.pipeline import (
    LangGraphOrchestrator,
    PlainOrchestrator,
    RAGPipeline,
    get_orchestrator,
)
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


class MockEmbeddingGenerator:
    """Deterministic mock embedding generator for testing."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.model = None

    def generate_query_embedding(self, query: str) -> list[float]:
        if self.model is not None:
            return self.model.encode(query, convert_to_numpy=True).tolist()
        return [0.1] * 384

    def generate_embeddings(self, texts: Any) -> list[list[float]] | list[float]:
        if isinstance(texts, str):
            return self.generate_query_embedding(texts)
        return [self.generate_query_embedding(t) for t in texts]


class MockLLMClient:
    """Mock LLM client returning deterministic responses."""

    def generate_answer(self, question: str, context: str) -> str:
        return "Replication is the primary technique for high availability in distributed systems."

    def validate_response(self, text: Any) -> str:
        return str(text) if text else "I could not generate an answer."


class TestPipelinePlain(unittest.TestCase):
    """Test plain pipeline execution."""

    def setUp(self):
        self.retriever = DocumentRetriever(
            top_k=2, similarity_threshold=0.3, retrieval_mode="hybrid"
        )
        self.embedder = MockEmbeddingGenerator()
        self.llm = MockLLMClient()
        self.prompt_config = load_prompt_config("v1")
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.llm,
            embedding_generator=self.embedder,
            prompt_config=self.prompt_config,
        )
        self.orchestrator = PlainOrchestrator(self.pipeline)

    def test_unanswerable_abstains(self):
        """Unanswerable query abstains and returns refusal without error."""
        chunks = [
            {
                "content": "Raft consensus algorithm manages replicated log.",
                "chunk_index": 0,
                "filename": "raft.pdf",
                "embedding": self.embedder.generate_query_embedding(
                    "Raft consensus algorithm manages replicated log."
                ),
            }
        ]
        unanswerable_q = "How do pneumatic disc brakes operate in heavy vehicles?"
        result = self.orchestrator.run(question=unanswerable_q, user_id=1, user_chunks=chunks)
        self.assertTrue(result["refused"])
        self.assertEqual(result["final_response"], self.prompt_config.refusal_message)

    def test_answerable_question_success(self):
        """Answerable query runs cleanly without logging pipeline warnings."""
        chunks = [
            {
                "content": "Replication is the primary technique for high availability.",
                "chunk_index": 0,
                "filename": "replication.pdf",
                "embedding": self.embedder.generate_query_embedding(
                    "Replication is the primary technique for high availability."
                ),
            }
        ]
        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            res = self.orchestrator.run(
                question="What is replication?", user_id=1, user_chunks=chunks
            )
        self.assertFalse(res["refused"])
        self.assertIn("Replication is the primary technique", res["final_response"])

    @patch("src.rag.pipeline.grounding_report", side_effect=RuntimeError("Grounding service down"))
    def test_grounding_error_refuses_when_enforce_true(self, mock_report):
        """When grounding raises and grounding_enforce=True, refuse with GROUNDING_ERROR."""
        pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.llm,
            embedding_generator=self.embedder,
            prompt_config=self.prompt_config,
            grounding_enforce=True,
        )
        orch = PlainOrchestrator(pipeline)
        chunks = [
            {
                "content": "Replication is the primary technique for high availability.",
                "chunk_index": 0,
                "filename": "replication.pdf",
                "embedding": self.embedder.generate_query_embedding(
                    "Replication is the primary technique for high availability."
                ),
            }
        ]
        res = orch.run(question="What is replication?", user_id=1, user_chunks=chunks)
        self.assertTrue(res["refused"])
        self.assertEqual(res["refusal_reason"], "GROUNDING_ERROR")
        self.assertEqual(res["final_response"], self.prompt_config.refusal_message)

    @patch("src.rag.pipeline.grounding_report", side_effect=RuntimeError("Grounding service down"))
    def test_grounding_error_continues_when_enforce_false(self, mock_report):
        """When grounding raises and grounding_enforce=False, log exception and continue."""
        pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.llm,
            embedding_generator=self.embedder,
            prompt_config=self.prompt_config,
            grounding_enforce=False,
        )
        orch = PlainOrchestrator(pipeline)
        chunks = [
            {
                "content": "Replication is the primary technique for high availability.",
                "chunk_index": 0,
                "filename": "replication.pdf",
                "embedding": self.embedder.generate_query_embedding(
                    "Replication is the primary technique for high availability."
                ),
            }
        ]
        with self.assertLogs("src.rag.pipeline", level="ERROR") as cm:
            res = orch.run(question="What is replication?", user_id=1, user_chunks=chunks)
        self.assertFalse(res["refused"])
        self.assertIn("Replication is the primary technique", res["final_response"])
        self.assertTrue(any("Grounding check failed" in log for log in cm.output))


@unittest.skipUnless(HAS_LANGGRAPH, "langgraph is not installed")
class TestLangGraphParity(unittest.TestCase):
    """Test identical behavior between Plain and LangGraph orchestrators on ci_fixture.pdf."""

    def setUp(self):
        from sentence_transformers import SentenceTransformer

        from src.processors.pdf_processor import PDFProcessor
        from src.processors.text_chunker import TextChunker

        pdf_path = "eval/fixtures/ci_fixture.pdf"
        self.assertTrue(os.path.exists(pdf_path), f"Missing {pdf_path}")

        pdf_processor = PDFProcessor()
        chunker = TextChunker(chunk_size=1000, overlap=200)
        pages_data = pdf_processor.extract_text_with_pages(pdf_path)
        raw_chunks = chunker.create_chunks_with_pages(pages_data)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: list[dict[str, Any]] = []
        for i, c in enumerate(raw_chunks):
            emb = model.encode(c.content, convert_to_numpy=True).tolist()
            self.chunks.append(
                {
                    "chunk_index": i,
                    "content": c.content,
                    "char_count": c.char_count,
                    "page": c.page,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "filename": "ci_fixture.pdf",
                    "source_type": "pdf",
                    "embedding": emb,
                }
            )

        self.retriever = DocumentRetriever(
            top_k=2, similarity_threshold=0.3, retrieval_mode="hybrid"
        )
        self.embedder = MockEmbeddingGenerator()
        self.llm = MockLLMClient()
        self.prompt_config = load_prompt_config("v1")

        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.llm,
            embedding_generator=self.embedder,
            prompt_config=self.prompt_config,
        )
        self.plain_orch = PlainOrchestrator(self.pipeline)
        self.graph_orch = LangGraphOrchestrator(self.pipeline)

    def test_answerable_question_parity(self):
        """For answerable question, plain and langgraph produce identical responses and citations."""
        q = "What is the primary technique for high availability in distributed systems?"
        with self.assertNoLogs("src.rag.pipeline", level="WARNING"):
            res_plain = self.plain_orch.run(question=q, user_id=1, user_chunks=self.chunks)
            res_graph = self.graph_orch.run(question=q, user_id=1, user_chunks=self.chunks)

        self.assertFalse(res_plain["refused"])
        self.assertFalse(res_graph["refused"])
        self.assertEqual(res_plain["answer"], res_graph["answer"])
        self.assertEqual(res_plain["final_response"], res_graph["final_response"])
        self.assertIn("Sources:", res_graph["final_response"])
        self.assertEqual(len(res_plain["retrieved_chunks"]), len(res_graph["retrieved_chunks"]))

    def test_unanswerable_question_parity(self):
        """For unanswerable question, plain and langgraph produce identical refusal."""
        q = "How do pneumatic disc brakes operate in heavy commercial vehicles?"
        res_plain = self.plain_orch.run(question=q, user_id=1, user_chunks=self.chunks)
        res_graph = self.graph_orch.run(question=q, user_id=1, user_chunks=self.chunks)

        self.assertTrue(res_plain["refused"])
        self.assertTrue(res_graph["refused"])
        self.assertEqual(res_plain["final_response"], self.prompt_config.refusal_message)
        self.assertEqual(res_graph["final_response"], self.prompt_config.refusal_message)
        self.assertEqual(res_plain["final_response"], res_graph["final_response"])


class TestOrchestratorFallback(unittest.TestCase):
    """Test get_orchestrator factory fallback."""

    def test_default_returns_plain(self):
        pipeline = MagicMock(spec=RAGPipeline)
        orch = get_orchestrator(pipeline, "plain")
        self.assertIsInstance(orch, PlainOrchestrator)

    def test_fallback_when_langgraph_missing(self):
        pipeline = MagicMock(spec=RAGPipeline)
        orch = get_orchestrator(pipeline, "langgraph")
        if not HAS_LANGGRAPH:
            self.assertIsInstance(orch, PlainOrchestrator)
        else:
            self.assertIsInstance(orch, LangGraphOrchestrator)
