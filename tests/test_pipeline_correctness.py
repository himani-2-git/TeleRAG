"""Tests for pipeline correctness: summary requests, partial answers and grounding errors."""

import unittest
from unittest.mock import MagicMock, patch

from src.rag.pipeline import (
    PlainOrchestrator,
    RAGPipeline,
    is_insufficient_context,
    is_summary_query,
    select_summary_chunks,
)
from src.rag.prompts import load_prompt_config
from src.rag.retriever import DocumentRetriever

TOPICS = ("fee", "refund", "weather")


class TopicEmbedder:
    """Fake embedder with one dimension per topic word and a last dimension for other text."""

    def generate_embeddings(self, text):
        lower = text.lower()
        vector = [1.0 if topic in lower else 0.0 for topic in TOPICS]
        if not any(vector):
            return [0.0] * len(TOPICS) + [1.0]
        return vector + [0.0]

    def generate_query_embedding(self, query):
        return self.generate_embeddings(query)


class ScriptedLLM:
    """Fake LLM that always returns the same answer."""

    def __init__(self, answer):
        self.answer = answer

    def generate_answer(self, question, context):
        return self.answer

    def validate_response(self, text):
        return text.strip() if text else ""


def make_fee_chunk():
    text = "The registration fee is five dollars."
    return {
        "document_id": 1,
        "chunk_index": 0,
        "content": text,
        "filename": "rules.pdf",
        "page_start": 1,
        "page_end": 1,
        "embedding": TopicEmbedder().generate_embeddings(text),
    }


def make_orchestrator(llm_answer, grounding_enforce=False):
    pipeline = RAGPipeline(
        retriever=DocumentRetriever(top_k=2, similarity_threshold=0.3, retrieval_mode="dense"),
        llm_client=ScriptedLLM(llm_answer),
        embedding_generator=TopicEmbedder(),
        prompt_config=load_prompt_config("v1"),
        grounding_enforce=grounding_enforce,
    )
    return PlainOrchestrator(pipeline)


class TestSummaryDetection(unittest.TestCase):
    """Only clear whole-document requests count as summary requests."""

    def test_clear_summary_requests_are_detected(self):
        for question in [
            "summarize",
            "Summarise this document",
            "Please summarize the entire book.",
            "Can you summarize this PDF?",
            "Give me a summary of the document",
            "What is the overview of this paper?",
            "What is this book about?",
            "Tell me about this document",
            "Can you summarize this document for me?",
            "Summarize this document, please",
            "Give me a short summary of the book briefly",
        ]:
            self.assertTrue(is_summary_query(question), question)

    def test_questions_that_only_mention_summary_words_are_not_summary_requests(self):
        for question in [
            "What is the summary of chapter 3?",
            "Summarize the Raft algorithm",
            "Give me an overview of recursion",
            "What are the main points of the leader election section?",
            "Is there a summary section in chapter 2?",
            "Explain the overview effect",
            "How do I summarize a list in Python?",
        ]:
            self.assertFalse(is_summary_query(question), question)

    def test_non_summary_question_goes_through_normal_retrieval(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=ScriptedLLM("x"),
            embedding_generator=TopicEmbedder(),
            prompt_config=load_prompt_config("v1"),
        )

        pipeline.node_retrieve(
            {
                "question": "What is the summary of chapter 3?",
                "user_id": 1,
                "user_chunks": [make_fee_chunk()],
            }
        )

        retriever.retrieve.assert_called_once()


class TestSummarySampling(unittest.TestCase):
    """Summaries must be built from the whole document, not only its first pages."""

    @staticmethod
    def make_chunks(count, document_id=1):
        return [
            {
                "document_id": document_id,
                "chunk_index": i,
                "page_start": i + 1,
                "content": f"chunk {i}",
            }
            for i in range(count)
        ]

    def test_long_document_is_sampled_from_start_middle_and_end(self):
        chunks = self.make_chunks(100)

        selected = select_summary_chunks(chunks, max_chunks=8)

        indexes = [c["chunk_index"] for c in selected]
        self.assertEqual(len(selected), 8)
        self.assertEqual(indexes, sorted(set(indexes)), "must be unique and in reading order")
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 99)
        self.assertTrue(any(40 <= i <= 60 for i in indexes), "must include the middle")

    def test_short_document_returns_every_chunk_in_reading_order(self):
        chunks = list(reversed(self.make_chunks(5)))

        selected = select_summary_chunks(chunks, max_chunks=8)

        self.assertEqual([c["chunk_index"] for c in selected], [0, 1, 2, 3, 4])

    def test_documents_are_kept_in_upload_order(self):
        chunks = self.make_chunks(3, document_id=2) + self.make_chunks(3, document_id=1)

        selected = select_summary_chunks(chunks, max_chunks=8)

        self.assertEqual([c["document_id"] for c in selected], [1, 1, 1, 2, 2, 2])

    def test_node_retrieve_uses_sampling_for_summary_requests(self):
        pipeline = make_orchestrator("unused").pipeline

        state = pipeline.node_retrieve(
            {
                "question": "summarize this document",
                "user_id": 1,
                "user_chunks": self.make_chunks(100),
            }
        )

        indexes = [c["chunk_index"] for c in state["retrieved_chunks"]]
        self.assertEqual(len(indexes), 8)
        self.assertEqual(indexes[-1], 99)


class TestPartialAnswers(unittest.TestCase):
    """A useful partial answer is kept; only answers with nothing grounded are refused."""

    def run_pipeline(self, llm_answer, **kwargs):
        orchestrator = make_orchestrator(llm_answer, **kwargs)
        return orchestrator.run(
            question="What is the fee?", user_id=1, user_chunks=[make_fee_chunk()]
        )

    def test_pure_refusal_sentences_are_detected(self):
        self.assertTrue(is_insufficient_context("The context does not mention this."))
        self.assertTrue(
            is_insufficient_context(
                "The context does not mention refunds. This is not mentioned in the context."
            )
        )

    def test_refusals_that_say_document_instead_of_context_are_detected(self):
        """The real model often says "the document does not ..." (seen in a live run)."""
        for answer in (
            "The document does not mention the refund policy.",
            "The document does not provide information about the refund policy.",
            "The provided document doesn't contain the answer.",
            "The documents do not mention this topic.",
            "The context does not specify the date.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(is_insufficient_context(answer))

    def test_partial_answer_that_says_document_is_kept(self):
        result = self.run_pipeline(
            "The registration fee is five dollars. "
            "However, the document does not provide information about the refund policy."
        )

        self.assertFalse(result["refused"])
        self.assertIn("five dollars", result["final_response"])

    def test_answer_with_real_content_is_not_a_pure_refusal(self):
        answer = "The fee is five dollars. The context does not mention refunds."
        self.assertFalse(is_insufficient_context(answer))

    def test_partial_answer_is_kept(self):
        result = self.run_pipeline(
            "The registration fee is five dollars. The context does not mention a refund policy."
        )

        self.assertFalse(result["refused"])
        self.assertIn("The registration fee is five dollars.", result["final_response"])
        self.assertIn("does not mention a refund policy", result["final_response"])
        self.assertIn("Sources:", result["final_response"])

    def test_refusal_marker_is_hidden_from_a_kept_partial_answer(self):
        result = self.run_pipeline("The registration fee is five dollars.\nINSUFFICIENT_CONTEXT")

        self.assertFalse(result["refused"])
        self.assertNotIn("INSUFFICIENT_CONTEXT", result["final_response"])
        self.assertIn("five dollars", result["final_response"])

    def test_answer_with_only_a_refusal_is_refused(self):
        result = self.run_pipeline("The context does not mention this.")

        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "INSUFFICIENT_CONTEXT")

    def test_marker_alone_is_refused(self):
        result = self.run_pipeline("INSUFFICIENT_CONTEXT")

        self.assertTrue(result["refused"])

    def test_refusal_mixed_with_ungrounded_filler_is_refused(self):
        result = self.run_pipeline(
            "Here is some filler text about nothing. INSUFFICIENT_CONTEXT. More filler follows."
        )

        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "INSUFFICIENT_CONTEXT")


class TestGroundingErrors(unittest.TestCase):
    """A failing grounding check must always be logged."""

    @patch("src.rag.pipeline.grounding_report", side_effect=RuntimeError("service down"))
    def test_error_is_logged_when_enforcement_is_on(self, _mock_report):
        orchestrator = make_orchestrator(
            "The registration fee is five dollars.", grounding_enforce=True
        )

        with self.assertLogs("src.rag.pipeline", level="ERROR") as logs:
            result = orchestrator.run(
                question="What is the fee?", user_id=1, user_chunks=[make_fee_chunk()]
            )

        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "GROUNDING_ERROR")
        self.assertTrue(any("Grounding check failed" in line for line in logs.output))

    @patch("src.rag.pipeline.grounding_report", side_effect=RuntimeError("service down"))
    def test_partial_answer_cannot_be_verified_so_it_is_refused(self, _mock_report):
        orchestrator = make_orchestrator(
            "The registration fee is five dollars. The context does not mention refunds."
        )

        with self.assertLogs("src.rag.pipeline", level="ERROR"):
            result = orchestrator.run(
                question="What is the fee?", user_id=1, user_chunks=[make_fee_chunk()]
            )

        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "GROUNDING_ERROR")


if __name__ == "__main__":
    unittest.main()
