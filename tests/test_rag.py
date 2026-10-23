"""Tests for RAG engine functionality."""

import unittest
from unittest.mock import Mock, patch

from src.rag.embeddings import EmbeddingGenerator
from src.rag.llm_client import LLMClient, LLMUnavailableError
from src.rag.retriever import DocumentRetriever


class TestEmbeddingGenerator(unittest.TestCase):
    """Test embedding generator."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a small model for testing
        self.generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    def test_generate_single_embedding(self):
        """Test generating single embedding."""
        text = "This is a test sentence."
        embedding = self.generator.generate_embeddings(text)

        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)

    def test_generate_multiple_embeddings(self):
        """Test generating multiple embeddings."""
        texts = ["First sentence.", "Second sentence."]
        embeddings = self.generator.generate_embeddings(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertIsInstance(embeddings[1], list)


class TestDocumentRetriever(unittest.TestCase):
    """Test document retriever."""

    def setUp(self):
        """Set up test fixtures."""
        self.retriever = DocumentRetriever(top_k=3, similarity_threshold=0.5)

    def test_search_similar(self):
        """Test similarity search."""
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        chunks = [
            {"content": "Test chunk 1", "embedding": [0.1, 0.2, 0.3, 0.4], "filename": "test.pdf"},
            {"content": "Test chunk 2", "embedding": [0.9, 0.8, 0.7, 0.6], "filename": "test.pdf"},
        ]

        results = self.retriever.search_similar(query_embedding, chunks)

        self.assertGreater(len(results), 0)
        self.assertIn("similarity_score", results[0])

    def test_format_context(self):
        """Test context formatting."""
        chunks = [{"content": "Test content", "filename": "test.pdf", "similarity_score": 0.95}]

        context = self.retriever.format_context(chunks)

        self.assertIn("Test content", context)
        self.assertIn("test.pdf", context)

    def test_search_similar_no_match_returns_empty(self):
        """Test that query with no chunks meeting threshold returns empty list (no fallback)."""
        # Query orthogonal to chunks
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        chunks = [
            {
                "content": "Unrelated content 1",
                "embedding": [0.0, 1.0, 0.0, 0.0],  # similarity = 0.0
                "filename": "test.pdf",
            },
            {
                "content": "Unrelated content 2",
                "embedding": [0.1, 0.9, 0.0, 0.0],  # similarity ~ 0.11 < 0.5 threshold
                "filename": "test.pdf",
            },
        ]

        results = self.retriever.search_similar(query_embedding, chunks)
        self.assertEqual(results, [])

    def test_search_similar_filters_below_threshold(self):
        """Test that chunks below threshold are filtered out and results are ranked descending."""
        query_embedding = [1.0, 0.0, 0.0, 0.0]
        chunks = [
            {
                "content": "Low similarity chunk",
                "embedding": [0.2, 0.8, 0.0, 0.0],  # below 0.5 threshold
                "filename": "test.pdf",
            },
            {
                "content": "High similarity chunk",
                "embedding": [0.95, 0.05, 0.0, 0.0],  # well above 0.5 threshold (~0.998)
                "filename": "test.pdf",
            },
            {
                "content": "Moderate similarity chunk",
                "embedding": [0.7, 0.3, 0.0, 0.0],  # above 0.5 threshold (~0.919)
                "filename": "test.pdf",
            },
        ]

        results = self.retriever.search_similar(query_embedding, chunks)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["content"], "High similarity chunk")
        self.assertEqual(results[1]["content"], "Moderate similarity chunk")
        self.assertGreater(results[0]["similarity_score"], results[1]["similarity_score"])


class TestLLMClient(unittest.TestCase):
    """Test LLM client."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = LLMClient(api_key="test_key", model="test_model")

    def test_format_prompt(self):
        """Test prompt formatting."""
        question = "What is this about?"
        context = "This is test context."

        messages = self.client.format_prompt(question, context)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(question, messages[1]["content"])
        self.assertIn(context, messages[1]["content"])
        self.assertIn("INSUFFICIENT_CONTEXT", messages[0]["content"])

    def test_validate_response(self):
        """Test response validation."""
        response = "  Test response  "
        validated = self.client.validate_response(response)

        self.assertEqual(validated, "Test response")

    def test_validate_none_response(self):
        """Test validation of None response returns empty string."""
        validated = self.client.validate_response(None)
        self.assertEqual(validated, "")

    @patch("requests.post")
    def test_send_request_success(self, mock_post):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test answer"}}]}
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        response = self.client.send_request(messages)

        self.assertEqual(response, "Test answer")

    @patch("requests.post")
    def test_send_request_failure(self, mock_post):
        """Test failed API request."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        with self.assertLogs("src.rag.llm_client", level="ERROR"):
            response = self.client.send_request(messages)
        self.assertIsNone(response)

    @patch("requests.post")
    def test_send_request_model_not_found(self, mock_post):
        """Test API request when configured model is not found."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = '{"error": {"message": "Model not found"}}'
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        with self.assertLogs("src.rag.llm_client", level="ERROR"):
            response = self.client.send_request(messages)

        self.assertIsNone(response)
        # Verify it didn't retry 3 times on 404
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    def test_check_availability_success(self, mock_post):
        """Test check_availability when model is available (HTTP 200)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        is_ok, status_code, reason = self.client.check_availability()
        self.assertTrue(is_ok)
        self.assertEqual(status_code, 200)
        self.assertEqual(reason, "OK")

    @patch("requests.post")
    def test_check_availability_failure(self, mock_post):
        """Test check_availability when model returns 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = '{"error": {"message": "Model not found"}}'
        mock_post.return_value = mock_response

        is_ok, status_code, reason = self.client.check_availability()
        self.assertFalse(is_ok)
        self.assertEqual(status_code, 404)
        self.assertIn("Model not found", reason)

    @patch("requests.post")
    def test_custom_llm_base_url_and_no_key(self, mock_post):
        """Test client with custom LLM base URL (e.g. Ollama localhost) without an API key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Local answer"}}]}
        mock_post.return_value = mock_response

        client = LLMClient(api_key="", model="qwen2:7b", base_url="http://localhost:11434/v1")
        self.assertEqual(client.base_url, "http://localhost:11434/v1/chat/completions")

        messages = [{"role": "user", "content": "Test local"}]
        response = client.send_request(messages)

        self.assertEqual(response, "Local answer")
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        self.assertNotIn("Authorization", headers)
        self.assertEqual(mock_post.call_args[1]["timeout"], (5.0, 120.0))

    @patch("requests.post")
    def test_rate_limit_429_retries_at_most_once(self, mock_post):
        """Test that HTTP 429 rate limit is retried at most once and does not loop repeatedly."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = '{"error": {"message": "Rate limit reached"}}'
        mock_post.return_value = mock_response

        client = LLMClient(api_key="key", model="qwen", max_retries=5, retry_delay=0.01)
        messages = [{"role": "user", "content": "Test 429"}]

        with self.assertLogs("src.rag.llm_client", level="WARNING"):
            with patch("time.sleep") as mock_sleep:
                response = client.send_request(messages)

        self.assertIsNone(response)
        # Should attempt initial request + at most 1 retry = 2 total requests, not 5
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("requests.post")
    def test_send_request_includes_max_tokens_and_concise_prompt(self, mock_post):
        """Test that send_request passes max_tokens in payload and format_prompt includes 150-word constraint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Concise answer"}}]}
        mock_post.return_value = mock_response

        client = LLMClient(api_key="test_key", model="qwen2:7b", max_tokens=250)
        messages = client.format_prompt("What is recursion?", "Context text")

        # Verify prompt instructions
        self.assertIn("at most 150 words", messages[0]["content"])

        # Verify payload max_tokens
        client.send_request(messages)
        mock_post.assert_called_once()
        sent_payload = mock_post.call_args[1]["json"]
        self.assertEqual(sent_payload["max_tokens"], 250)

    @patch("requests.post")
    def test_generate_answer_failure_raises_llm_unavailable_error(self, mock_post):
        """Test that failure across all retries raises LLMUnavailableError."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = LLMClient(api_key="key", model="qwen", max_retries=3, retry_delay=0.01)
        with self.assertLogs("src.rag.llm_client", level="ERROR"):
            with patch("time.sleep"):
                with self.assertRaises(LLMUnavailableError):
                    client.generate_answer("What is this?", "Some context")
        self.assertEqual(mock_post.call_count, 3)

    @patch("requests.post")
    def test_exponential_backoff_delays(self, mock_post):
        """Test that retries use exponential backoff delay * 2 ** attempt."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = LLMClient(api_key="key", model="qwen", max_retries=3, retry_delay=1.5)
        with self.assertLogs("src.rag.llm_client", level="ERROR"):
            with patch("time.sleep") as mock_sleep:
                client.send_request([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_has_calls([unittest.mock.call(1.5), unittest.mock.call(3.0)])

    def test_refusal_path_produces_no_footer(self):
        """Test that LLMUnavailableError handled by pipeline produces no Sources footer."""
        from src.rag.pipeline import PlainOrchestrator, RAGPipeline
        from src.rag.prompts import load_prompt_config

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [{"content": "c", "filename": "doc.pdf", "page": 1}]
        mock_retriever.format_context.return_value = "c"
        mock_llm = Mock()
        mock_llm.generate_answer.side_effect = LLMUnavailableError("Service down")

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            embedding_generator=Mock(),
            prompt_config=load_prompt_config("v1"),
        )
        orch = PlainOrchestrator(pipeline)
        result = orch.run("test question", 1, [{"content": "c", "filename": "doc.pdf", "page": 1}])
        self.assertTrue(result["refused"])
        self.assertEqual(result["refusal_reason"], "LLM_UNAVAILABLE")
        self.assertNotIn("Sources:", result["final_response"])
        self.assertIn("unavailable", result["final_response"].lower())

    def test_config_reads_llm_env_vars(self):
        """Test that get_config reads LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY from environment."""
        import os

        from src.config import get_config

        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "test_token",
                "LLM_MODEL": "custom-model",
                "LLM_API_KEY": "custom-key",
                "LLM_BASE_URL": "http://localhost:11434/v1",
            },
            clear=True,
        ):
            cfg = get_config()
            self.assertEqual(cfg.llm_model, "custom-model")
            self.assertEqual(cfg.llm_api_key, "custom-key")
            self.assertEqual(cfg.llm_base_url, "http://localhost:11434/v1")

    def test_extract_response_length_finish_reason_appends_cutoff_notice(self):
        """Test that finish_reason='length' appends truncation notice."""
        data = {
            "choices": [
                {
                    "message": {"content": "Partial sentence that gets cut"},
                    "finish_reason": "length",
                }
            ]
        }
        res = self.client._extract_response(data)
        self.assertIn("Partial sentence that gets cut", res)
        self.assertTrue(
            res.endswith("\n\n(Answer was cut off; ask a narrower question for the full detail.)")
        )

    def test_extract_response_stop_finish_reason_leaves_answer_unchanged(self):
        """Test that finish_reason='stop' leaves answer intact without note."""
        data = {
            "choices": [
                {
                    "message": {"content": "Complete sentence."},
                    "finish_reason": "stop",
                }
            ]
        }
        res = self.client._extract_response(data)
        self.assertEqual(res, "Complete sentence.")

    def test_config_llm_max_tokens_default_1024(self):
        """Test that default LLM_MAX_TOKENS is 1024 in Config and get_config."""
        import os

        from src.config import Config, get_config

        cfg = Config(telegram_bot_token="test")
        self.assertEqual(cfg.llm_max_tokens, 1024)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_MAX_TOKENS", None)
            loaded = get_config(require_telegram=False)
            self.assertEqual(loaded.llm_max_tokens, 1024)

    def test_strip_footers_removes_truncation_notice(self):
        """Test that strip_footers strips the truncation disclaimer before grounding check."""
        from src.rag.grounding import strip_footers

        text = (
            "Actual answer.\n\n(Answer was cut off; ask a narrower question for the full detail.)"
        )
        cleaned = strip_footers(text)
        self.assertEqual(cleaned, "Actual answer.")
