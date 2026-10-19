"""Unit tests for versioned prompts loader."""

import os
import tempfile
import unittest
import yaml

from src.rag.prompts import PromptConfig, load_prompt_config


class TestPrompts(unittest.TestCase):
    """Test suite for versioned prompt loading and formatting."""

    def test_load_v1_prompt(self):
        """Verify loading v1 prompt works and returns all required fields."""
        config = load_prompt_config("v1")
        self.assertIsInstance(config, PromptConfig)
        self.assertEqual(config.version, "v1")
        self.assertIn("INSUFFICIENT_CONTEXT", config.system_prompt)
        self.assertIn("{context}", config.user_template)
        self.assertIn("{question}", config.user_template)
        self.assertTrue(config.refusal_message.startswith("🤔"))
        self.assertEqual(config.insufficient_context_marker, "INSUFFICIENT_CONTEXT")

    def test_missing_file_raises(self):
        """Verify loading a nonexistent version raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as exc_info:
            load_prompt_config("v_nonexistent_9999")
        self.assertIn("not found", str(exc_info.exception).lower())

    def test_missing_keys_raises(self):
        """Verify missing required keys in YAML raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_yaml = os.path.join(tmpdir, "bad.yaml")
            with open(bad_yaml, "w", encoding="utf-8") as f:
                yaml.dump({"version": "bad", "system_prompt": "hello"}, f)

            with self.assertRaises(ValueError) as exc_info:
                load_prompt_config("bad", prompts_dir=tmpdir)
            self.assertIn("missing required keys", str(exc_info.exception).lower())

    def test_text_matches_original_code_strings(self):
        """Verify loaded v1 prompt strings exactly match what the codebase previously hardcoded."""
        config = load_prompt_config("v1")
        expected_system = (
            "You are a helpful assistant that answers questions based on provided document context. "
            "Use only the information from the context to answer questions. "
            "When citing information, cite the source in the format [<filename>, p. <page>]. "
            "When the provided context does not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT (and nothing else). "
            "Be concise and accurate. Answer in at most 150 words unless the user explicitly asks for detail."
        )
        expected_user = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        expected_refusal = "🤔 I couldn't find relevant information in your documents to answer this question."
        expected_marker = "INSUFFICIENT_CONTEXT"

        self.assertEqual(config.system_prompt, expected_system)
        self.assertEqual(config.user_template, expected_user)
        self.assertEqual(config.refusal_message, expected_refusal)
        self.assertEqual(config.insufficient_context_marker, expected_marker)

    def test_format_user_prompt(self):
        """Verify format_user_prompt formats context and question correctly."""
        config = load_prompt_config("v1")
        formatted = config.format_user_prompt(context="Alpha beta", question="What is alpha?")
        self.assertEqual(formatted, "Context:\nAlpha beta\n\nQuestion: What is alpha?\n\nAnswer:")


if __name__ == "__main__":
    unittest.main()
