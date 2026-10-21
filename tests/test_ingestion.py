"""Unit tests for multi-format document ingestion (Markdown, TXT, Web) and SSRF validation."""

import unittest
from unittest.mock import patch, Mock
from src.processors.markdown_processor import MarkdownProcessor
from src.processors.web_processor import (
    validate_url_ssrf,
    extract_html_content,
    WebProcessor,
)
from src.processors.text_chunker import TextChunker
from src.rag.retriever import DocumentRetriever
from src.bot.handlers import BotHandlers


class TestMultiFormatIngestion(unittest.TestCase):
    """Test markdown, text, web ingestion and citation labels."""

    def setUp(self):
        self.retriever = DocumentRetriever(top_k=5, similarity_threshold=0.3)
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            llm_client=Mock(),
            config=Mock(),
        )

    def test_markdown_heading_chunking_and_labels(self):
        """Test markdown heading hierarchy extraction and label propagation."""
        md_text = """Introductory paragraph before any heading.

# Architecture
The architecture is modular.

## Storage Layer
SQLite is used for metadata.

### Vector Storage
NumPy or Chroma backends store vectors.

## Ingestion Layer
Markdown and web pages are parsed.
"""
        processor = MarkdownProcessor(chunk_size=1000, overlap=200)
        chunks = processor.process_markdown(md_text, source_name="README.md")

        self.assertGreaterEqual(len(chunks), 4)
        labels = [c.section_label for c in chunks]
        self.assertIn("Overview", labels)
        self.assertIn("Architecture", labels)
        self.assertIn("Architecture > Storage Layer", labels)
        self.assertIn("Architecture > Storage Layer > Vector Storage", labels)
        self.assertIn("Architecture > Ingestion Layer", labels)

        for c in chunks:
            self.assertEqual(c.source_type, "markdown")
            self.assertEqual(c.source_name, "README.md")
            self.assertIsNotNone(c.content)

    def test_markdown_long_section_chunking(self):
        """Test that a section exceeding chunk_size is split while keeping heading label."""
        long_paragraph = "This is a detailed sentence about TeleRAG RAG architecture. " * 30  # ~1800 chars
        md_text = f"# Big Chapter\n\n{long_paragraph}"

        processor = MarkdownProcessor(chunk_size=500, overlap=100)
        chunks = processor.process_markdown(md_text, source_name="big.md")

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c.section_label, "Big Chapter")
            self.assertEqual(c.source_type, "markdown")

    def test_txt_ingestion(self):
        """Test plain text chunking and metadata."""
        txt_text = "Line 1: TeleRAG supports text files.\nLine 2: Simple chunking is applied."
        chunker = TextChunker(chunk_size=1000, overlap=200)
        chunks = chunker.create_chunks(txt_text)

        self.assertEqual(len(chunks), 1)
        self.assertIn("TeleRAG supports text files", chunks[0].content)

    def test_citation_labels_formatting(self):
        """Test retriever context formatting and handler source footer across all types."""
        chunks = [
            {
                "filename": "guide.pdf",
                "source_type": "pdf",
                "page_start": 3,
                "page_end": 4,
                "content": "PDF content on pages 3-4."
            },
            {
                "filename": "README.md",
                "source_type": "markdown",
                "section_label": "Setup > Installation",
                "content": "Run pip install -r requirements.txt."
            },
            {
                "filename": "Python Docs (docs.python.org)",
                "source_type": "web",
                "section_label": "Python Docs (docs.python.org)",
                "content": "Python 3.12 release notes."
            },
            {
                "filename": "notes.txt",
                "source_type": "text",
                "content": "Plain text note."
            }
        ]

        context = self.retriever.format_context(chunks)
        self.assertIn("[guide.pdf, p. 3-4]\nPDF content on pages 3-4.", context)
        self.assertIn("[README.md, Setup > Installation]\nRun pip install -r requirements.txt.", context)
        self.assertIn("[Python Docs (docs.python.org)]\nPython 3.12 release notes.", context)
        self.assertIn("[notes.txt]\nPlain text note.", context)

        footer = self.handlers._format_source_footer(chunks)
        self.assertIn("• guide.pdf (p. 3-4)", footer)
        self.assertIn("• README.md (Setup > Installation)", footer)
        self.assertIn("• Python Docs (docs.python.org)", footer)
        self.assertIn("• notes.txt", footer)

    def test_html_extraction_from_fixture(self):
        """Test HTML stripping and content extraction from static fixture string."""
        html_fixture = """<!DOCTYPE html>
<html>
<head>
    <title>TeleRAG Documentation</title>
    <style>body { font-family: sans-serif; }</style>
    <script>console.log("should be stripped");</script>
</head>
<body>
    <header><nav><a href="/">Home</a></nav></header>
    <main>
        <h1>System Architecture</h1>
        <p>TeleRAG is a hybrid RAG Telegram bot built for personal and small-team use.</p>
        <p>It supports hybrid search and cross-encoder reranking.</p>
    </main>
    <footer>Copyright 2026 TeleRAG</footer>
</body>
</html>"""
        title, domain, clean_text = extract_html_content(html_fixture, "https://telerag.org/docs")
        self.assertEqual(title, "TeleRAG Documentation")
        self.assertEqual(domain, "telerag.org")
        self.assertIn("System Architecture", clean_text)
        self.assertIn("TeleRAG is a hybrid RAG Telegram bot", clean_text)
        self.assertNotIn("console.log", clean_text)
        self.assertNotIn("Home", clean_text)  # nav stripped
        self.assertNotIn("Copyright", clean_text)  # footer stripped


class TestSSRFValidation(unittest.TestCase):
    """Test SSRF protection rules across dangerous IPs and schemes."""

    def test_url_validator_rejects_loopback_ip(self):
        is_safe, msg = validate_url_ssrf("http://127.0.0.1/admin")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    def test_url_validator_rejects_localhost(self):
        is_safe, msg = validate_url_ssrf("http://localhost:8080/api")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    def test_url_validator_rejects_private_class_a(self):
        is_safe, msg = validate_url_ssrf("http://10.0.0.1/secret")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    def test_url_validator_rejects_cloud_metadata(self):
        is_safe, msg = validate_url_ssrf("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    def test_url_validator_rejects_file_scheme(self):
        is_safe, msg = validate_url_ssrf("file:///etc/passwd")
        self.assertFalse(is_safe)
        self.assertIn("only http and https urls are supported", msg.lower())

    def test_url_validator_rejects_ftp_scheme(self):
        is_safe, msg = validate_url_ssrf("ftp://example.com/files")
        self.assertFalse(is_safe)
        self.assertIn("only http and https urls are supported", msg.lower())

    @patch("src.processors.web_processor.requests.get")
    @patch("src.processors.web_processor.validate_url_ssrf")
    def test_redirect_to_private_ip_is_blocked(self, mock_validate, mock_get):
        """Test that a redirect pointing to a private IP is intercepted and blocked."""
        # Initial URL is safe, redirect target is unsafe
        mock_validate.side_effect = [
            (True, ""),  # initial URL
            (False, "Access to private network address is forbidden.")  # redirect target
        ]

        mock_resp = Mock()
        mock_resp.is_redirect = True
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "http://192.168.1.1/admin"}
        mock_get.return_value = mock_resp

        processor = WebProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.fetch_url("https://public-site.com/bounce")

        self.assertIn("Redirect blocked by SSRF policy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
