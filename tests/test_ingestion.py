"""Unit tests for multi-format document ingestion, SSRF validation, and safety."""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pypdf import PdfWriter

from src.bot.handlers import BotHandlers
from src.config import Config
from src.processors.markdown_processor import MarkdownProcessor
from src.processors.pdf_processor import PDFProcessor
from src.processors.text_chunker import TextChunk, TextChunker
from src.processors.web_processor import (
    WebProcessor,
    build_web_document_name,
    extract_html_content,
    validate_url_ssrf,
)
from src.rag.retriever import DocumentRetriever
from src.storage.document_store import sanitize_filename


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
            config=Config(telegram_bot_token="test"),
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
        long_paragraph = (
            "This is a detailed sentence about TeleRAG RAG architecture. " * 30
        )  # ~1800 chars
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
                "content": "PDF content on pages 3-4.",
            },
            {
                "filename": "README.md",
                "source_type": "markdown",
                "section_label": "Setup > Installation",
                "content": "Run pip install -r requirements.txt.",
            },
            {
                "filename": "Python Docs (docs.python.org)",
                "source_type": "web",
                "section_label": "Python Docs (docs.python.org)",
                "content": "Python 3.12 release notes.",
            },
            {"filename": "notes.txt", "source_type": "text", "content": "Plain text note."},
        ]

        context = self.retriever.format_context(chunks)
        self.assertIn("[guide.pdf, p. 3-4]\nPDF content on pages 3-4.", context)
        self.assertIn(
            "[README.md, Setup > Installation]\nRun pip install -r requirements.txt.", context
        )
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
            (False, "Access to private network address is forbidden."),  # redirect target
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

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_web_processor_decimal_hex_octal_ip_blocked(self, mock_getaddrinfo):
        """Decimal, hex, and octal integer formats resolving to 127.0.0.1 must be blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]
        variants = [
            "http://2130706433/page",
            "http://0x7f000001/page",
            "http://017700000001/page",
        ]
        for url in variants:
            is_safe, msg = validate_url_ssrf(url)
            self.assertFalse(is_safe)
            self.assertIn("forbidden", msg.lower())

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_web_processor_mixed_dns_answer_blocked(self, mock_getaddrinfo):
        """Resolver answer list [public, private] must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
            (2, 1, 6, "", ("10.0.0.1", 80)),
        ]
        is_safe, msg = validate_url_ssrf("https://dual.example.com/page")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    @patch("src.processors.web_processor.requests.get")
    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_web_processor_redirect_to_metadata_blocked(self, mock_getaddrinfo, mock_get):
        """A redirect to 169.254.169.254 must be blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]
        mock_resp = Mock()
        mock_resp.is_redirect = True
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        mock_get.return_value = mock_resp

        processor = WebProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.fetch_url("https://example.com/safe")
        self.assertIn("Redirect blocked by SSRF policy", str(ctx.exception))

    @patch("src.processors.web_processor.requests.get")
    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_web_processor_redirect_loop_blocked(self, mock_getaddrinfo, mock_get):
        """A redirect loop must be aborted after max_redirects."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]
        mock_resp = Mock()
        mock_resp.is_redirect = True
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://example.com/loop"}
        mock_get.return_value = mock_resp

        processor = WebProcessor()
        with self.assertRaises(ValueError) as ctx:
            processor.fetch_url("https://example.com/loop")
        self.assertIn("Too many redirects", str(ctx.exception))

    def test_web_processor_credentials_in_url_blocked(self):
        """URLs containing credentials must be rejected."""
        is_safe, msg = validate_url_ssrf("http://user:pass@example.com/page")
        self.assertFalse(is_safe)
        self.assertIn("credentials", msg.lower())

    def test_web_processor_ipv4_mapped_ipv6_blocked(self):
        """IPv4-mapped IPv6 address ::ffff:127.0.0.1 must be blocked."""
        is_safe, msg = validate_url_ssrf("http://[::ffff:127.0.0.1]/page")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", msg.lower())

    def test_web_processor_link_local_multicast_reserved_blocked(self):
        """Link-local, multicast, and reserved IP ranges must be blocked."""
        restricted_urls = [
            "http://169.254.1.1/page",
            "http://224.0.0.1/page",
            "http://240.0.0.1/page",
            "http://0.0.0.0/page",
            "http://[fe80::1]/page",
            "http://[ff02::1]/page",
        ]
        for url in restricted_urls:
            is_safe, msg = validate_url_ssrf(url)
            self.assertFalse(is_safe, f"{url} should be blocked")
            self.assertIn("forbidden", msg.lower())


class TestWebDocumentNaming(unittest.TestCase):
    """Test URL ingestion document naming rules."""

    def test_build_web_document_name_from_title(self):
        name = build_web_document_name(
            "https://example.com/blog/intro", title="Introduction to TeleRAG"
        )
        self.assertEqual(name, "Introduction to TeleRAG")
        self.assertNotIn("(example.com)", name)

    def test_build_web_document_name_from_path_when_no_title(self):
        name = build_web_document_name("https://example.com/docs/api/v1/overview", title="")
        self.assertEqual(name, "docs/api/v1/overview")
        self.assertNotIn("(example.com)", name)

    def test_build_web_document_name_from_path_when_title_matches_domain(self):
        name = build_web_document_name("https://example.com/guides/setup.html", title="example.com")
        self.assertEqual(name, "guides/setup.html")
        self.assertNotIn("(example.com)", name)

    def test_build_web_document_name_trimmed_to_60_chars(self):
        long_title = "A" * 80
        name = build_web_document_name("https://example.com/page", title=long_title)
        self.assertEqual(len(name), 60)
        self.assertEqual(name, "A" * 60)

    @patch.object(WebProcessor, "fetch_url")
    def test_process_url_sets_citation_label_without_domain_suffix(self, mock_fetch):
        mock_fetch.return_value = (
            "Getting Started with Vector Search",
            "docs.example.com",
            "This is a paragraph about vector search.",
            b"fake html bytes",
        )
        processor = WebProcessor(chunk_size=500, overlap=50)
        label, clean_text, raw_bytes, chunks = processor.process_url(
            "https://docs.example.com/guide"
        )
        self.assertEqual(label, "Getting Started with Vector Search")
        self.assertNotIn("(docs.example.com)", label)
        self.assertTrue(all(c.source_name == label for c in chunks))


class TestFilenameSanitization(unittest.TestCase):
    """Test filename sanitization against traversal, reserved names, unicode, and limits."""

    def test_path_traversal_elimination(self):
        self.assertEqual(sanitize_filename("../../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(sanitize_filename("a/b/c/doc.pdf"), "doc.pdf")
        self.assertEqual(sanitize_filename(r"a\b\c\doc.txt"), "doc.txt")
        self.assertEqual(sanitize_filename(".."), "document")
        self.assertEqual(sanitize_filename("."), "document")

    def test_invalid_characters(self):
        sanitized = sanitize_filename('file:name*with?chars"and<bars>|here.pdf')
        self.assertEqual(sanitized, "file_name_with_chars_and_bars__here.pdf")
        self.assertNotIn(":", sanitized)
        self.assertNotIn("*", sanitized)
        self.assertNotIn("?", sanitized)
        self.assertNotIn('"', sanitized)
        self.assertNotIn("<", sanitized)
        self.assertNotIn(">", sanitized)
        self.assertNotIn("|", sanitized)

        # Control characters
        self.assertEqual(sanitize_filename("doc\x00\x1f\x7fname.txt"), "docname.txt")

    def test_windows_reserved_device_names(self):
        self.assertEqual(sanitize_filename("CON.pdf"), "_CON.pdf")
        self.assertEqual(sanitize_filename("con.pdf"), "_con.pdf")
        self.assertEqual(sanitize_filename("NUL.txt"), "_NUL.txt")
        self.assertEqual(sanitize_filename("prn.md"), "_prn.md")
        self.assertEqual(sanitize_filename("aux"), "_aux")
        self.assertEqual(sanitize_filename("com1.pdf"), "_com1.pdf")
        self.assertEqual(sanitize_filename("lpt9.txt"), "_lpt9.txt")
        # Non-reserved stems containing reserved words are safe
        self.assertEqual(sanitize_filename("contrast.pdf"), "contrast.pdf")

    def test_unicode_preservation(self):
        self.assertEqual(sanitize_filename("документ_2026.pdf"), "документ_2026.pdf")
        self.assertEqual(sanitize_filename("报告_final.txt"), "报告_final.txt")
        self.assertEqual(sanitize_filename("résumé.md"), "résumé.md")

    def test_length_capping(self):
        long_name = "a" * 300 + ".pdf"
        sanitized = sanitize_filename(long_name, max_length=120)
        self.assertLessEqual(len(sanitized), 120)
        self.assertTrue(sanitized.endswith(".pdf"))

        long_stem_no_ext = "b" * 300
        sanitized_no_ext = sanitize_filename(long_stem_no_ext, max_length=120)
        self.assertEqual(len(sanitized_no_ext), 120)

    def test_edge_cases(self):
        self.assertEqual(sanitize_filename(""), "document")
        self.assertEqual(sanitize_filename(None), "document")
        self.assertEqual(sanitize_filename("..."), "document")
        self.assertEqual(sanitize_filename(".pdf"), ".pdf")


class TestIngestionSafety(unittest.IsolatedAsyncioTestCase):
    """Test ingestion behavior on edge cases and safety constraints."""

    def setUp(self):
        self.handlers = BotHandlers(
            pdf_processor=PDFProcessor(),
            text_chunker=TextChunker(chunk_size=500, overlap=50),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            llm_client=Mock(),
            config=Config(telegram_bot_token="test", max_chunks_per_document=5),
        )

    async def test_handle_encrypted_pdf_rejected(self):
        """Encrypted PDF upload is cleanly rejected with user-facing message."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            enc_path = os.path.join(tmp_dir, "enc.pdf")
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.encrypt("super_secret")
            with open(enc_path, "wb") as f:
                writer.write(f)

            update = Mock()
            update.effective_user.id = 42
            update.message.document.file_name = "enc.pdf"
            update.message.document.file_size = os.path.getsize(enc_path)
            update.message.reply_text = AsyncMock()

            async def fake_download(dest):
                with open(enc_path, "rb") as src_f, open(dest, "wb") as dst_f:
                    dst_f.write(src_f.read())

            file_mock = Mock()
            file_mock.download_to_drive = AsyncMock(side_effect=fake_download)
            update.message.document.get_file = AsyncMock(return_value=file_mock)

            context = Mock()
            context.bot.send_chat_action = AsyncMock()

            await self.handlers.handle_document(update, context)

            # Assert rejection message sent
            reply_calls = [c.args[0] for c in update.message.reply_text.call_args_list]
            self.assertTrue(any("Encrypted PDF" in msg for msg in reply_calls))
            self.handlers.document_store.store_document.assert_not_called()

    async def test_handle_pdf_with_no_extractable_text_rejected(self):
        """PDF with no extractable text is rejected without crashing."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            blank_path = os.path.join(tmp_dir, "blank.pdf")
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(blank_path, "wb") as f:
                writer.write(f)

            update = Mock()
            update.effective_user.id = 42
            update.message.document.file_name = "blank.pdf"
            update.message.document.file_size = os.path.getsize(blank_path)
            update.message.reply_text = AsyncMock()

            async def fake_download(dest):
                with open(blank_path, "rb") as src_f, open(dest, "wb") as dst_f:
                    dst_f.write(src_f.read())

            file_mock = Mock()
            file_mock.download_to_drive = AsyncMock(side_effect=fake_download)
            update.message.document.get_file = AsyncMock(return_value=file_mock)

            context = Mock()
            context.bot.send_chat_action = AsyncMock()

            with self.assertLogs("src.processors.pdf_processor", level="ERROR"):
                await self.handlers.handle_document(update, context)

            reply_calls = [c.args[0] for c in update.message.reply_text.call_args_list]
            self.assertTrue(
                any("No text content" in msg for msg in reply_calls),
                f"Expected 'No text content' in replies: {reply_calls}",
            )
            self.handlers.document_store.store_document.assert_not_called()

    async def test_handle_document_over_max_chunks_limit(self):
        """Document creating more than max_chunks_per_document is halted with error message."""
        self.handlers.pdf_processor = Mock()
        self.handlers.pdf_processor.validate_pdf.return_value = (True, None)
        self.handlers.pdf_processor.extract_text_with_pages.return_value = [
            (1, "Page 1"),
            (2, "Page 2"),
        ]
        # Text chunker returns 10 chunks, while limit is 5
        self.handlers.text_chunker = Mock()
        self.handlers.text_chunker.create_chunks_with_pages.return_value = [
            TextChunk(content=f"Chunk {i}", chunk_index=i, start_pos=0, end_pos=10, char_count=10)
            for i in range(10)
        ]

        update = Mock()
        update.effective_user.id = 42
        update.message.document.file_name = "large.pdf"
        update.message.document.file_size = 1000
        update.message.reply_text = AsyncMock()

        file_mock = Mock()
        file_mock.download_to_drive = AsyncMock()
        update.message.document.get_file = AsyncMock(return_value=file_mock)

        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        await self.handlers.handle_document(update, context)

        reply_calls = [c.args[0] for c in update.message.reply_text.call_args_list]
        self.assertTrue(any("exceeds the maximum limit of 5 chunks" in msg for msg in reply_calls))
        self.handlers.embedding_generator.generate_embeddings.assert_not_called()
        self.handlers.document_store.store_document.assert_not_called()

    async def test_uploaded_filenames_sanitization_and_isolation(self):
        """Uploaded filenames containing traversal, Windows devices, unicode, and 300 chars are sanitized."""
        test_filenames = [
            ("../../etc/passwd.pdf", "passwd.pdf"),
            (r"..\..\system32\cmd.pdf", "cmd.pdf"),
            ("CON.pdf", "_CON.pdf"),
            ("NUL.txt", "_NUL.txt"),
            ("документ_2026.pdf", "документ_2026.pdf"),
            ("a" * 300 + ".pdf", ("a" * 116 + ".pdf")),
        ]

        self.handlers.pdf_processor = Mock()
        self.handlers.pdf_processor.validate_pdf.return_value = (True, None)
        self.handlers.pdf_processor.extract_text_with_pages.return_value = [(1, "Safe content")]
        self.handlers.text_chunker = Mock()
        self.handlers.text_chunker.create_chunks_with_pages.return_value = [
            TextChunk(content="Safe chunk", chunk_index=0, start_pos=0, end_pos=10, char_count=10)
        ]
        self.handlers.text_chunker.create_chunks.return_value = [
            Mock(content="Text chunk", chunk_index=0, start_pos=0, end_pos=10, char_count=10)
        ]
        self.handlers.embedding_generator.generate_embeddings.return_value = [[0.1] * 384]

        context = Mock()
        context.bot.send_chat_action = AsyncMock()

        for raw_name, expected_stored_name in test_filenames:
            self.handlers.document_store.store_document.reset_mock()
            update = Mock()
            update.effective_user.id = 42
            update.message.document.file_name = raw_name
            update.message.document.file_size = 100
            update.message.reply_text = AsyncMock()

            def make_download(text_bytes):
                async def fake_download(dest):
                    with open(dest, "wb") as f:
                        f.write(text_bytes)

                return fake_download

            file_mock = Mock()
            file_mock.download_to_drive = AsyncMock(side_effect=make_download(b"test file content"))
            update.message.document.get_file = AsyncMock(return_value=file_mock)

            await self.handlers.handle_document(update, context)

            self.handlers.document_store.store_document.assert_called_once()
            called_kwargs = self.handlers.document_store.store_document.call_args.kwargs
            stored_filename = called_kwargs.get("filename")
            self.assertEqual(stored_filename, expected_stored_name)
            self.assertNotIn("/", stored_filename)
            self.assertNotIn("\\", stored_filename)
            self.assertNotIn("..", stored_filename)


if __name__ == "__main__":
    unittest.main()
