"""Unit tests for safe PDF fetching."""

import unittest
from unittest.mock import Mock, patch

import requests

from src.processors.pdf_fetcher import fetch_pdf_bytes


class TestPDFFetcher(unittest.TestCase):
    """Test fetch_pdf_bytes SSRF and streaming size limits."""

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, ""))
    @patch("src.processors.pdf_fetcher.requests.get")
    def test_fetch_pdf_success(self, mock_get, mock_ssrf):
        """Test successful PDF byte fetching."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.is_redirect = False
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.iter_content.return_value = [b"%PDF-1.4 sample content"]
        mock_get.return_value = mock_resp

        data = fetch_pdf_bytes("https://example.com/doc.pdf", max_bytes=1024 * 1024)
        self.assertEqual(data, b"%PDF-1.4 sample content")

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(False, "Local address"))
    def test_fetch_pdf_ssrf_blocked(self, mock_ssrf):
        """Test initial URL SSRF rejection raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("http://127.0.0.1/doc.pdf", max_bytes=1024)
        self.assertIn("Forbidden URL", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, ""))
    @patch("src.processors.pdf_fetcher.requests.get")
    def test_fetch_pdf_invalid_content_type(self, mock_get, mock_ssrf):
        """Test non-PDF content type raises ValueError."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.is_redirect = False
        mock_resp.headers = {"content-type": "text/html"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/page", max_bytes=1024)
        self.assertIn("doesn't point to a PDF file", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, ""))
    @patch("src.processors.pdf_fetcher.requests.get")
    def test_fetch_pdf_too_many_redirects(self, mock_get, mock_ssrf):
        """Test exceeding max redirects raises ValueError."""
        mock_resp = Mock()
        mock_resp.status_code = 302
        mock_resp.is_redirect = True
        mock_resp.headers = {"Location": "https://example.com/redirect"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/doc.pdf", max_bytes=1024)
        self.assertIn("Too many redirects", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, ""))
    @patch("src.processors.pdf_fetcher.requests.get")
    def test_fetch_pdf_declared_size_exceeded(self, mock_get, mock_ssrf):
        """Test Content-Length exceeding max_bytes raises ValueError before streaming."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.is_redirect = False
        mock_resp.headers = {"content-type": "application/pdf", "content-length": "2048"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/large.pdf", max_bytes=1024)
        self.assertIn("File too large", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.validate_url_ssrf", return_value=(True, ""))
    @patch("src.processors.pdf_fetcher.requests.get")
    def test_fetch_pdf_timeout(self, mock_get, mock_ssrf):
        """Test requests timeout is converted to user-friendly ValueError."""
        mock_get.side_effect = requests.exceptions.Timeout("Timed out")
        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/doc.pdf", max_bytes=1024)
        self.assertIn("Download timeout", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
