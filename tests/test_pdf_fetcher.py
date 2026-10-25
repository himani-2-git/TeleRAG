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

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_fetch_pdf_decimal_hex_octal_ip_blocked(self, mock_getaddrinfo):
        """Decimal, hex, and octal integer formats resolving to 127.0.0.1 must be blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 80))]
        variants = [
            "http://2130706433/doc.pdf",
            "http://0x7f000001/doc.pdf",
            "http://017700000001/doc.pdf",
        ]
        for url in variants:
            with self.assertRaises(ValueError) as ctx:
                fetch_pdf_bytes(url, max_bytes=1024)
            self.assertIn("Forbidden URL", str(ctx.exception))

    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_fetch_pdf_mixed_dns_answer_blocked(self, mock_getaddrinfo):
        """DNS returning both public and private addresses must be blocked."""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
            (2, 1, 6, "", ("10.0.0.1", 80)),
        ]
        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://dual.example.com/doc.pdf", max_bytes=1024)
        self.assertIn("Forbidden URL", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.requests.get")
    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_fetch_pdf_redirect_to_metadata_blocked(self, mock_getaddrinfo, mock_get):
        """A redirect to 169.254.169.254 must be blocked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]
        mock_resp = Mock()
        mock_resp.is_redirect = True
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/safe.pdf", max_bytes=1024)
        self.assertIn("Redirect blocked", str(ctx.exception))

    @patch("src.processors.pdf_fetcher.requests.get")
    @patch("src.processors.web_processor.socket.getaddrinfo")
    def test_fetch_pdf_redirect_loop(self, mock_getaddrinfo, mock_get):
        """A redirect loop must be aborted after max_redirects."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 80))]
        mock_resp = Mock()
        mock_resp.is_redirect = True
        mock_resp.status_code = 302
        mock_resp.headers = {"Location": "https://example.com/loop"}
        mock_get.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("https://example.com/loop", max_bytes=1024)
        self.assertIn("Too many redirects", str(ctx.exception))

    def test_fetch_pdf_credentials_in_url_blocked(self):
        """URLs containing credentials must be rejected before any network request."""
        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("http://user:pass@example.com/doc.pdf", max_bytes=1024)
        self.assertIn("Forbidden URL", str(ctx.exception))

    def test_fetch_pdf_ipv4_mapped_ipv6_blocked(self):
        """IPv4-mapped IPv6 address ::ffff:127.0.0.1 must be blocked."""
        with self.assertRaises(ValueError) as ctx:
            fetch_pdf_bytes("http://[::ffff:127.0.0.1]/doc.pdf", max_bytes=1024)
        self.assertIn("Forbidden URL", str(ctx.exception))

    def test_fetch_pdf_link_local_multicast_reserved_blocked(self):
        """Link-local, multicast, and reserved IP ranges must be blocked."""
        restricted_urls = [
            "http://169.254.1.1/doc.pdf",
            "http://224.0.0.1/doc.pdf",
            "http://240.0.0.1/doc.pdf",
            "http://0.0.0.0/doc.pdf",
            "http://[fe80::1]/doc.pdf",
            "http://[ff02::1]/doc.pdf",
        ]
        for url in restricted_urls:
            with self.assertRaises(ValueError) as ctx:
                fetch_pdf_bytes(url, max_bytes=1024)
            self.assertIn("Forbidden URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
