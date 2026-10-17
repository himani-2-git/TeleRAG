"""Tests for bot handlers and security validations."""

import unittest
from unittest.mock import Mock, AsyncMock, patch
from src.bot.handlers import BotHandlers

class DummyConfig:
    """Mock configuration for testing."""
    max_file_size_mb = 20

class TestBotHandlersSSRF(unittest.IsolatedAsyncioTestCase):
    """Test SSRF validation and redirect handling in BotHandlers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DummyConfig()
        self.handlers = BotHandlers(
            pdf_processor=Mock(),
            text_chunker=Mock(),
            document_store=Mock(),
            embedding_generator=Mock(),
            retriever=Mock(),
            openrouter_client=Mock(),
            config=self.config
        )
    
    def test_non_http_scheme_rejected(self):
        """Test that non-http/https schemes (ftp, file, gopher, etc.) are rejected."""
        schemes = [
            "ftp://example.com/doc.pdf",
            "file:///etc/passwd",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8="
        ]
        for url in schemes:
            is_safe, error = self.handlers._is_safe_url(url)
            self.assertFalse(is_safe, f"Scheme in {url} should be rejected")
            self.assertIn("Only HTTP and HTTPS URLs are supported", error)
    
    def test_private_ip_literal_rejected(self):
        """Test that private, loopback, and link-local IP literals are rejected."""
        private_urls = [
            "http://127.0.0.1/test.pdf",
            "http://10.0.0.1/test.pdf",
            "http://192.168.1.1/test.pdf",
            "http://172.16.0.1/test.pdf",
            "http://169.254.169.254/latest/meta-data/"
        ]
        for url in private_urls:
            is_safe, error = self.handlers._is_safe_url(url)
            self.assertFalse(is_safe, f"URL {url} should be rejected")
            self.assertIn("forbidden", error.lower())
    
    @patch('src.bot.handlers.socket.getaddrinfo')
    def test_domain_resolving_to_private_ip_rejected(self, mock_getaddrinfo):
        """Test that domain names resolving to internal IPs are rejected."""
        mock_getaddrinfo.return_value = [
            (2, 1, 0, '', ('127.0.0.1', 80))
        ]
        is_safe, error = self.handlers._is_safe_url("http://internal.service.local/test.pdf")
        self.assertFalse(is_safe)
        self.assertIn("forbidden", error.lower())
    
    @patch('src.bot.handlers.requests.get')
    async def test_redirect_to_loopback_rejected(self, mock_requests_get):
        """Test that redirects to 127.0.0.1 are detected and blocked."""
        # Initial request returns a 302 redirect to 127.0.0.1
        mock_redirect = Mock()
        mock_redirect.status_code = 302
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://127.0.0.1/secret.pdf"}
        mock_redirect.close = Mock()
        
        mock_requests_get.return_value = mock_redirect
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url') as mock_safe:
            # First hop is safe, second hop (127.0.0.1) is unsafe
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden.")
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/download.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        self.assertEqual(mock_requests_get.call_count, 1)
        mock_redirect.close.assert_called()
    
    @patch('src.bot.handlers.requests.get')
    async def test_redirect_to_cloud_metadata_rejected(self, mock_requests_get):
        """Test that redirects to 169.254.169.254 are detected and blocked."""
        mock_redirect = Mock()
        mock_redirect.status_code = 301
        mock_redirect.is_redirect = True
        mock_redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data/credentials.pdf"}
        mock_redirect.close = Mock()
        
        mock_requests_get.return_value = mock_redirect
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url') as mock_safe:
            mock_safe.side_effect = [
                (True, ""),
                (False, "Access to private or local network resources is forbidden.")
            ]
            await self.handlers._handle_url_upload(update, context, "https://example.com/doc.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("Redirect blocked" in r and "forbidden" in r for r in replies))
        mock_redirect.close.assert_called()
    
    @patch('src.bot.handlers.requests.get')
    async def test_streaming_max_file_size_enforced(self, mock_requests_get):
        """Test that MAX_FILE_SIZE_MB is strictly enforced while streaming even without Content-Length."""
        self.config.max_file_size_mb = 1  # 1 MB limit
        limit_bytes = 1 * 1024 * 1024
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.is_redirect = False
        mock_response.headers = {'content-type': 'application/pdf'}  # No content-length header
        mock_response.close = Mock()
        # Stream chunks totaling over 1 MB (e.g., 3 chunks of 500KB)
        chunk = b"A" * (500 * 1024)
        mock_response.iter_content.return_value = [chunk, chunk, chunk]
        mock_requests_get.return_value = mock_response
        
        update = Mock()
        update.effective_user.id = 123
        update.effective_chat.id = 456
        update.message.reply_text = AsyncMock()
        context = Mock()
        context.bot.send_chat_action = AsyncMock()
        
        with patch.object(self.handlers, '_is_safe_url', return_value=(True, "")):
            await self.handlers._handle_url_upload(update, context, "https://example.com/large.pdf")
        
        update.message.reply_text.assert_called()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        self.assertTrue(any("File too large" in r for r in replies))
        mock_response.close.assert_called()

if __name__ == '__main__':
    unittest.main()
