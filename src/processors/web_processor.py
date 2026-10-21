"""Web page processor with SSRF protection, size caps, and static HTML extraction."""

import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .text_chunker import TextChunk, TextChunker

logger = logging.getLogger(__name__)


def validate_url_ssrf(url: str) -> tuple[bool, str]:
    """Validate URL against SSRF threats, non-HTTP schemes, and private IP ranges.

    Args:
        url: URL string to check.

    Returns:
        Tuple of (is_safe, error_message).
    """
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            return False, "Only HTTP and HTTPS URLs are supported."

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: missing hostname."

        lower_host = hostname.lower().strip("[]")
        if lower_host in ("localhost", "localhost.localdomain"):
            return False, "Access to localhost or loopback addresses is forbidden."

        # Check if the hostname itself is a literal IP address
        try:
            direct_ip = ipaddress.ip_address(lower_host)
            if str(direct_ip) == "169.254.169.254":
                return False, "Access to cloud metadata service (169.254.169.254) is forbidden."
            if (
                direct_ip.is_private
                or direct_ip.is_loopback
                or direct_ip.is_link_local
                or direct_ip.is_multicast
                or direct_ip.is_reserved
                or direct_ip.is_unspecified
            ):
                return (
                    False,
                    f"Access to private, loopback, or restricted network address ({lower_host}) is forbidden.",
                )
        except ValueError:
            pass  # Hostname is a domain name, proceed to DNS resolution

        # Resolve DNS
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except (socket.gaierror, socket.herror, OSError) as e:
            logger.debug("DNS resolution failed for %s: %s", hostname, e)
            return False, f"Could not resolve hostname '{hostname}'."

        if not addr_info:
            return False, f"Could not resolve hostname '{hostname}'."

        for addr in addr_info:
            ip_str = addr[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False, f"Invalid IP address resolved: {ip_str}"

            if str(ip) == "169.254.169.254":
                return False, "Access to cloud metadata service (169.254.169.254) is forbidden."

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return (
                    False,
                    f"Access to private, loopback, or restricted network address ({ip_str}) is forbidden.",
                )

        return True, ""
    except (ValueError, TypeError) as e:
        logger.debug("URL validation error: %s", e)
        return False, "URL validation failed."


def extract_html_content(html: str, url: str) -> tuple[str, str, str]:
    """Extract clean title, domain, and main text from HTML string.

    Args:
        html: Raw HTML content.
        url: Original URL for fallback domain and title derivation.

    Returns:
        Tuple of (title, domain, clean_text).
    """
    soup = BeautifulSoup(html, "html.parser")
    domain = urlparse(url).netloc or "web"

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()
    if not title:
        title = domain

    # Remove non-content and layout tags
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe"]
    ):
        tag.decompose()

    # Prioritize semantic main content containers
    main_elem = soup.find("main") or soup.find("article") or soup.body or soup
    raw_text = main_elem.get_text(separator="\n", strip=True)
    clean_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    return title, domain, clean_text


class WebProcessor:
    """Handles fetching web pages with SSRF prevention and content chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        timeout: int = 15,
        max_bytes: int = 2 * 1024 * 1024,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    def fetch_url(self, url: str, max_redirects: int = 3) -> tuple[str, str, str, bytes]:
        """Fetch URL enforcing SSRF validation, size caps, and redirect hop checking.

        Args:
            url: URL to fetch.
            max_redirects: Maximum allowed redirect hops (default 3).

        Returns:
            Tuple of (title, domain, clean_text, raw_bytes).

        Raises:
            ValueError: If URL fails security validation, size caps, or content type constraints.
        """
        is_safe, error_msg = validate_url_ssrf(url)
        if not is_safe:
            raise ValueError(f"SSRF validation failed: {error_msg}")

        headers = {
            "User-Agent": "TeleRAG/1.0 (Web Ingestion Assistant)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }

        current_url = url
        redirect_count = 0

        while True:
            response = requests.get(
                current_url,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )

            # Handle redirects manually to re-verify every target hop
            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                redirect_count += 1
                if redirect_count > max_redirects:
                    response.close()
                    raise ValueError(f"Too many redirects (exceeded limit of {max_redirects}).")

                location = response.headers.get("Location")
                if not location:
                    response.close()
                    raise ValueError("Redirect response received without Location header.")

                next_url = urljoin(current_url, location)
                is_safe_next, next_err = validate_url_ssrf(next_url)
                if not is_safe_next:
                    response.close()
                    raise ValueError(f"Redirect blocked by SSRF policy: {next_err}")

                response.close()
                current_url = next_url
                continue

            response.raise_for_status()
            break

        # Validate content type is text/html
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            response.close()
            raise ValueError(
                f"Unsupported content type '{content_type}'. Only text/html is supported."
            )

        # Check content-length header if provided
        cl_header = response.headers.get("content-length")
        if cl_header:
            try:
                if int(cl_header) > self.max_bytes:
                    response.close()
                    raise ValueError(
                        f"Content length exceeds {self.max_bytes // (1024 * 1024)} MB limit."
                    )
            except ValueError:
                pass

        # Stream body with strict size check
        downloaded = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded.extend(chunk)
                    if len(downloaded) > self.max_bytes:
                        raise ValueError(
                            f"Downloaded content exceeds {self.max_bytes // (1024 * 1024)} MB limit."
                        )
        finally:
            response.close()

        raw_bytes = bytes(downloaded)
        html_str = raw_bytes.decode("utf-8", errors="replace")
        title, domain, clean_text = extract_html_content(html_str, current_url)

        if not clean_text:
            raise ValueError("No readable text content extracted from web page.")

        return title, domain, clean_text, raw_bytes

    def process_url(self, url: str) -> tuple[str, str, bytes, list[TextChunk]]:
        """Fetch, extract, and chunk web page content.

        Args:
            url: Web page URL.

        Returns:
            Tuple of (citation_label, clean_text, raw_bytes, chunks).
        """
        title, domain, clean_text, raw_bytes = self.fetch_url(url)
        citation_label = f"{title} ({domain})"
        raw_chunks = self.chunker.create_chunks(clean_text)

        chunks: list[TextChunk] = []
        for i, rc in enumerate(raw_chunks):
            chunk = TextChunk(
                content=rc.content,
                chunk_index=i,
                start_pos=rc.start_pos,
                end_pos=rc.end_pos,
                char_count=rc.char_count,
                source_type="web",
                source_name=citation_label,
                section_label=citation_label,
            )
            chunks.append(chunk)

        logger.info(f"Processed web page '{citation_label}' into {len(chunks)} chunks")
        return citation_label, clean_text, raw_bytes, chunks
