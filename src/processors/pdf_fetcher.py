"""PDF fetcher with SSRF validation, redirect handling, and streaming size limits."""

import io
from urllib.parse import urljoin

import requests

from .web_processor import validate_url_ssrf


def fetch_pdf_bytes(url: str, max_bytes: int, timeout: int = 30) -> bytes:
    """Download PDF bytes safely with SSRF validation and streaming size cap.

    Args:
        url: URL of the PDF to download.
        max_bytes: Maximum allowed file size in bytes.
        timeout: Network request timeout in seconds.

    Returns:
        Downloaded PDF bytes.

    Raises:
        ValueError: On SSRF failure, redirect violations, invalid content type, or exceeding max size.
    """
    is_safe, err = validate_url_ssrf(url)
    if not is_safe:
        raise ValueError(f"Forbidden URL: {err}")

    headers = {
        "User-Agent": "TeleRAG/1.0 (PDF Document Assistant)",
        "Accept": "application/pdf, application/octet-stream;q=0.9, */*;q=0.8",
    }

    current_url = url
    max_redirects = 3
    redirect_count = 0

    while True:
        try:
            response = requests.get(
                current_url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            )
        except requests.exceptions.Timeout as e:
            raise ValueError("Download timeout. The file took too long to download.") from e
        except requests.exceptions.RequestException as e:
            raise ValueError("Failed to download PDF. Please check the URL and try again.") from e

        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            redirect_count += 1
            if redirect_count > max_redirects:
                response.close()
                raise ValueError("Too many redirects (maximum 3 allowed).")

            location = response.headers.get("Location")
            if not location:
                response.close()
                raise ValueError("Redirect without location header.")

            next_url = urljoin(current_url, location)
            is_safe, err = validate_url_ssrf(next_url)
            if not is_safe:
                response.close()
                raise ValueError(f"Redirect blocked: {err}")

            response.close()
            current_url = next_url
            continue

        if response.status_code != 200:
            response.close()
            raise ValueError(f"HTTP request failed with status code {response.status_code}.")

        # Check content type
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not current_url.lower().endswith(".pdf"):
            response.close()
            raise ValueError(f"The URL doesn't point to a PDF file (Content-Type: {content_type}).")

        # Check declared Content-Length
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
                if declared_size > max_bytes:
                    response.close()
                    max_mb = max_bytes // (1024 * 1024)
                    raise ValueError(f"File too large. Maximum size is {max_mb}MB.")
            except ValueError as ve:
                if "File too large" in str(ve):
                    raise

        # Stream content with size limit
        buf = io.BytesIO()
        downloaded = 0
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        max_mb = max_bytes // (1024 * 1024)
                        raise ValueError(f"File too large. Maximum size is {max_mb}MB.")
                    buf.write(chunk)
        finally:
            response.close()

        return buf.getvalue()
