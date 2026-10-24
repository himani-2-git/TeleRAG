"""Post-generation grounding check for RAG answers."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundingReport:
    """Report detailing citation and factual grounding metrics."""

    applicable: bool
    citation_coverage: float
    total_sentences: int
    supported_sentences: int
    unsupported_sentences: list[str] = field(default_factory=list)
    max_scores: list[float] = field(default_factory=list)
    chunk_support_counts: list[int] = field(default_factory=list)
    supporting_chunk_indices: list[int] = field(default_factory=list)


def strip_footers(text: str) -> str:
    """Strip Sources footer and 'Answered in' timing footer from text."""
    if not text:
        return ""
    # Strip truncation note if present
    cleaned = re.sub(
        r"(?:\n\s*|\A)\(Answer was cut off; ask a narrower question for the full detail\.\)\s*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip timing footer (e.g., 'Answered in 2 s with qwen2.5:3b')
    cleaned = re.sub(
        r"(?:\n\s*|\A)Answered in \d+(?:\.\d+)?\s*s\s+with\s+.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip sources footer (e.g., 'Sources:\n[ci_fixture.pdf, p. 1] ...')
    cleaned = re.sub(
        r"(?:\n\s*|\A)Sources:\s*.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split clean text into sentences."""
    if not text:
        return []
    # Normalize whitespace/newlines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sentences: list[str] = []
    for line in lines:
        # Split line on terminal punctuation followed by space or end of line
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            p = part.strip()
            # Filter out tiny noise / markdown list bullets
            p_clean = re.sub(r"^[-*•\d\.\)]\s*", "", p).strip()
            if len(p_clean) > 3:
                sentences.append(p)
    return sentences


def _is_refusal_or_empty(text: str) -> bool:
    """Check if the text is empty or an explicit refusal."""
    if not text:
        return True
    trimmed = text.strip()
    if not trimmed:
        return True
    lower = trimmed.lower()
    if lower.startswith("insufficient_context") or "insufficient_context" in lower:
        return True
    refusal_cues = [
        "couldn't find relevant information",
        "could not find relevant information",
        "no relevant information",
        "no document content available",
        "does not contain information",
        "not mentioned in the context",
    ]
    return any(cue in lower for cue in refusal_cues)


def _get_embedding(embedder: Any, text: str) -> np.ndarray:
    """Generate vector embedding for single text using embedder."""
    if hasattr(embedder, "generate_embeddings"):
        res = embedder.generate_embeddings(text)
    elif hasattr(embedder, "generate_query_embedding"):
        res = embedder.generate_query_embedding(text)
    elif hasattr(embedder, "encode"):
        res = embedder.encode(text)
    elif callable(embedder):
        res = embedder(text)
    else:
        raise ValueError(f"Unsupported embedder interface: {type(embedder)}")

    arr = np.array(res, dtype=float)
    if arr.ndim > 1:
        arr = arr[0]
    return arr


def _get_chunk_text_and_embedding(chunk: Any, embedder: Any) -> tuple[str, np.ndarray]:
    """Extract text and embedding for a chunk, generating embedding if missing."""
    text = ""
    emb = None
    if isinstance(chunk, dict):
        text = chunk.get("content", "") or chunk.get("text", "")
        if "embedding" in chunk and chunk["embedding"] is not None:
            emb = np.array(chunk["embedding"], dtype=float)
    elif hasattr(chunk, "content"):
        text = getattr(chunk, "content", "")
        if hasattr(chunk, "embedding") and chunk.embedding is not None:
            emb = np.array(chunk.embedding, dtype=float)
    elif isinstance(chunk, str):
        text = chunk

    if emb is None or len(emb) == 0:
        emb = _get_embedding(embedder, text)
    return text, emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def grounding_report(
    answer: str | None,
    chunks: list[dict[str, Any]] | list[Any],
    embedder: Any,
    threshold: float = 0.45,
) -> GroundingReport:
    """Compute sentence-level grounding report comparing answer against retrieved chunks.

    Args:
        answer: The model's answer string (may include footers).
        chunks: Retrieved chunks (dicts, Chunk objects, or strings).
        embedder: Embedding generator instance or function.
        threshold: Minimum cosine similarity required to consider a sentence supported.

    Returns:
        GroundingReport instance.
    """
    if answer is None:
        return GroundingReport(
            applicable=False,
            citation_coverage=1.0,
            total_sentences=0,
            supported_sentences=0,
            unsupported_sentences=[],
            max_scores=[],
        )

    # 1. Strip footers first
    clean_body = strip_footers(answer)

    # 2. Check refusal / empty
    if _is_refusal_or_empty(clean_body):
        return GroundingReport(
            applicable=False,
            citation_coverage=1.0,
            total_sentences=0,
            supported_sentences=0,
            unsupported_sentences=[],
            max_scores=[],
        )

    # 3. Split into sentences
    sentences = split_sentences(clean_body)
    if not sentences:
        return GroundingReport(
            applicable=False,
            citation_coverage=1.0,
            total_sentences=0,
            supported_sentences=0,
            unsupported_sentences=[],
            max_scores=[],
        )

    # 4. Handle empty chunks
    if not chunks:
        return GroundingReport(
            applicable=True,
            citation_coverage=0.0,
            total_sentences=len(sentences),
            supported_sentences=0,
            unsupported_sentences=sentences,
            max_scores=[0.0] * len(sentences),
        )

    # 5. Extract chunk embeddings
    chunk_embeddings: list[np.ndarray] = []
    for c in chunks:
        _, emb = _get_chunk_text_and_embedding(c, embedder)
        chunk_embeddings.append(emb)

    supported_count = 0
    unsupported: list[str] = []
    max_scores: list[float] = []
    chunk_support = [0] * len(chunks)
    chunk_max_sim = [0.0] * len(chunks)

    for sentence in sentences:
        s_vec = _get_embedding(embedder, sentence)
        best_sim = 0.0
        for idx, c_vec in enumerate(chunk_embeddings):
            sim = cosine_similarity(s_vec, c_vec)
            if sim > best_sim:
                best_sim = sim
            if sim > chunk_max_sim[idx]:
                chunk_max_sim[idx] = sim
            if sim >= threshold:
                chunk_support[idx] += 1

        max_scores.append(best_sim)
        if best_sim >= threshold:
            supported_count += 1
        else:
            unsupported.append(sentence)

    coverage = supported_count / len(sentences) if sentences else 1.0

    supporting_indices = [i for i in range(len(chunks)) if chunk_support[i] > 0]
    supporting_indices.sort(key=lambda i: (-chunk_support[i], -chunk_max_sim[i], i))

    return GroundingReport(
        applicable=True,
        citation_coverage=coverage,
        total_sentences=len(sentences),
        supported_sentences=supported_count,
        unsupported_sentences=unsupported,
        max_scores=max_scores,
        chunk_support_counts=chunk_support,
        supporting_chunk_indices=supporting_indices,
    )
