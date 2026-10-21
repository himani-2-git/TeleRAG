"""Cross-encoder reranking module for semantic re-ordering of retrieved candidates."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidate chunks using a pretrained cross-encoder model."""

    _models: dict[str, Any] = {}

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker with model identifier."""
        self.model_name = model_name

    @classmethod
    def _get_model(cls, model_name: str):
        """Lazy load and cache cross-encoder model keyed by model_name."""
        if model_name not in cls._models:
            logger.info("Loading CrossEncoder model '%s' on CPU", model_name)
            from sentence_transformers import CrossEncoder

            cls._models[model_name] = CrossEncoder(model_name, device="cpu")
        return cls._models[model_name]

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Rerank candidate chunks against the query using cross-encoder scoring."""
        if not candidates:
            return []

        if not query.strip():
            return candidates[:top_k]

        model = self._get_model(self.model_name)
        pairs = [[query, c.get("content", "")] for c in candidates]
        scores = model.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores, strict=False):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
