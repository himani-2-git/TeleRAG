"""Common internal document model for diverse source types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSection:
    """Represents a logical section within a document."""

    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
