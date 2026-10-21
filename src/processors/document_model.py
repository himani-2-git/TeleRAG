"""Common internal document model for diverse source types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSection:
    """Represents a logical section within a document."""

    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalDocument:
    """Common internal representation of an ingested document."""

    source_name: str
    source_type: str  # 'pdf' | 'markdown' | 'text' | 'web'
    sections: list[DocumentSection] = field(default_factory=list)
    raw_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
