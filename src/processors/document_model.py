"""Common internal document model for diverse source types."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DocumentSection:
    """Represents a logical section within a document."""
    label: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalDocument:
    """Common internal representation of an ingested document."""
    source_name: str
    source_type: str  # 'pdf' | 'markdown' | 'text' | 'web'
    sections: List[DocumentSection] = field(default_factory=list)
    raw_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
