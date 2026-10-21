"""Markdown processor for heading-aware chunking and citation labeling."""

import logging
import re

from .text_chunker import TextChunk, TextChunker

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Processes markdown documents by splitting at headings and generating path labels."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    def extract_sections(self, text: str, default_label: str = "Overview") -> list[tuple[str, str]]:
        """Split markdown text into sections based on heading hierarchy.

        Args:
            text: Raw markdown text.
            default_label: Label for any content preceding the first heading.

        Returns:
            List of (heading_path, section_content) tuples.
        """
        lines = text.splitlines()
        sections: list[tuple[str, list[str]]] = []
        hierarchy: list[tuple[int, str]] = []

        current_label = default_label
        current_lines: list[str] = []

        heading_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = heading_regex.match(line.strip())
            if match:
                # Flush previous section if it has non-whitespace content
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_label, current_lines))

                level = len(match.group(1))
                title = match.group(2).strip()

                # Pop headings of equal or deeper level
                while hierarchy and hierarchy[-1][0] >= level:
                    hierarchy.pop()
                hierarchy.append((level, title))

                current_label = " > ".join(t for _, t in hierarchy)
                current_lines = [line]
            else:
                current_lines.append(line)

        # Flush final section
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_label, current_lines))

        # Join lines into strings
        return [
            (label, "\n".join(sec_lines).strip())
            for label, sec_lines in sections
            if "\n".join(sec_lines).strip()
        ]

    def process_markdown(self, text: str, source_name: str = "document.md") -> list[TextChunk]:
        """Process markdown content into chunks with heading path labels.

        Args:
            text: Raw markdown string.
            source_name: Name of the markdown file.

        Returns:
            List of TextChunk instances with section_label set to heading hierarchy.
        """
        if not text or not text.strip():
            logger.warning("Empty markdown text provided for processing")
            return []

        sections = self.extract_sections(text, default_label="Overview")
        all_chunks: list[TextChunk] = []
        global_pos = 0

        for label, section_text in sections:
            sec_len = len(section_text)
            if sec_len <= self.chunk_size:
                chunk = TextChunk(
                    content=section_text,
                    chunk_index=len(all_chunks),
                    start_pos=global_pos,
                    end_pos=global_pos + sec_len,
                    char_count=sec_len,
                    source_type="markdown",
                    source_name=source_name,
                    section_label=label,
                )
                all_chunks.append(chunk)
                global_pos += sec_len + 2
            else:
                # Long section: split with existing chunking algorithm
                sub_chunks = self.chunker.create_chunks(section_text)
                for sc in sub_chunks:
                    chunk = TextChunk(
                        content=sc.content,
                        chunk_index=len(all_chunks),
                        start_pos=global_pos + sc.start_pos,
                        end_pos=global_pos + sc.end_pos,
                        char_count=sc.char_count,
                        source_type="markdown",
                        source_name=source_name,
                        section_label=label,
                    )
                    all_chunks.append(chunk)
                global_pos += sec_len + 2

        # Ensure chunk indexes are strictly sequential
        for idx, chunk in enumerate(all_chunks):
            chunk.chunk_index = idx

        logger.info(
            f"Processed markdown into {len(all_chunks)} chunks across {len(sections)} sections"
        )
        return all_chunks
