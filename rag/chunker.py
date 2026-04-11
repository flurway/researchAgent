"""
文档分块 — 递归分割 + 结构化分割
"""
import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    doc_id: str
    doc_title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return int(len(self.content) / 1.5)


class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = ["\n\n", "\n", "。", ".", "；", ";", "，", ",", " ", ""]

    def chunk_document(self, text: str, doc_id: str, doc_title: str = "",
                       metadata: Optional[dict] = None) -> list[DocumentChunk]:
        if not text.strip():
            return []
        raw_chunks = self._split_recursive(text, self.separators)
        merged = self._merge_small_chunks(raw_chunks)
        overlapped = self._add_overlap(merged)

        chunks = []
        for i, content in enumerate(overlapped):
            cid = hashlib.md5(f"{doc_id}:{i}:{content[:50]}".encode()).hexdigest()[:12]
            chunks.append(DocumentChunk(
                chunk_id=cid, content=content.strip(), doc_id=doc_id,
                doc_title=doc_title, chunk_index=i, metadata=metadata or {},
            ))
        logger.info(f"Chunked '{doc_title}' → {len(chunks)} chunks")
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = separators[0]
        if sep == "":
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        parts = text.split(sep)
        result, current = [], ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._split_recursive(part, separators[1:]))
                    current = ""
                else:
                    current = part
        if current:
            result.append(current)
        return result

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        merged, buf = [], ""
        for c in chunks:
            if len(buf) + len(c) < self.min_chunk_size:
                buf += c
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            merged.append(buf)
        return merged

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            result.append(chunks[i-1][-self.chunk_overlap:] + chunks[i])
        return result


class StructuredChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    def chunk_document(self, text: str, doc_id: str, doc_title: str = "",
                       metadata: Optional[dict] = None) -> list[DocumentChunk]:
        sections = self._extract_sections(text)
        fallback = RecursiveChunker(self.chunk_size, self.chunk_overlap)
        chunks = []
        for sec in sections:
            sec_meta = {**(metadata or {}), "section_title": sec["title"], "section_level": sec["level"]}
            if len(sec["content"]) <= self.chunk_size:
                cid = hashlib.md5(f"{doc_id}:{sec['title']}".encode()).hexdigest()[:12]
                chunks.append(DocumentChunk(
                    chunk_id=cid, content=f"[{sec['title']}]\n{sec['content']}".strip(),
                    doc_id=doc_id, doc_title=doc_title, chunk_index=len(chunks), metadata=sec_meta,
                ))
            else:
                for sc in fallback.chunk_document(sec["content"], doc_id, doc_title, sec_meta):
                    sc.content = f"[{sec['title']}]\n{sc.content}"
                    sc.chunk_index = len(chunks)
                    chunks.append(sc)
        return chunks

    def _extract_sections(self, text: str) -> list[dict]:
        headings = list(self.heading_pattern.finditer(text))
        if not headings:
            return [{"title": "Full Document", "level": 0, "content": text}]
        sections = []
        for i, m in enumerate(headings):
            level = len(m.group(1))
            title = m.group(2).strip()
            start = m.end()
            end = headings[i+1].start() if i+1 < len(headings) else len(text)
            sections.append({"title": title, "level": level, "content": text[start:end].strip()})
        return sections
