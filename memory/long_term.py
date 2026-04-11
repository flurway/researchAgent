"""
长期记忆 — FAISS 向量检索 + 时间衰减 + 去重
"""
import os
import json
import time
import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    memory_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    keywords: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class LongTermMemory:
    def __init__(self, index_path: str, embedding_dim: int = 512):
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.entries: list[MemoryEntry] = []
        self.index = None
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
        self._load()

    def _load(self):
        meta_path = f"{self.index_path}_meta.json"
        index_file = f"{self.index_path}.faiss"
        try:
            import faiss
            if os.path.exists(index_file) and os.path.exists(meta_path):
                self.index = faiss.read_index(index_file)
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**e) for e in data.get("entries", [])]
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
        except ImportError:
            logger.warning("FAISS not installed, long-term memory disabled")
            self.index = None

    def save(self):
        if self.index is None:
            return
        import faiss
        os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else ".", exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "entries": [
                    {"content": e.content, "memory_type": e.memory_type, "timestamp": e.timestamp,
                     "session_id": e.session_id, "keywords": e.keywords, "metadata": e.metadata}
                    for e in self.entries
                ]
            }, f, ensure_ascii=False, indent=2)

    async def add_memory(self, content: str, memory_type: str, embedding: np.ndarray,
                         session_id: str = "", keywords: list[str] = None, metadata: dict = None):
        if self.index is None:
            return
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding = embedding.astype(np.float32).reshape(1, -1)

        if self.index.ntotal > 0:
            scores, indices = self.index.search(embedding, 1)
            if scores[0][0] > 0.9:
                idx = indices[0][0]
                self.entries[idx].content = content
                self.entries[idx].timestamp = time.time()
                self.entries[idx].keywords = keywords or []
                self.save()
                return

        self.entries.append(MemoryEntry(
            content=content, memory_type=memory_type, session_id=session_id,
            keywords=keywords or [], metadata=metadata or {},
        ))
        self.index.add(embedding)
        self.save()

    async def search(self, query_embedding: np.ndarray, top_k: int = 5,
                     memory_type: Optional[str] = None, decay_days: int = 30) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        now = time.time()
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if memory_type and entry.memory_type != memory_type:
                continue
            age_days = (now - entry.timestamp) / 86400
            decay_factor = np.exp(-age_days / decay_days)
            results.append({
                "content": entry.content, "memory_type": entry.memory_type,
                "score": float(score) * decay_factor, "age_days": round(age_days, 1),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        return {
            "total_memories": len(self.entries),
            "index_size": self.index.ntotal if self.index else 0,
        }
