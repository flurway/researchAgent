"""
多路召回检索器 — FAISS 向量检索 + BM25 + RRF 融合
"""
import os
import json
import math
import logging
import numpy as np
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass
from rag.chunker import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    source: str = ""


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.term_freqs: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.chunks: list[DocumentChunk] = []

    def add_documents(self, chunks: list[DocumentChunk]):
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for token in set(tokens):
                self.doc_freq[token] += 1
            self.term_freqs.append(dict(tf))
            self.doc_lens.append(len(tokens))
            self.chunks.append(chunk)
        self.doc_count = len(self.chunks)
        self.avg_doc_len = sum(self.doc_lens) / max(self.doc_count, 1)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores = []
        for i in range(self.doc_count):
            score = 0.0
            for token in query_tokens:
                if token not in self.term_freqs[i]:
                    continue
                tf = self.term_freqs[i][token]
                df = self.doc_freq.get(token, 0)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avg_doc_len))
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        for i in range(len(cn_chars) - 1):
            tokens.append(cn_chars[i] + cn_chars[i+1])
        tokens.extend(cn_chars)
        return tokens


class HybridRetriever:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.bm25_index = BM25Index()
        self.chunks: list[DocumentChunk] = []
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            from config import config
            self._embedding_model = SentenceTransformer(config.rag.embedding_model)
        return self._embedding_model

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        model = self._get_embedding_model()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def build_index(self, chunks: list[DocumentChunk]):
        import faiss
        self.chunks = chunks
        texts = [c.content for c in chunks]
        logger.info(f"Building embeddings for {len(texts)} chunks...")
        embeddings = self.get_embeddings(texts)

        if len(chunks) < 10000:
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        else:
            nlist = min(int(np.sqrt(len(chunks))), 256)
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
            self.faiss_index.train(embeddings)
            self.faiss_index.nprobe = 10
        self.faiss_index.add(embeddings)

        self.bm25_index = BM25Index()
        self.bm25_index.add_documents(chunks)
        logger.info(f"Index built: {self.faiss_index.ntotal} vectors, {self.bm25_index.doc_count} BM25 docs")

    async def search(self, query: str, top_k_dense: int = 20, top_k_sparse: int = 20,
                     metadata_filter: Optional[dict] = None) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        # Dense
        qe = self.get_embeddings([query])
        d_scores, d_indices = self.faiss_index.search(qe, top_k_dense)
        dense = {}
        for rank, (sc, idx) in enumerate(zip(d_scores[0], d_indices[0])):
            if 0 <= idx < len(self.chunks):
                dense[self.chunks[idx].chunk_id] = (rank, float(sc))
        # Sparse
        sparse_hits = self.bm25_index.search(query, top_k_sparse)
        sparse = {}
        for rank, (idx, sc) in enumerate(sparse_hits):
            sparse[self.chunks[idx].chunk_id] = (rank, float(sc))
        # Metadata filter
        filtered_ids = self._apply_metadata_filter(metadata_filter) if metadata_filter else None
        # RRF fusion
        K = 60
        fused = []
        for cid in set(dense.keys()) | set(sparse.keys()):
            if filtered_ids is not None and cid not in filtered_ids:
                continue
            rrf = 0.0
            ds = ss = 0.0
            sources = []
            if cid in dense:
                rrf += 1.0 / (K + dense[cid][0])
                ds = dense[cid][1]
                sources.append("dense")
            if cid in sparse:
                rrf += 1.0 / (K + sparse[cid][0])
                ss = sparse[cid][1]
                sources.append("sparse")
            chunk = next(c for c in self.chunks if c.chunk_id == cid)
            fused.append(RetrievalResult(chunk=chunk, score=rrf, dense_score=ds, sparse_score=ss, source="+".join(sources)))
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def _apply_metadata_filter(self, filters: dict) -> set[str]:
        valid = set()
        for c in self.chunks:
            match = True
            for k, cond in filters.items():
                v = c.metadata.get(k)
                if v is None:
                    match = False; break
                if isinstance(cond, str) and cond.startswith(">="):
                    if str(v) < cond[2:]: match = False
                elif isinstance(cond, str) and cond.startswith("<="):
                    if str(v) > cond[2:]: match = False
                elif v != cond:
                    match = False
            if match:
                valid.add(c.chunk_id)
        return valid

    def save_index(self, path: str):
        import faiss
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if self.faiss_index:
            faiss.write_index(self.faiss_index, f"{path}.faiss")
        with open(f"{path}_chunks.json", "w", encoding="utf-8") as f:
            json.dump([{"chunk_id": c.chunk_id, "content": c.content, "doc_id": c.doc_id,
                        "doc_title": c.doc_title, "chunk_index": c.chunk_index, "metadata": c.metadata}
                       for c in self.chunks], f, ensure_ascii=False)

    def load_index(self, path: str):
        import faiss
        self.faiss_index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_chunks.json", "r", encoding="utf-8") as f:
            self.chunks = [DocumentChunk(**item) for item in json.load(f)]
        self.bm25_index = BM25Index()
        self.bm25_index.add_documents(self.chunks)
