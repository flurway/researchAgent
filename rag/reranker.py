"""
重排序 (LLM pointwise) + 引用溯源
"""
import asyncio
import logging
from rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, use_model_rerank: bool = False):
        self.use_model_rerank = use_model_rerank
        self._model = None

    async def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 8) -> list[RetrievalResult]:
        if not candidates:
            return []
        if self.use_model_rerank:
            return self._model_rerank(query, candidates, top_k)
        return await self._llm_rerank(query, candidates, top_k)

    def _model_rerank(self, query, candidates, top_k):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
            except ImportError:
                return candidates[:top_k]
        pairs = [(query, c.chunk.content[:512]) for c in candidates]
        scores = self._model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    async def _llm_rerank(self, query, candidates, top_k):
        from models.deepseek import llm_client

        async def score_one(c):
            msgs = [
                {"role": "system", "content": "评估文档与查询的相关性(0-10)，只输出数字。10=完全相关 0=无关"},
                {"role": "user", "content": f"查询: {query}\n\n文档: {c.chunk.content[:500]}\n\n分数:"},
            ]
            r = await llm_client.chat(msgs, temperature=0.0, max_tokens=5)
            try:
                return float(r["content"].strip())
            except ValueError:
                return 5.0

        scores = await asyncio.gather(*[score_one(c) for c in candidates])
        for c, s in zip(candidates, scores):
            c.score = s
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]


class CitationTracer:
    def build_citation_context(self, results: list[RetrievalResult]) -> tuple[str, dict]:
        parts, cmap = [], {}
        for i, r in enumerate(results, 1):
            c = r.chunk
            cid = str(i)
            cmap[cid] = {
                "doc_title": c.doc_title, "doc_id": c.doc_id, "chunk_id": c.chunk_id,
                "section": c.metadata.get("section_title", ""),
                "source": c.metadata.get("source", ""),
                "author": c.metadata.get("author", ""),
                "year": c.metadata.get("year", ""),
            }
            info = f"[来源{cid}] {c.doc_title}"
            if c.metadata.get("section_title"):
                info += f" > {c.metadata['section_title']}"
            parts.append(f"{info}\n{c.content}")
        return "\n\n---\n\n".join(parts), cmap

    def format_citations(self, cmap: dict) -> str:
        lines = []
        for cid, info in cmap.items():
            p = [f"[{cid}]"]
            if info.get("author"): p.append(info["author"])
            p.append(f'"{info["doc_title"]}"')
            if info.get("year"): p.append(f"({info['year']})")
            if info.get("section"): p.append(f"Section: {info['section']}")
            lines.append(" ".join(p))
        return "\n".join(lines)
