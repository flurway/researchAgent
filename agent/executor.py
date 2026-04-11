"""
任务执行器 — 调用工具 + 管理中间状态
"""
import logging
from models.deepseek import llm_client
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker, CitationTracer
from config import config

logger = logging.getLogger(__name__)


class StepResult:
    def __init__(self, step_id: int, action: str, success: bool,
                 output: str = "", data: dict = None, error: str = ""):
        self.step_id = step_id
        self.action = action
        self.success = success
        self.output = output
        self.data = data or {}
        self.error = error

    def to_dict(self) -> dict:
        return {"step_id": self.step_id, "action": self.action,
                "success": self.success, "output": self.output[:500], "error": self.error}


class TaskExecutor:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.reranker = CrossEncoderReranker(use_model_rerank=False)
        self.citation_tracer = CitationTracer()
        self.execution_context: dict[int, StepResult] = {}
        self.all_citations: dict = {}

    async def execute_step(self, step: dict) -> StepResult:
        action = step.get("action", "")
        step_id = step.get("step_id", 0)
        params = step.get("input_params", {})
        logger.info(f"Executing step {step_id}: {action}")
        dep_ctx = self._gather_dep_context(step.get("depends_on", []))

        try:
            if action == "search_knowledge_base":
                result = await self._search(params)
            elif action == "read_document_detail":
                result = await self._read(params)
            elif action == "summarize_content":
                result = await self._summarize(params, dep_ctx)
            elif action == "compare_concepts":
                result = await self._compare(params, dep_ctx)
            elif action == "generate_research_report":
                result = await self._report(params)
            elif action == "ask_user_clarification":
                result = StepResult(step_id, action, True, output=params.get("question", ""),
                                    data={"needs_user_input": True, "options": params.get("options", [])})
            else:
                result = StepResult(step_id, action, False, error=f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Step {step_id} failed: {e}")
            result = StepResult(step_id, action, False, error=str(e))

        result.step_id = step_id
        result.action = action
        self.execution_context[step_id] = result
        return result

    def _gather_dep_context(self, depends_on: list[int]) -> str:
        parts = []
        for dep_id in depends_on:
            dep = self.execution_context.get(dep_id)
            if dep and dep.success:
                parts.append(f"[步骤{dep_id}结果] {dep.output[:500]}")
        return "\n\n".join(parts)

    async def _search(self, params: dict) -> StepResult:
        results = await self.retriever.search(
            query=params.get("query", ""),
            top_k_dense=config.rag.top_k_dense,
            top_k_sparse=config.rag.top_k_sparse,
            metadata_filter=params.get("filters"),
        )
        if not results:
            return StepResult(0, "search_knowledge_base", False, error="No results found")
        reranked = await self.reranker.rerank(params.get("query", ""), results, top_k=params.get("top_k", 5))
        context, citations = self.citation_tracer.build_citation_context(reranked)
        self.all_citations.update(citations)
        return StepResult(0, "search_knowledge_base", True, output=context,
                          data={"num_results": len(reranked), "citations": citations, "query": params.get("query")})

    async def _read(self, params: dict) -> StepResult:
        doc_id = params.get("doc_id", "")
        chunks = [c for c in self.retriever.chunks if c.doc_id == doc_id]
        if params.get("section"):
            chunks = [c for c in chunks if c.metadata.get("section_title", "").lower() == params["section"].lower()]
        if not chunks:
            return StepResult(0, "read_document_detail", False, error=f"Doc not found: {doc_id}")
        chunks.sort(key=lambda c: c.chunk_index)
        return StepResult(0, "read_document_detail", True,
                          output="\n\n".join(c.content for c in chunks)[:3000])

    async def _summarize(self, params: dict, dep_ctx: str) -> StepResult:
        content = params.get("content", dep_ctx)
        focus = params.get("focus", "")
        prompt = f"摘要(≤{params.get('max_length',300)}字)"
        if focus: prompt += f"，关注: {focus}"
        r = await llm_client.chat([
            {"role": "system", "content": "你是精准的研究摘要助手。"},
            {"role": "user", "content": f"{prompt}\n\n{content[:4000]}"},
        ], temperature=0.2)
        return StepResult(0, "summarize_content", True, output=r["content"])

    async def _compare(self, params: dict, dep_ctx: str) -> StepResult:
        prompt = f"对比: {', '.join(params.get('concepts', []))}"
        if params.get("dimensions"): prompt += f"\n维度: {', '.join(params['dimensions'])}"
        if dep_ctx: prompt += f"\n参考:\n{dep_ctx}"
        r = await llm_client.chat([
            {"role": "system", "content": "你是技术分析专家，善于多维对比。"},
            {"role": "user", "content": prompt},
        ], temperature=0.3)
        return StepResult(0, "compare_concepts", True, output=r["content"])

    async def _report(self, params: dict) -> StepResult:
        all_outputs = [r.output for _, r in sorted(self.execution_context.items()) if r.success and r.output]
        material = "\n\n---\n\n".join(all_outputs)
        fmt = {"brief": "简报(≤500字)", "detailed": "结构化报告(背景/发现/分析/结论)", "academic": "学术风格报告"}
        r = await llm_client.chat([
            {"role": "system", "content": f"你是研究报告撰写专家。{fmt.get(params.get('format','detailed'), fmt['detailed'])}\n所有观点标注 [来源N]。"},
            {"role": "user", "content": f"主题: {params.get('topic','')}\n\n素材:\n{material[:8000]}\n\n引用:\n{self.citation_tracer.format_citations(self.all_citations)}\n\n请生成报告。"},
        ], temperature=0.3, max_tokens=4096)
        return StepResult(0, "generate_research_report", True, output=r["content"], data={"citations": self.all_citations})
