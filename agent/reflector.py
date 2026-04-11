"""
自我反思 — 质量评估 + 幻觉检测
"""
import json
import logging
from models.deepseek import llm_client
from agent.executor import StepResult

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """你是研究质量评估器。评估维度:
1. 信息充分性(0-10) 2. 信息一致性(0-10) 3. 覆盖度(0-10) 4. 引用质量(0-10)

输出JSON:
{
    "scores": {"sufficiency": 8, "consistency": 9, "coverage": 7, "citation_quality": 8},
    "overall_score": 8.0,
    "is_sufficient": true,
    "missing_aspects": [],
    "suggested_actions": [{"action": "search_knowledge_base", "reason": "...", "params": {"query": "..."}}],
    "reasoning": "理由"
}"""


class Reflector:
    def __init__(self, confidence_threshold: float = 0.7, max_rounds: int = 3):
        self.confidence_threshold = confidence_threshold
        self.max_rounds = max_rounds
        self.reflection_count = 0

    async def reflect(self, original_query: str, step_results: list[StepResult], plan: dict) -> dict:
        self.reflection_count += 1
        if self.reflection_count > self.max_rounds:
            return {"is_sufficient": True, "overall_score": 0.6, "reasoning": "达到最大反思轮数", "suggested_actions": []}

        summary = [{"step_id": r.step_id, "action": r.action, "success": r.success,
                     "output_preview": r.output[:300], "error": r.error} for r in step_results]
        result = await llm_client.get_json_response([
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": f"问题: {original_query}\n计划: {json.dumps(plan, ensure_ascii=False)[:1000]}\n结果:\n{json.dumps(summary, ensure_ascii=False)}\n请评估。"},
        ])
        if not result:
            return {"is_sufficient": True, "overall_score": 0.6, "suggested_actions": []}
        score = result.get("overall_score", 5.0) / 10.0
        result["is_sufficient"] = score >= self.confidence_threshold
        logger.info(f"Reflection #{self.reflection_count}: score={score:.2f}, sufficient={result['is_sufficient']}")
        return result

    async def check_hallucination(self, generated_text: str, source_documents: str) -> dict:
        result = await llm_client.get_json_response([
            {"role": "system", "content": "检查生成内容是否有源文档支撑。输出JSON: {\"hallucinations\": [{\"claim\": \"...\", \"status\": \"supported|unsupported\"}], \"hallucination_rate\": 0.0}"},
            {"role": "user", "content": f"源文档:\n{source_documents[:4000]}\n\n生成内容:\n{generated_text[:2000]}\n\n请检查。"},
        ])
        return result or {"hallucinations": [], "hallucination_rate": 0.0}

    def reset(self):
        self.reflection_count = 0
