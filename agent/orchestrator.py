"""
ResearchGPT 主编排器

修复记录:
- [BUG FIX] _handle_deep_research 中 replan 失效问题
  原因: for step in plan["steps"] 的迭代器在循环开始时绑定了原始列表，
        plan 被重新赋值后迭代器仍走老列表，导致新计划永远不会被执行。
  修复: 改为 while + index 循环，replan 后重置 steps 列表和 index，
        从新计划的第一步重新开始执行。
"""
import logging
import numpy as np
from typing import Optional
from config import config
from models.deepseek import llm_client
from memory.short_term import ConversationMemory
from memory.long_term import LongTermMemory
from rag.retriever import HybridRetriever
from agent.intent import (IntentClassifier, INTENT_CHITCHAT, INTENT_NEED_CLARIFY,
                           INTENT_FOLLOW_UP, INTENT_DIRECT_SEARCH, INTENT_DEEP_RESEARCH)
from agent.planner import TaskPlanner
from agent.executor import TaskExecutor
from agent.reflector import Reflector

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """你是 ResearchGPT，专业的 AI 研究助手。

## 回答原则
- 基于检索到的文档回答，不编造信息
- 关键观点标注引用 [来源N]
- 简单问题直接精准回答；复杂问题结构化分析
- 如果检索不到信息，诚实说明"""


class ResearchAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.intent_classifier = IntentClassifier()
        self.planner = TaskPlanner()
        self.reflector = Reflector(
            confidence_threshold=config.agent.confidence_threshold,
            max_rounds=config.agent.max_reflection_rounds,
        )
        self.long_term_memory = LongTermMemory(
            index_path=config.memory.long_term_index_path,
            embedding_dim=config.rag.embedding_dim,
        )
        self.sessions: dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(
                max_turns=config.memory.max_short_term_turns,
                summary_threshold=config.memory.summary_threshold,
            )
            self.sessions[session_id].session_id = session_id
        return self.sessions[session_id]

    async def chat(self, user_message: str, session_id: str = "default") -> dict:
        memory = self.get_or_create_session(session_id)

        # ===== 意图判断 =====
        intent_result = await self.intent_classifier.classify(user_message, memory)
        intent = intent_result.get("intent", INTENT_DIRECT_SEARCH)
        logger.info(f"[{session_id}] Intent: {intent} | Query: {user_message[:50]}")

        # 闲聊
        if intent == INTENT_CHITCHAT:
            resp = await self._handle_chitchat(user_message, memory)
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", resp)
            return {"response": resp, "intent": intent, "citations": {}}

        # 需要澄清
        if intent == INTENT_NEED_CLARIFY:
            q = intent_result.get("clarify_question", "能否更具体地描述你的研究问题？")
            memory.add_turn("user", user_message, {"intent": intent})
            memory.add_turn("assistant", q)
            return {"response": q, "intent": intent, "citations": {},
                    "needs_clarification": True, "clarification_question": q,
                    "missing_info": intent_result.get("missing_info", [])}

        # 追问 → 指代消解
        if intent == INTENT_FOLLOW_UP:
            resolved = await self.intent_classifier.resolve_follow_up(user_message, memory)
            re_intent = await self.intent_classifier.classify(resolved)
            intent = re_intent.get("intent", INTENT_DIRECT_SEARCH)
            search_query = resolved
        else:
            search_query = intent_result.get("refined_query", user_message)

        # ===== 长期记忆 =====
        lt_ctx = await self._retrieve_long_term_memory(search_query)

        # ===== 执行研究 =====
        if intent == INTENT_DIRECT_SEARCH:
            result = await self._handle_direct_search(user_message, search_query, memory, lt_ctx)
        else:
            result = await self._handle_deep_research(
                user_message, search_query,
                intent_result.get("sub_questions", []), memory, lt_ctx,
            )

        # ===== 更新记忆 =====
        memory.add_turn("user", user_message, {"intent": intent})
        memory.add_turn("assistant", result["response"])
        if len(memory.turns) > memory.summary_threshold:
            await memory.compress_history(llm_client)
        if result.get("citations"):
            await self._save_to_long_term_memory(search_query, result["response"], session_id)

        result["intent"] = intent
        return result

    async def _handle_chitchat(self, message: str, memory: ConversationMemory) -> str:
        msgs = memory.build_context_messages(
            system_prompt="你是 ResearchGPT，友好的AI研究助手。简短回应闲聊，温和引导回研究话题。",
            current_query=message,
        )
        r = await llm_client.chat(msgs, temperature=0.7)
        return r["content"]

    async def _handle_direct_search(self, original_query: str, search_query: str,
                                     memory: ConversationMemory, lt_ctx: str) -> dict:
        executor = TaskExecutor(self.retriever)
        sr = await executor.execute_step({
            "step_id": 1, "action": "search_knowledge_base",
            "input_params": {"query": search_query, "top_k": config.rag.top_k_rerank},
            "depends_on": [],
        })
        if not sr.success:
            msgs = memory.build_context_messages(
                system_prompt=RESEARCH_SYSTEM_PROMPT, current_query=original_query, rag_context=lt_ctx,
            )
            r = await llm_client.chat(msgs)
            return {"response": r["content"] + "\n\n(注: 未在知识库中找到相关文档，以上基于模型自身知识)", "citations": {}}

        rag_ctx = sr.output
        if lt_ctx:
            rag_ctx = f"[历史研究参考]\n{lt_ctx}\n\n{rag_ctx}"
        msgs = memory.build_context_messages(
            system_prompt=RESEARCH_SYSTEM_PROMPT, current_query=original_query, rag_context=rag_ctx,
        )
        r = await llm_client.chat(msgs)
        return {"response": r["content"], "citations": sr.data.get("citations", {})}

    async def _handle_deep_research(self, original_query: str, search_query: str,
                                     sub_questions: list[str], memory: ConversationMemory,
                                     lt_ctx: str) -> dict:
        """
        深度研究 — Plan-and-Execute

        [BUG FIX] 旧代码用 for step in plan["steps"] 循环，replan 后新计划
        不会被执行。改为 while + index，replan 时重置 steps 和 index。
        """
        # --- 规划 ---
        plan = await self.planner.create_plan(
            query=search_query, sub_questions=sub_questions, context=lt_ctx,
        )

        # --- 执行 (修复后的循环) ---
        executor = TaskExecutor(self.retriever)
        self.reflector.reset()
        all_step_results: list = []
        replan_count = 0
        max_replans = config.agent.max_reflection_rounds

        # ★ 关键修复: 用 while + index 替代 for 循环
        # 这样 replan 后可以重置 steps 和 step_idx，执行新计划
        steps = list(plan.get("steps", []))
        step_idx = 0

        while step_idx < len(steps):
            step = steps[step_idx]

            # 检查是否需要用户输入
            if step.get("action") == "ask_user_clarification":
                result = await executor.execute_step(step)
                return {
                    "response": result.output,
                    "needs_clarification": True,
                    "clarification_question": result.output,
                    "citations": {}, "plan": plan,
                }

            result = await executor.execute_step(step)
            all_step_results.append(result)

            if not result.success and replan_count < max_replans:
                # ★ 关键修复: replan 后替换 steps 列表并重置 index
                logger.warning(f"Step {step['step_id']} failed: {result.error}, triggering replan #{replan_count+1}")
                new_plan = await self.planner.replan(
                    original_plan=plan,
                    completed_steps=[r.to_dict() for r in all_step_results if r.success],
                    failed_step=step,
                    failure_reason=result.error,
                )
                # 用新计划的步骤替换当前待执行列表
                new_steps = new_plan.get("steps", [])
                if new_steps:
                    plan = new_plan
                    steps = list(new_steps)
                    step_idx = 0      # ★ 从新计划的第一步开始
                    replan_count += 1
                    logger.info(f"Replan applied: {len(steps)} new steps, restarting execution")
                    continue
                else:
                    # 新计划为空，跳过继续
                    step_idx += 1
                    continue
            elif not result.success:
                logger.warning(f"Step failed and max replans ({max_replans}) reached, skipping")

            step_idx += 1

        # --- 反思 ---
        reflection = await self.reflector.reflect(search_query, all_step_results, plan)

        if not reflection.get("is_sufficient", True):
            for action in reflection.get("suggested_actions", [])[:2]:
                supp = await executor.execute_step({
                    "step_id": len(all_step_results) + 1,
                    "action": action.get("action", "search_knowledge_base"),
                    "input_params": action.get("params", {}),
                    "depends_on": [],
                })
                all_step_results.append(supp)

        # --- 生成报告 ---
        report = await executor.execute_step({
            "step_id": 99, "action": "generate_research_report",
            "input_params": {
                "topic": search_query,
                "findings": [r.output[:200] for r in all_step_results if r.success],
                "format": "detailed",
            },
            "depends_on": [r.step_id for r in all_step_results],
        })

        # --- 幻觉检测 ---
        source_docs = "\n".join(r.output for r in all_step_results if r.success)
        hall_check = await self.reflector.check_hallucination(report.output, source_docs[:6000])

        response = report.output
        if hall_check.get("hallucination_rate", 0) > 0.3:
            response += "\n\n⚠️ 部分内容可能缺乏充分文档支撑，建议进一步验证。"

        return {
            "response": response,
            "citations": executor.all_citations,
            "plan": plan,
            "reflection": reflection,
            "hallucination_check": hall_check,
        }

    async def _retrieve_long_term_memory(self, query: str) -> str:
        try:
            qe = self.retriever.get_embeddings([query])[0]
            memories = await self.long_term_memory.search(query_embedding=qe, top_k=config.memory.max_long_term_results)
            parts = [m["content"] for m in memories if m["score"] > 0.3]
            return "\n\n".join(parts[:3]) if parts else ""
        except Exception as e:
            logger.warning(f"Long-term memory retrieval failed: {e}")
            return ""

    async def _save_to_long_term_memory(self, query: str, response: str, session_id: str):
        try:
            emb = self.retriever.get_embeddings([query])[0]
            await self.long_term_memory.add_memory(
                content=f"研究问题: {query}\n结论摘要: {response[:500]}",
                memory_type="research_summary", embedding=emb,
                session_id=session_id, keywords=query.split()[:5],
            )
        except Exception as e:
            logger.warning(f"Failed to save long-term memory: {e}")
