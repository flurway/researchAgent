"""
意图判断模块 — 五类分流 + 指代消解
"""
import logging
from typing import Optional
from models.deepseek import llm_client
from memory.short_term import ConversationMemory

logger = logging.getLogger(__name__)

INTENT_DIRECT_SEARCH = "direct_search"
INTENT_NEED_CLARIFY = "need_clarify"
INTENT_DEEP_RESEARCH = "deep_research"
INTENT_FOLLOW_UP = "follow_up"
INTENT_CHITCHAT = "chitchat"

INTENT_SYSTEM_PROMPT = """你是研究助手的意图分析模块。分析用户提问的意图类型。

## 意图类型
1. direct_search: 明确的可直接检索的问题 (如: "BERT预训练目标是什么？")
2. need_clarify: 太模糊/太大，需要引导 (如: "帮我了解大模型")
3. deep_research: 需要多步分析的复杂问题 (如: "对比RAG和Fine-tune的适用场景")
4. follow_up: 基于上文的追问 (如: "那第二种呢？")
5. chitchat: 闲聊 (如: "你好")

## 输出格式 (严格JSON)
{
    "intent": "意图类型",
    "confidence": 0.0-1.0,
    "refined_query": "优化后的检索query",
    "missing_info": ["缺少的信息维度"],
    "clarify_question": "引导用户的提问",
    "sub_questions": ["拆解的子问题"],
    "reasoning": "判断理由"
}"""


class IntentClassifier:
    async def classify(self, user_query: str, conversation_memory: Optional[ConversationMemory] = None) -> dict:
        messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
        if conversation_memory and conversation_memory.turns:
            recent = conversation_memory.get_recent_turns(n=3)
            ctx = "\n".join([f"{'用户' if t['role']=='user' else '助手'}: {t['content'][:200]}" for t in recent])
            messages.append({"role": "user", "content": f"对话上下文:\n{ctx}\n\n当前提问: {user_query}\n\n请分析意图，返回JSON。"})
        else:
            messages.append({"role": "user", "content": f"用户提问: {user_query}\n\n请分析意图，返回JSON。"})

        result = await llm_client.get_json_response(messages)
        if not result:
            return {"intent": INTENT_DIRECT_SEARCH, "confidence": 0.5, "refined_query": user_query}
        logger.info(f"Intent: {result.get('intent')} | Query: {user_query[:50]}")
        return result

    async def resolve_follow_up(self, user_query: str, conversation_memory: ConversationMemory) -> str:
        recent = conversation_memory.get_recent_turns(n=4)
        ctx = "\n".join([f"{'用户' if t['role']=='user' else '助手'}: {t['content'][:300]}" for t in recent])
        messages = [
            {"role": "system", "content": "你是指代消解模块。将用户的追问改写为完整的、可独立理解的研究问题。只输出改写后的问题。"},
            {"role": "user", "content": f"上下文:\n{ctx}\n\n追问: {user_query}\n\n完整问题:"},
        ]
        result = await llm_client.chat(messages, temperature=0.1, max_tokens=200)
        resolved = result["content"].strip()
        logger.info(f"Resolved: '{user_query}' → '{resolved}'")
        return resolved
