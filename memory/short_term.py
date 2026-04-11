"""
短期记忆 — 滑动窗口 + 摘要压缩 + 锚定信息
"""
import time
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    def __init__(self, max_turns: int = 20, summary_threshold: int = 10):
        self.turns: list[ConversationTurn] = []
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.summary: str = ""
        self.anchored_info: dict = {}
        self.session_id: str = ""
        self.topic_keywords: list[str] = []

    def add_turn(self, role: str, content: str, metadata: Optional[dict] = None):
        self.turns.append(ConversationTurn(
            role=role, content=content, metadata=metadata or {},
        ))
        if role == "user" and metadata and "keywords" in metadata:
            self.topic_keywords.extend(metadata["keywords"])
            self.topic_keywords = list(set(self.topic_keywords))[-20:]

    def get_recent_turns(self, n: int = 5) -> list[dict]:
        recent = self.turns[-n:] if n < len(self.turns) else self.turns
        return [{"role": t.role, "content": t.content} for t in recent]

    def set_anchor(self, key: str, value: str):
        self.anchored_info[key] = value

    def build_context_messages(
        self,
        system_prompt: str,
        current_query: str,
        rag_context: str = "",
        max_tokens: int = 12000,
    ) -> list[dict]:
        """
        构建 LLM 上下文

        token 预算动态分配:
        - RAG 结果多 → 压缩对话轮数 (3轮)
        - RAG 结果少 → 保留更多对话 (8轮)
        - 锚定信息永不压缩
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 摘要 + 锚定
        meta_parts = []
        if self.summary:
            meta_parts.append(f"[之前对话摘要]\n{self.summary}")
        if self.anchored_info:
            anchors = "\n".join([f"- {k}: {v}" for k, v in self.anchored_info.items()])
            meta_parts.append(f"[用户约束信息]\n{anchors}")
        if meta_parts:
            messages.append({"role": "system", "content": "\n\n".join(meta_parts)})

        # 动态对话轮数
        rag_token_est = len(rag_context) // 3
        if rag_token_est > 3000:
            keep_turns = min(3, len(self.turns))
        elif rag_token_est > 1500:
            keep_turns = min(5, len(self.turns))
        else:
            keep_turns = min(8, len(self.turns))

        recent = self.turns[-keep_turns:] if keep_turns < len(self.turns) else self.turns
        for turn in recent:
            messages.append({"role": turn.role, "content": turn.content})

        if rag_context:
            messages.append({
                "role": "system",
                "content": f"[检索到的相关文档]\n请基于以下内容回答，并标注引用来源 [来源N]:\n\n{rag_context}",
            })

        messages.append({"role": "user", "content": current_query})
        return messages

    async def compress_history(self, llm_client) -> str:
        if len(self.turns) <= self.summary_threshold:
            return self.summary

        to_compress = self.turns[:-5]
        self.turns = self.turns[-5:]

        history_text = "\n".join([
            f"{'用户' if t.role == 'user' else '助手'}: {t.content[:300]}"
            for t in to_compress
        ])
        messages = [
            {
                "role": "system",
                "content": "将以下对话压缩为摘要(≤300字)。保留: 研究主题、关键结论、约束条件、未解决问题。丢弃: 寒暄、重复。",
            },
            {"role": "user", "content": f"对话历史:\n{history_text}"},
        ]
        result = await llm_client.chat(messages, temperature=0.1, max_tokens=500)
        if self.summary:
            self.summary = f"{self.summary}\n---\n{result['content']}"
        else:
            self.summary = result["content"]
        return self.summary

    def get_stats(self) -> dict:
        return {
            "total_turns": len(self.turns),
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary),
            "anchored_keys": list(self.anchored_info.keys()),
            "topic_keywords": self.topic_keywords,
        }
