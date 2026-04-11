"""
Agent 工具定义 (OpenAI Function Calling 格式，DeepSeek 兼容)
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "从知识库中检索与查询相关的文档片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询"},
                    "filters": {"type": "object", "description": "过滤条件", "properties": {
                        "year": {"type": "string"}, "source": {"type": "string"}, "author": {"type": "string"},
                    }},
                    "top_k": {"type": "integer", "description": "返回数量", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_detail",
            "description": "获取某个文档的更详细内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "文档ID"},
                    "section": {"type": "string", "description": "章节名称"},
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_content",
            "description": "对内容进行摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "需要摘要的文本"},
                    "focus": {"type": "string", "description": "关注点"},
                    "max_length": {"type": "integer", "default": 300},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_concepts",
            "description": "对比分析多个概念/方法。",
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {"type": "array", "items": {"type": "string"}, "description": "对比概念列表"},
                    "dimensions": {"type": "array", "items": {"type": "string"}, "description": "对比维度"},
                },
                "required": ["concepts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_research_report",
            "description": "基于已收集信息生成研究报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "研究主题"},
                    "findings": {"type": "array", "items": {"type": "string"}, "description": "关键发现"},
                    "format": {"type": "string", "enum": ["brief", "detailed", "academic"], "default": "detailed"},
                },
                "required": ["topic", "findings"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_clarification",
            "description": "向用户提出澄清问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "向用户的问题"},
                    "reason": {"type": "string", "description": "为什么需要这个信息"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "可选项"},
                },
                "required": ["question", "reason"],
            },
        },
    },
]
