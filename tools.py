"""
Agent 工具集定义

面试考点: Agent 有哪些工具？工具的粒度怎么设计？

工具设计原则:
1. 原子性: 每个工具只做一件事，职责清晰
2. 可组合性: 工具之间可以串联 (检索→阅读→摘要)
3. 幂等性: 同样的输入，多次执行结果一致
4. 错误友好: 工具失败时返回有意义的错误信息，而不是崩溃

工具粒度决策 (面试重点):
- 太粗: "research_topic" 一个工具搞定所有 → Agent 没有推理空间，退化为普通 RAG
- 太细: "tokenize_text", "compute_tf", "compute_idf" → Agent 需要太多步骤，效率低
- 合适: 每个工具对应一个可独立验证的阶段性成果
  例: search_papers → 返回论文列表 (可验证: 论文是否相关？)
      read_document → 返回文档内容 (可验证: 是否是目标文档？)
      summarize → 返回摘要 (可验证: 摘要是否准确？)
"""

# ============================================================
# Function Calling 工具定义 (OpenAI 格式，DeepSeek 兼容)
# ============================================================

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从知识库中检索与查询相关的文档片段。"
                "适用于: 查找某个技术概念、方法论、实验结果等。"
                "返回: 相关文档片段列表 (含标题、来源、相关度得分)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询，应该是一个具体的问题或关键词组合",
                    },
                    "filters": {
                        "type": "object",
                        "description": "可选的过滤条件，如 {\"year\": \">=2023\", \"source\": \"arxiv\"}",
                        "properties": {
                            "year": {"type": "string"},
                            "source": {"type": "string"},
                            "author": {"type": "string"},
                        },
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的结果数量，默认 5",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_detail",
            "description": (
                "获取某个文档的更详细内容。"
                "当检索结果的片段不够完整时，使用此工具获取上下文。"
                "输入文档ID，返回该文档的完整内容或更大的片段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "文档的唯一标识符",
                    },
                    "section": {
                        "type": "string",
                        "description": "可选，指定查看的章节名称",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_content",
            "description": (
                "对一段较长的内容进行摘要。"
                "适用于: 论文摘要、报告总结、多文档综合。"
                "输入原文，返回结构化摘要。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "需要摘要的文本内容",
                    },
                    "focus": {
                        "type": "string",
                        "description": "摘要的关注点，如'方法论'、'实验结果'、'局限性'",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "摘要的最大字数，默认 300",
                        "default": 300,
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_concepts",
            "description": (
                "对比分析两个或多个技术概念/方法。"
                "适用于: 技术选型对比、方法论比较、框架对比。"
                "返回结构化的对比分析。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要对比的概念列表",
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "对比维度，如 ['性能', '复杂度', '适用场景']",
                    },
                },
                "required": ["concepts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_research_report",
            "description": (
                "基于已收集的信息，生成结构化的研究报告。"
                "这是研究流程的最后一步，在收集和分析足够信息后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "研究主题",
                    },
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "已收集的关键发现列表",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["brief", "detailed", "academic"],
                        "description": "报告格式: brief(简报), detailed(详细), academic(学术风格)",
                        "default": "detailed",
                    },
                },
                "required": ["topic", "findings"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_clarification",
            "description": (
                "当信息不足以继续研究时，向用户提出澄清问题。"
                "只在确实需要用户输入时使用，不要过度询问。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "向用户提出的问题",
                    },
                    "reason": {
                        "type": "string",
                        "description": "为什么需要这个信息",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选项，帮助用户快速回答",
                    },
                },
                "required": ["question", "reason"],
            },
        },
    },
]