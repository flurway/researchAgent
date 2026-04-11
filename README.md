# ResearchGPT — AI 研究助手

基于 **Agent (Plan-and-Execute) + RAG (FAISS + BM25) + 长短期记忆** 的智能研究系统。

## 三种使用方式

### 方式一：命令行交互 (推荐本地测试)

不需要启动服务，直接跑：

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export DEEPSEEK_API_KEY="your-key"

# 上传文档 + 直接对话
python cli.py upload paper.pdf --chat

# 或者先上传，再对话
python cli.py upload paper1.pdf
python cli.py upload paper2.md --strategy structured
python cli.py chat

# 查看状态
python cli.py status
```

CLI 内置命令：
- `/upload <file>` — 上传文档
- `/docs` — 查看文档列表
- `/memory` — 查看记忆状态
- `/clear` — 清空当前会话
- `/session <id>` — 切换会话
- `/quit` — 退出

### 方式二：Web 界面

```bash
python main.py
# 打开 http://localhost:8000
```

左侧栏上传文档，右侧直接对话。支持显示意图类型、执行计划、引用来源。

### 方式三：API 调用

```bash
# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8000

# 上传文档
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@paper.pdf" -F "chunking_strategy=structured"

# 对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "RAG有哪些chunking策略？", "session_id": "user1"}'
```

## 项目结构

```
research_gpt/
├── main.py                  # FastAPI 入口 (UTF-8 修复)
├── cli.py                   # 命令行交互工具
├── config.py                # 全局配置
├── static/index.html        # Web 前端
├── agent/
│   ├── orchestrator.py      # 主编排器 (replan 修复)
│   ├── intent.py            # 意图判断 + 指代消解
│   ├── planner.py           # Plan-and-Execute 任务规划
│   ├── executor.py          # 工具执行器
│   ├── reflector.py         # 自我反思 + 幻觉检测
│   └── tools.py             # Agent 工具定义
├── rag/
│   ├── retriever.py         # 多路召回 (FAISS + BM25 + RRF)
│   ├── reranker.py          # 重排序 + 引用溯源
│   └── chunker.py           # 文档分块
└── memory/
    ├── short_term.py        # 短期记忆 (滑动窗口 + 摘要压缩)
    └── long_term.py         # 长期记忆 (FAISS 持久化)
```

## Bug 修复记录

### 1. 中文乱码 (main.py)
**原因**: FastAPI 默认 `JSONResponse` 内部用 `json.dumps(ensure_ascii=True)`，中文变 `\u4e2d\u6587`。
**修复**: 自定义 `UTF8JSONResponse` 类，强制 `ensure_ascii=False`，设为全局默认响应类。

### 2. Replan 失效 (orchestrator.py)
**原因**: `for step in plan["steps"]` 的迭代器在循环开始时绑定了原始列表。`plan = await replan(...)` 虽然重新赋值了变量，但 for 循环的迭代器仍指向老列表，新计划永远不会被执行。
**修复**: 改为 `while step_idx < len(steps)` 循环。Replan 后用新计划的步骤替换 `steps` 列表，并将 `step_idx` 重置为 0，从新计划第一步重新执行。同时加入 `replan_count` 上限防止无限重规划。

## 核心设计要点

### Agent 任务拆分
- Plan-and-Execute 架构，先全局规划再逐步执行
- 拆分粒度：每步是"可验证的阶段性成果"
- 失败时 replan，最多重规划 3 次

### 上下文构建
- RAG 结果优先占位（~4000 tokens），动态压缩对话历史
- 步骤间通过 depends_on 声明依赖，只注入相关上下文
- 锚定信息（用户约束）永不压缩

### 意图判断
- 5 类分流：chitchat / need_clarify / follow_up / direct_search / deep_research
- 模糊问题先澄清再检索，避免无效检索浪费 token
- Follow-up 做指代消解，确保检索 query 质量
