#!/usr/bin/env python3
"""
ResearchGPT 本地命令行交互工具

使用方式:
  # 交互式对话 (不需要启动 FastAPI 服务)
  python cli.py chat

  # 上传文档到知识库
  python cli.py upload paper.pdf
  python cli.py upload notes.md --strategy structured

  # 查看知识库状态
  python cli.py status

  # 上传文档 + 直接开始对话
  python cli.py upload paper.pdf --chat
"""
import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(__file__))

from config import config
from rag.retriever import HybridRetriever
from rag.chunker import RecursiveChunker, StructuredChunker
from agent.orchestrator import ResearchAgent

logging.basicConfig(
    level=logging.WARNING,  # CLI 模式下只显示警告以上
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================================
# 终端颜色
# ============================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    PURPLE = "\033[35m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def print_banner():
    print(f"""
{C.PURPLE}{C.BOLD}╔══════════════════════════════════════╗
║         ResearchGPT CLI              ║
║   AI 研究助手 · 本地交互模式          ║
╚══════════════════════════════════════╝{C.RESET}
""")


def print_help():
    print(f"""{C.DIM}
命令:
  /upload <file>    上传文档到知识库
  /search <query>   直接 Web 搜索（不走 Agent 规划）
  /docs             查看已索引文档
  /memory           查看记忆状态
  /clear            清空当前会话
  /session <id>     切换会话
  /help             显示帮助
  /quit             退出

提示: 直接输入问题即可，即使没有上传文档也能通过 Web 搜索回答
{C.RESET}""")


# ============================================================
# 核心功能
# ============================================================
class CLIApp:
    def __init__(self):
        self.retriever = HybridRetriever(embedding_dim=config.rag.embedding_dim)
        self.agent: ResearchAgent = None
        self.session_id = "cli_default"
        self._initialized = False

    def initialize(self):
        """初始化 Agent 和索引"""
        os.makedirs(config.data_dir, exist_ok=True)

        # 尝试加载已有索引
        index_path = config.rag.faiss_index_path
        if os.path.exists(f"{index_path}.faiss"):
            try:
                self.retriever.load_index(index_path)
                print(f"{C.GREEN}✓ 已加载索引: {self.retriever.faiss_index.ntotal} 个向量, "
                      f"{len(self.retriever.chunks)} 个文档片段{C.RESET}")
            except Exception as e:
                print(f"{C.YELLOW}⚠ 加载索引失败: {e}{C.RESET}")

        self.agent = ResearchAgent(self.retriever)
        self._initialized = True

    def upload_document(self, filepath: str, strategy: str = "recursive"):
        """上传文档"""
        if not os.path.exists(filepath):
            print(f"{C.RED}✗ 文件不存在: {filepath}{C.RESET}")
            return False

        filename = os.path.basename(filepath)
        print(f"{C.CYAN}⟳ 正在处理: {filename}{C.RESET}")

        # 读取文件
        text = ""
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif filepath.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(filepath)
                text = "\n\n".join([page.get_text() for page in doc])
                doc.close()
            except ImportError:
                print(f"{C.RED}✗ 需要 PyMuPDF: pip install PyMuPDF{C.RESET}")
                return False
        else:
            print(f"{C.RED}✗ 不支持的格式，请使用 .txt .md .pdf{C.RESET}")
            return False

        if not text.strip():
            print(f"{C.RED}✗ 文档内容为空{C.RESET}")
            return False

        # 分块
        import uuid
        doc_id = str(uuid.uuid4())[:8]

        if strategy == "structured":
            chunker = StructuredChunker(chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap)
        else:
            chunker = RecursiveChunker(chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap,
                                       min_chunk_size=config.rag.min_chunk_size)

        chunks = chunker.chunk_document(text, doc_id, filename, {"source": "cli_upload", "filename": filename})

        if not chunks:
            print(f"{C.RED}✗ 分块结果为空{C.RESET}")
            return False

        # 建立/更新索引
        if self.retriever.faiss_index is None:
            self.retriever.build_index(chunks)
        else:
            import numpy as np
            texts = [c.content for c in chunks]
            embeddings = self.retriever.get_embeddings(texts)
            self.retriever.faiss_index.add(embeddings)
            self.retriever.chunks.extend(chunks)
            self.retriever.bm25_index.add_documents(chunks)

        # 保存索引
        self.retriever.save_index(config.rag.faiss_index_path)

        print(f"{C.GREEN}✓ 已索引: {filename} → {len(chunks)} 个片段{C.RESET}")
        print(f"{C.DIM}  知识库总计: {self.retriever.faiss_index.ntotal} 个向量{C.RESET}")
        return True

    async def chat_once(self, message: str) -> dict:
        """发送一条消息"""
        if not self._initialized:
            self.initialize()
        return await self.agent.chat(user_message=message, session_id=self.session_id)

    def show_docs(self):
        """显示文档列表"""
        if not self.retriever.chunks:
            print(f"{C.DIM}知识库为空，请先上传文档{C.RESET}")
            return
        docs = {}
        for c in self.retriever.chunks:
            if c.doc_id not in docs:
                docs[c.doc_id] = {"title": c.doc_title, "chunks": 0}
            docs[c.doc_id]["chunks"] += 1
        print(f"\n{C.BOLD}📚 知识库文档 ({len(docs)} 个){C.RESET}")
        for did, info in docs.items():
            print(f"  {C.CYAN}•{C.RESET} {info['title']}  {C.DIM}({info['chunks']} 片段, id={did}){C.RESET}")
        print(f"\n{C.DIM}总计: {len(self.retriever.chunks)} 个片段, "
              f"{self.retriever.faiss_index.ntotal if self.retriever.faiss_index else 0} 个向量{C.RESET}\n")

    def show_memory(self):
        """显示记忆状态"""
        if not self.agent:
            print(f"{C.DIM}Agent 未初始化{C.RESET}")
            return
        mem = self.agent.get_or_create_session(self.session_id)
        stats = mem.get_stats()
        lt = self.agent.long_term_memory.get_stats()
        print(f"\n{C.BOLD}🧠 记忆状态{C.RESET}")
        print(f"  短期记忆: {stats['total_turns']} 轮对话")
        if stats['has_summary']:
            print(f"  历史摘要: {stats['summary_length']} 字")
        if stats['anchored_keys']:
            print(f"  锚定信息: {', '.join(stats['anchored_keys'])}")
        if stats['topic_keywords']:
            print(f"  主题关键词: {', '.join(stats['topic_keywords'][:10])}")
        print(f"  长期记忆: {lt['total_memories']} 条")
        print()


def format_response(result: dict):
    """格式化输出 Agent 响应"""
    intent = result.get("intent", "")
    intent_colors = {
        "direct_search": C.BLUE, "deep_research": C.PURPLE,
        "need_clarify": C.YELLOW, "chitchat": C.GRAY, "follow_up": C.GREEN,
    }
    ic = intent_colors.get(intent, C.DIM)

    # Intent badge
    print(f"\n{ic}[{intent}]{C.RESET}", end="")

    # Plan info
    plan = result.get("plan")
    if plan and plan.get("steps"):
        steps = plan["steps"]
        print(f" {C.DIM}· {plan.get('complexity','?')} · {len(steps)} 步计划{C.RESET}")
        for s in steps:
            print(f"  {C.DIM}{s.get('step_id','-')}. {s.get('description','')[:60]}{C.RESET}")
    else:
        print()

    # Response body
    print(f"\n{C.BOLD}{result['response']}{C.RESET}")

    # Citations
    citations = result.get("citations", {})
    if citations:
        print(f"\n{C.DIM}📎 引用来源:{C.RESET}")
        for cid, info in citations.items():
            sec = f" > {info['section']}" if info.get('section') else ""
            src = info.get('source', '')
            url_part = f"\n     {C.CYAN}{src}{C.RESET}" if src.startswith('http') else ""
            print(f"  {C.DIM}[{cid}] {info.get('doc_title','')}{sec}{C.RESET}{url_part}")

    # Reflection
    reflection = result.get("reflection")
    if reflection and reflection.get("scores"):
        sc = reflection["scores"]
        print(f"\n{C.DIM}📊 质量评分: 充分性={sc.get('sufficiency','?')} 一致性={sc.get('consistency','?')} "
              f"覆盖度={sc.get('coverage','?')} 引用={sc.get('citation_quality','?')}{C.RESET}")

    print()


# ============================================================
# 主循环
# ============================================================
async def interactive_loop(app: CLIApp):
    """交互式对话循环"""
    print_banner()
    app.initialize()
    print_help()

    while True:
        try:
            user_input = input(f"{C.GREEN}你: {C.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C.DIM}再见!{C.RESET}")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit" or cmd == "/exit":
                print(f"{C.DIM}再见!{C.RESET}")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/docs":
                app.show_docs()
            elif cmd == "/memory":
                app.show_memory()
            elif cmd == "/clear":
                app.session_id = f"cli_{datetime.now().strftime('%H%M%S')}"
                print(f"{C.GREEN}✓ 会话已清空，新会话: {app.session_id}{C.RESET}")
            elif cmd == "/session":
                if len(parts) > 1:
                    app.session_id = parts[1]
                    print(f"{C.GREEN}✓ 切换到会话: {app.session_id}{C.RESET}")
                else:
                    print(f"{C.DIM}当前会话: {app.session_id}{C.RESET}")
            elif cmd == "/upload":
                if len(parts) > 1:
                    filepath = parts[1].strip()
                    strategy = "structured" if filepath.endswith(".md") else "recursive"
                    app.upload_document(filepath, strategy)
                else:
                    print(f"{C.YELLOW}用法: /upload <文件路径>{C.RESET}")
            elif cmd == "/search":
                if len(parts) > 1:
                    query = parts[1].strip()
                    print(f"{C.DIM}⟳ 搜索中: {query}{C.RESET}")
                    try:
                        from rag.web_search import web_searcher
                        results = await web_searcher.search(query)
                        if results:
                            print(f"\n{C.BOLD}🔍 搜索结果 ({len(results)} 条){C.RESET}")
                            for i, r in enumerate(results, 1):
                                icon = {"arxiv": "📄", "scholar": "🎓", "github": "💻", "blog": "📝"}.get(r.source, "🔗")
                                print(f"  {C.CYAN}{i}. {icon} {r.title}{C.RESET}")
                                print(f"     {C.DIM}{r.snippet[:120]}{C.RESET}")
                                print(f"     {C.DIM}{r.url}{C.RESET}")
                            print(f"\n{C.DIM}提示: 直接提问即可，Agent 会自动决定是否搜索{C.RESET}\n")
                        else:
                            print(f"{C.YELLOW}未找到结果{C.RESET}")
                    except Exception as e:
                        print(f"{C.RED}搜索失败: {e}{C.RESET}")
                else:
                    print(f"{C.YELLOW}用法: /search <关键词>{C.RESET}")
            else:
                print(f"{C.YELLOW}未知命令: {cmd}，输入 /help 查看帮助{C.RESET}")
            continue

        # 发送消息
        print(f"{C.DIM}⟳ 思考中...{C.RESET}", end="\r")
        try:
            result = await app.chat_once(user_input)
            # 清除 "思考中" 行
            print(" " * 40, end="\r")
            format_response(result)
        except Exception as e:
            print(f"\n{C.RED}✗ 错误: {e}{C.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="ResearchGPT CLI")
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # chat 子命令
    chat_p = sub.add_parser("chat", help="交互式对话")
    chat_p.add_argument("--session", default="cli_default", help="会话 ID")

    # upload 子命令
    upload_p = sub.add_parser("upload", help="上传文档")
    upload_p.add_argument("file", help="文件路径")
    upload_p.add_argument("--strategy", choices=["recursive", "structured"], default="recursive")
    upload_p.add_argument("--chat", action="store_true", help="上传后直接进入对话")

    # status 子命令
    sub.add_parser("status", help="查看系统状态")

    args = parser.parse_args()

    app = CLIApp()

    if args.command == "upload":
        app.initialize()
        ok = app.upload_document(args.file, args.strategy)
        if ok and args.chat:
            asyncio.run(interactive_loop(app))

    elif args.command == "status":
        app.initialize()
        app.show_docs()
        app.show_memory()
        print(f"{C.BOLD}系统状态{C.RESET}")
        print(f"  DeepSeek API Key: {'✓ 已设置' if config.deepseek.api_key else '✗ 未设置 (export DEEPSEEK_API_KEY=...)'}")
        print(f"  Embedding 模型: {config.rag.embedding_model}")
        print(f"  数据目录: {config.data_dir}")
        print()

    elif args.command == "chat" or args.command is None:
        if args.command == "chat":
            app.session_id = args.session
        asyncio.run(interactive_loop(app))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
