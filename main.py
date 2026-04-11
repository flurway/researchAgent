"""
ResearchGPT — FastAPI 入口

修复记录:
- [BUG FIX] 中文乱码: 自定义 UTF8JSONResponse 替代默认 JSONResponse
  原因: FastAPI 默认 JSONResponse 使用 json.dumps 的 ensure_ascii=True，
        中文被转义为 \\uXXXX。某些客户端（如 curl、浏览器直接访问）显示乱码。
  修复: 自定义 Response 类，强制 ensure_ascii=False + utf-8 编码。
"""
import os
import uuid
import json
import logging
from typing import Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import config
from rag.retriever import HybridRetriever
from rag.chunker import RecursiveChunker, StructuredChunker, DocumentChunk
from agent.orchestrator import ResearchAgent

logging.basicConfig(level=config.log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# [FIX] 自定义 JSON 响应类，解决中文乱码
# ============================================================
class UTF8JSONResponse(Response):
    """
    强制 UTF-8 编码的 JSON 响应

    FastAPI 默认的 JSONResponse 内部调用:
        json.dumps(content, ensure_ascii=True)
    这会把 "你好" 变成 "\\u4f60\\u597d"

    本类强制:
        json.dumps(content, ensure_ascii=False).encode("utf-8")
    确保中文正常显示
    """
    media_type = "application/json; charset=utf-8"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,   # ← 关键: 不转义非ASCII字符
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# ============================================================
# 全局状态
# ============================================================
retriever = HybridRetriever(embedding_dim=config.rag.embedding_dim)
agent: Optional[ResearchAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, "uploads"), exist_ok=True)
    index_path = config.rag.faiss_index_path
    if os.path.exists(f"{index_path}.faiss"):
        try:
            retriever.load_index(index_path)
            logger.info(f"Loaded index: {retriever.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
    agent = ResearchAgent(retriever)
    logger.info("ResearchGPT started")
    yield
    if retriever.faiss_index and retriever.faiss_index.ntotal > 0:
        retriever.save_index(index_path)
    agent.long_term_memory.save()
    logger.info("ResearchGPT shutdown")


app = FastAPI(
    title="ResearchGPT",
    description="AI 研究助手",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,  # ← 全局默认使用 UTF-8 JSON
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件 (前端)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# Request/Response 模型
# ============================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    response: str
    intent: str = ""
    citations: dict = {}
    needs_clarification: bool = False
    clarification_question: str = ""
    plan: Optional[dict] = None
    reflection: Optional[dict] = None


class DocUploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


# ============================================================
# API 端点
# ============================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        result = await agent.chat(user_message=request.message, session_id=request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocUploadResponse)
async def upload_document(file: UploadFile = File(...), chunking_strategy: str = "recursive"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    content = await file.read()
    text = ""
    if file.filename.endswith((".txt", ".md")):
        text = content.decode("utf-8", errors="ignore")
    elif file.filename.endswith(".pdf"):
        try:
            import fitz
            import io
            doc = fitz.open(stream=content, filetype="pdf")
            text = "\n\n".join([page.get_text() for page in doc])
            doc.close()
        except ImportError:
            raise HTTPException(status_code=400, detail="需要 PyMuPDF: pip install PyMuPDF")
    else:
        raise HTTPException(status_code=400, detail="支持 .txt .md .pdf")
    if not text.strip():
        raise HTTPException(status_code=400, detail="文档内容为空")

    doc_id = str(uuid.uuid4())[:8]
    chunker = (StructuredChunker if chunking_strategy == "structured" else RecursiveChunker)(
        chunk_size=config.rag.chunk_size, chunk_overlap=config.rag.chunk_overlap,
    )
    chunks = chunker.chunk_document(text, doc_id, file.filename, {"source": "upload", "filename": file.filename})

    if retriever.faiss_index is None:
        retriever.build_index(chunks)
    else:
        _add_chunks(chunks)
    retriever.save_index(config.rag.faiss_index_path)
    return DocUploadResponse(doc_id=doc_id, filename=file.filename, num_chunks=len(chunks),
                             message=f"已索引 {len(chunks)} 个片段")


def _add_chunks(new_chunks: list[DocumentChunk]):
    texts = [c.content for c in new_chunks]
    embeddings = retriever.get_embeddings(texts)
    retriever.faiss_index.add(embeddings)
    retriever.chunks.extend(new_chunks)
    retriever.bm25_index.add_documents(new_chunks)


@app.get("/documents/list")
async def list_documents():
    if not retriever.chunks:
        return {"documents": [], "total_chunks": 0}
    docs = {}
    for c in retriever.chunks:
        if c.doc_id not in docs:
            docs[c.doc_id] = {"doc_id": c.doc_id, "title": c.doc_title, "num_chunks": 0}
        docs[c.doc_id]["num_chunks"] += 1
    return {"documents": list(docs.values()), "total_chunks": len(retriever.chunks)}


@app.get("/memory/stats")
async def memory_stats(session_id: str = "default"):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    mem = agent.get_or_create_session(session_id)
    return {
        "short_term": mem.get_stats(),
        "long_term": agent.long_term_memory.get_stats(),
        "index": {"total_chunks": len(retriever.chunks),
                   "index_vectors": retriever.faiss_index.ntotal if retriever.faiss_index else 0},
    }


@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": retriever.faiss_index is not None,
            "index_size": retriever.faiss_index.ntotal if retriever.faiss_index else 0}


# 前端页面入口
@app.get("/")
async def index():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html; charset=utf-8")
    return {"message": "ResearchGPT API is running. POST /chat to start."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
