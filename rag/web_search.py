"""
Web 搜索 + 网页抓取模块

面试考点: 为什么研究助手需要搜索能力？只靠知识库不行吗？

答:
1. 知识库是"静态的"——用户上传的文档有限，无法覆盖所有话题
2. 搜索是"动态的"——能获取最新的论文、博客、新闻
3. 两者互补: 知识库提供深度 (用户精选的核心文档)，搜索提供广度 (互联网上的海量信息)
4. Agent 可以根据知识库的检索结果决定是否需要补充搜索
   例: 知识库里有 RAG 的论文但没有最新的 benchmark，Agent 会自动搜索补充

工具设计:
- web_search: 搜索引擎查询，返回摘要列表 (轻量，用于发现信息)
- fetch_webpage: 抓取完整网页内容 (重量，用于深入阅读)

面试点: 为什么分两个工具而不是一个？
答: 搜索结果的摘要通常就够用了 (90% 的场景)，只有需要深入阅读时才抓全文。
    合并成一个工具会导致每次搜索都抓取 10 个网页，延迟爆炸且浪费 token。
"""
import re
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "web"         # web / news / arxiv
    metadata: dict = field(default_factory=dict)


@dataclass
class WebPage:
    url: str
    title: str
    content: str                # 提取后的正文
    content_length: int = 0


class WebSearcher:
    """
    Web 搜索引擎封装

    使用 duckduckgo-search (免费，无需 API Key)
    生产环境可替换为: Bing Search API / Google Custom Search / Tavily / SerpAPI
    
    面试延伸:
    Q: 搜索结果的质量怎么保证？
    A: 1. 多 query 策略: 一个问题生成多个搜索 query，覆盖不同角度
       2. 结果去重: 同一个 URL 只保留一次
       3. 优先级排序: 学术来源 (arxiv, scholar) > 技术博客 > 一般网页
       4. 搜索结果也走 rerank，只保留与原始问题相关的结果
    """

    def __init__(self, max_results: int = 8):
        self.max_results = max_results
        self._ddgs = None

    def _get_ddgs(self):
        if self._ddgs is None:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        return self._ddgs

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "cn-zh",
        search_type: str = "text",
    ) -> list[SearchResult]:
        """
        执行 Web 搜索

        Args:
            query: 搜索关键词
            max_results: 最大结果数
            region: 搜索区域 (cn-zh 中文优先, wt-wt 全球)
            search_type: text / news
        
        Returns:
            SearchResult 列表
        """
        k = max_results or self.max_results
        
        # 在线程池中执行同步搜索 (duckduckgo-search 是同步库)
        loop = asyncio.get_event_loop()
        try:
            if search_type == "news":
                raw_results = await loop.run_in_executor(
                    None, lambda: list(self._get_ddgs().news(query, max_results=k, region=region))
                )
            else:
                raw_results = await loop.run_in_executor(
                    None, lambda: list(self._get_ddgs().text(query, max_results=k, region=region))
                )
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

        results = []
        seen_urls = set()
        for item in raw_results:
            url = item.get("href") or item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            results.append(SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("body", "") or item.get("description", ""),
                source=self._classify_source(url),
                metadata={
                    "date": item.get("date", ""),
                    "source_name": item.get("source", ""),
                },
            ))

        logger.info(f"Web search: '{query}' → {len(results)} results")
        return results

    async def search_multi_query(
        self,
        queries: list[str],
        max_per_query: int = 5,
    ) -> list[SearchResult]:
        """
        多 query 搜索 + 去重融合

        面试点: 为什么要用多个 query？
        答: 单个 query 可能有信息偏差。
            例: "RAG 的缺点" 和 "RAG limitations challenges" 搜到的内容不同。
            多 query 能覆盖更多角度。
        """
        tasks = [self.search(q, max_results=max_per_query) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # 去重融合
        seen_urls = set()
        merged = []
        for results in all_results:
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    merged.append(r)

        # 按来源优先级排序
        source_priority = {"arxiv": 0, "scholar": 1, "github": 2, "blog": 3, "web": 4}
        merged.sort(key=lambda r: source_priority.get(r.source, 5))
        return merged

    def _classify_source(self, url: str) -> str:
        """根据 URL 分类来源"""
        url_lower = url.lower()
        if "arxiv.org" in url_lower:
            return "arxiv"
        elif "scholar.google" in url_lower or "semanticscholar" in url_lower:
            return "scholar"
        elif "github.com" in url_lower:
            return "github"
        elif any(d in url_lower for d in ["medium.com", "blog", "dev.to", "towardsdatascience", "zhihu.com", "csdn.net", "juejin.cn"]):
            return "blog"
        else:
            return "web"


class WebFetcher:
    """
    网页内容抓取器

    抓取网页并提取正文 (去除 HTML 标签、导航栏、广告等)
    
    面试点: 为什么不直接把整个 HTML 丢给 LLM？
    答: 1. HTML 标签、CSS、JS 占了 80%+ 的 token，正文只有 20%
       2. 导航栏、侧边栏、页脚等噪声会干扰 LLM 理解
       3. 清洗后的正文: 信噪比高，token 使用效率高
    """

    def __init__(self, timeout: int = 15, max_content_length: int = 8000):
        self.timeout = timeout
        self.max_content_length = max_content_length

    async def fetch(self, url: str) -> Optional[WebPage]:
        """
        抓取并提取网页正文
        """
        loop = asyncio.get_event_loop()
        try:
            page = await loop.run_in_executor(None, lambda: self._fetch_sync(url))
            return page
        except Exception as e:
            logger.error(f"Fetch failed: {url} → {e}")
            return None

    def _fetch_sync(self, url: str) -> Optional[WebPage]:
        import requests
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        resp = requests.get(url, headers=headers, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        html = resp.text

        title = self._extract_title(html)
        content = self._extract_content(html)

        if len(content) < 50:
            return None

        # 截断过长内容
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[内容已截断]"

        return WebPage(
            url=url,
            title=title,
            content=content,
            content_length=len(content),
        )

    def _extract_title(self, html: str) -> str:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_content(self, html: str) -> str:
        """
        从 HTML 提取正文

        简化版本 (生产环境用 readability-lxml 或 trafilatura):
        1. 移除 script/style/nav/header/footer 标签
        2. 移除所有 HTML 标签
        3. 合并空白行
        """
        # 移除不需要的标签块
        for tag in ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]:
            html = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', html, flags=re.IGNORECASE | re.DOTALL)

        # 移除 HTML 注释
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        # 把 <br> <p> <div> <li> <h1-6> 转为换行
        html = re.sub(r'<br\s*/?\s*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</(p|div|li|h[1-6]|tr|blockquote)>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<(p|div|li|h[1-6]|tr|blockquote)[^>]*>', '\n', html, flags=re.IGNORECASE)

        # 移除所有剩余 HTML 标签
        text = re.sub(r'<[^>]+>', '', html)

        # 解码 HTML 实体
        import html as html_lib
        text = html_lib.unescape(text)

        # 清理空白
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        # 合并过于碎片化的行 (连续单字行通常是导航残留)
        cleaned = []
        for line in lines:
            if len(line) > 10 or (cleaned and len(cleaned[-1]) > 10):
                cleaned.append(line)

        return "\n\n".join(cleaned)


# 全局单例
web_searcher = WebSearcher()
web_fetcher = WebFetcher()
