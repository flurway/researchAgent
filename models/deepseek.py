"""
DeepSeek API 封装 — 异步调用 + 并发控制 + 重试
"""
import json
import asyncio
import logging
from typing import Optional
from openai import AsyncOpenAI
from config import config

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.deepseek.api_key,
            base_url=config.deepseek.base_url,
        )
        self._semaphore = asyncio.Semaphore(config.deepseek.max_concurrent_calls)

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> dict:
        async with self._semaphore:
            return await self._call_with_retry(
                messages=messages,
                tools=tools,
                model=model or config.deepseek.model,
                temperature=temperature if temperature is not None else config.deepseek.temperature,
                max_tokens=max_tokens or config.deepseek.max_tokens,
                response_format=response_format,
            )

    async def _call_with_retry(self, max_retries: int = 3, **kwargs) -> dict:
        for attempt in range(max_retries):
            try:
                params = {
                    "model": kwargs["model"],
                    "messages": kwargs["messages"],
                    "temperature": kwargs["temperature"],
                    "max_tokens": kwargs["max_tokens"],
                }
                if kwargs.get("tools"):
                    params["tools"] = kwargs["tools"]
                    params["tool_choice"] = "auto"
                if kwargs.get("response_format"):
                    params["response_format"] = kwargs["response_format"]

                response = await self.client.chat.completions.create(**params)
                choice = response.choices[0]

                tool_calls = None
                if choice.message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                return {
                    "content": choice.message.content or "",
                    "tool_calls": tool_calls,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(wait_time)

    async def get_json_response(self, messages: list[dict], model: Optional[str] = None) -> dict:
        result = await self.chat(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            return json.loads(result["content"])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {result['content'][:200]}")
            return {}


llm_client = DeepSeekClient()
