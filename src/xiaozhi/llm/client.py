"""DeepSeek 流式对话客户端（OpenAI 兼容协议）。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ..config import LlmConfig


class DeepSeekClient:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
        )

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._cfg.model,
            messages=messages,
            temperature=self._cfg.temperature,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
