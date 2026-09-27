"""一次辅导会话：系统提示词 + 多轮上下文。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..config import TutorConfig
from ..vision.question_parser import Question
from .client import DeepSeekClient

# 学生明确要求直接给答案时，临时覆盖分步引导策略
DIRECT_STYLE_OVERRIDE = """
# 本次会话临时调整

学生要求直接给答案。请先完整给出答案和解题步骤，最后补一句"哪里没看懂可以再问我"。
"""

OPENING_TEMPLATE = """我刚拍了一道题给你看，内容如下：

{question}

请开始辅导我。"""


class Tutor:
    def __init__(self, cfg: TutorConfig, system_prompt: str, client: DeepSeekClient) -> None:
        self._client = client
        prompt = system_prompt.replace("__GRADE__", cfg.grade)
        if cfg.style == "direct":
            prompt += DIRECT_STYLE_OVERRIDE
        self._system_prompt = prompt
        self._question: Question | None = None
        self._messages: list[dict[str, str]] = []

    @property
    def question(self) -> Question | None:
        return self._question

    @property
    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def start_question(self, question: Question) -> None:
        """开始新题：重置上下文。"""
        self._question = question
        self._messages = []

    def opening_prompt(self) -> str:
        """本题的第一句话，交给模型生成开场引导。"""
        if self._question is None:
            raise RuntimeError("还没有载入题目，请先调用 start_question()")
        return OPENING_TEMPLATE.format(question=self._question.to_prompt())

    def reset(self) -> None:
        self._question = None
        self._messages = []

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """把学生这句话交给模型，流式吐回讲解内容。"""
        self._messages.append({"role": "user", "content": user_text})
        collected: list[str] = []
        async for piece in self._client.stream_chat(self._payload()):
            collected.append(piece)
            yield piece
        self._messages.append({"role": "assistant", "content": "".join(collected)})

    def _payload(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self._system_prompt}, *self._messages]
