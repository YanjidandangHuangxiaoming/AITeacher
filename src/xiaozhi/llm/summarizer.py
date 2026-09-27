"""结题归档：让模型根据整段对话判断掌握情况和知识点。

每题只在结题时调用一次，不是每轮都调，所以成本可接受。
归档失败不能让主流程受影响，一律降级成"未知"。
"""

from __future__ import annotations

from ..store.models import (
    MASTERY_FAILED,
    MASTERY_MASTERED,
    MASTERY_PARTIAL,
    QuestionSummary,
)
from ..utils.json import extract_json_object
from .client import DeepSeekClient

VALID_MASTERY = {MASTERY_MASTERED, MASTERY_PARTIAL, MASTERY_FAILED}
MAX_TRANSCRIPT_CHARS = 6000

SYSTEM_PROMPT = """你是一位教学记录助手。我会给你一段我和初中生围绕一道题的完整对话，
请客观总结这次辅导的情况。

只输出 JSON，不要任何解释，不要用 ``` 包裹：
{"topics": ["知识点1"], "mastery": "掌握", "summary": "一句话说明他卡在哪，或哪里做得好"}

要求：
- topics 给 1~3 个，具体到初中知识点的粒度，比如"一元二次方程-配方法"，不要写"数学"这种大词
- mastery 只能取 "掌握"、"部分掌握"、"未掌握" 三者之一
- summary 控制在 40 字以内，客观描述，不要夸张，不要用"非常棒"这类空话
"""


class QuestionSummarizer:
    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    async def summarize(self, messages: list[dict[str, str]]) -> QuestionSummary:
        transcript = render_transcript(messages)
        if not transcript:
            return QuestionSummary()

        payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        try:
            text = "".join([piece async for piece in self._client.stream_chat(payload)])
        except Exception:
            return QuestionSummary()
        return parse_summary(text)


def render_transcript(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        speaker = "学生" if message["role"] == "user" else "小智"
        lines.append(f"{speaker}：{message['content']}")
    return "\n".join(lines)[:MAX_TRANSCRIPT_CHARS]


def parse_summary(text: str) -> QuestionSummary:
    payload = extract_json_object(text)
    if payload is None:
        return QuestionSummary()

    raw_topics = payload.get("topics")
    if isinstance(raw_topics, str):
        raw_topics = [raw_topics]
    topics = (
        [str(item).strip() for item in raw_topics if str(item).strip()]
        if isinstance(raw_topics, list)
        else []
    )

    mastery = str(payload.get("mastery") or "").strip()
    if mastery not in VALID_MASTERY:
        mastery = ""

    return QuestionSummary(
        topics=topics[:3],
        mastery=mastery,
        summary=str(payload.get("summary") or "").strip()[:80],
    )
