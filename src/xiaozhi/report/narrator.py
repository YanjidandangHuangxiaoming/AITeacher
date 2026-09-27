"""周报的叙述段落：给家长看的那几句人话。

统计数字由 weekly.py 算好，这里只负责把数字写成一段读得懂的话。
模型不可用时返回空串，报告照样出。
"""

from __future__ import annotations

from ..llm.client import DeepSeekClient
from .weekly import WeeklyReport, describe_stats

SYSTEM_PROMPT = """你是一位初中老师，正在给家长写一段简短的学习反馈。

要求：
- 3 到 4 句话，语气客观平和，不要夸张，不要用感叹号
- 先说他练了多少、掌握得怎么样，再说需要在哪方面多练
- 最后给一条具体、可操作的建议，比如"每天练两道配方法"，
  不要写"继续努力""多加练习"这种空话
- 只输出这段话本身，不要标题、不要列表、不要 Markdown
- 数据不足时（比如这周一道题都没讲）就直接说明情况，不要编造
"""


class WeeklyNarrator:
    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    async def narrate(self, report: WeeklyReport) -> str:
        payload = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": describe_stats(report)},
        ]
        try:
            text = "".join([piece async for piece in self._client.stream_chat(payload)])
        except Exception:
            return ""
        return " ".join(text.split())[:400]
