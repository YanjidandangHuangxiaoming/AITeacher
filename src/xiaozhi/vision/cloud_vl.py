"""通义千问-VL 识题，走 DashScope 的 OpenAI 兼容接口。"""

from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from ..config import VisionConfig
from ..utils.image import encode_image_data_uri
from .base import VisionClient

RECOGNIZE_PROMPT = """你是一位试卷数字化助手。请仔细观察这张图片，把上面的题目信息提取成 JSON。

要求：
1. 只输出 JSON 本体，不要任何解释文字，不要用 ``` 包裹。
2. 字段如下：
   subject: 学科，只能是 "数学"、"物理"、"化学"、"语文"、"英语"、"其他" 之一
   grade: 年级，如 "初一"、"初二"、"初三"；判断不出来就给空字符串
   stem: 完整题干文字，不要包含选项
   options: 选择题的选项数组，例如 ["A. 3", "B. 4"]；不是选择题就给空数组
   figure: 如果题目里有图形、表格或函数图像，用一段话把图里能看到的全部关键信息描述出来（角度、坐标、标注、数量关系）。没有图就给空字符串
   student_answer: 学生在图上手写的作答内容；没有就给空字符串
   notes: 图片模糊、被遮挡、条件不全等你需要提醒我的情况；没有就给空字符串
3. stem 必须是图片里真实存在的文字，不要自己补充或改写题目。
"""


class QwenVlClient(VisionClient):
    def __init__(self, cfg: VisionConfig) -> None:
        self._cfg = cfg
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
        )

    async def recognize(self, image_path: Path) -> str:
        data_uri = encode_image_data_uri(image_path)
        response = await self._client.chat.completions.create(
            model=self._cfg.model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": RECOGNIZE_PROMPT},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""
