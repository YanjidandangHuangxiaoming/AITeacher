"""图片 → 题目：截图、本地图片、以后的微信图片共用这一条流水线。"""

from __future__ import annotations

from pathlib import Path

from ..vision.base import VisionClient
from ..vision.question_parser import Question, parse_question


class IngestPipeline:
    def __init__(self, vision: VisionClient) -> None:
        self._vision = vision

    async def to_question(self, image_path: Path) -> Question:
        raw = await self._vision.recognize(image_path)
        return parse_question(raw)
