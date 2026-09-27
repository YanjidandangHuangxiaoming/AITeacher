"""识题客户端抽象：换服务商只需另写一个子类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VisionClient(ABC):
    @abstractmethod
    async def recognize(self, image_path: Path) -> str:
        """识别图片，返回描述题目内容的原始文本（约定为 JSON 字符串）。"""
