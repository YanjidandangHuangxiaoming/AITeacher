"""识别客户端抽象：换服务商只需另写一个子类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AsrError(RuntimeError):
    pass


class AsrClient(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """识别一段 16-bit 单声道 PCM 语音。

        这是同步阻塞调用，必须放在线程里执行（编排器用 asyncio.to_thread 包）。
        """
