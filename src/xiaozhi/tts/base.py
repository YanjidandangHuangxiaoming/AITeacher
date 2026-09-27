"""合成客户端抽象：换服务商只需另写一个子类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

StopCheck = Callable[[], bool]


class TtsError(RuntimeError):
    pass


class TtsClient(ABC):
    @abstractmethod
    def stream(self, text: str, should_stop: StopCheck) -> Iterator[bytes]:
        """流式合成文本，逐块产出 16-bit 单声道 PCM。

        should_stop() 返回真时，实现方必须尽快终止本次合成并停止产出。
        """
