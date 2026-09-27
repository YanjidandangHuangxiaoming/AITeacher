"""阿里 Paraformer 实时识别。

本地 VAD 已经把语音切成一整段，所以这里不需要真正的流式识别：开一次会话、
把整段音频喂完、拿最终文本。这样比全程保持长连接更省额度，也更好重试。
"""

from __future__ import annotations

from collections.abc import Iterator

import dashscope
import numpy as np
from dashscope.audio.asr import Recognition, RecognitionCallback

from ..config import AsrConfig
from .base import AsrClient, AsrError

# 每包 100ms，兼顾首包延迟与包数量
CHUNK_MS = 100


class _SentenceCollector(RecognitionCallback):
    """只收集句末结果，忽略中间态，避免把半句话重复拼进去。"""

    def __init__(self) -> None:
        self._sentences: list[str] = []
        self.error: str | None = None

    @property
    def text(self) -> str:
        return "".join(self._sentences).strip()

    def on_event(self, result) -> None:
        try:
            sentence = result.get_sentence()
        except Exception:
            return
        if not isinstance(sentence, dict) or not sentence.get("sentence_end"):
            return
        piece = sentence.get("text") or ""
        if piece:
            self._sentences.append(piece)

    def on_error(self, result) -> None:
        self.error = str(result)

    def on_open(self) -> None:
        pass

    def on_complete(self) -> None:
        pass

    def on_close(self) -> None:
        pass


class ParaformerAsr(AsrClient):
    def __init__(self, cfg: AsrConfig, sample_rate: int = 16000) -> None:
        self._cfg = cfg
        self._sample_rate = sample_rate
        if cfg.api_key:
            dashscope.api_key = cfg.api_key

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        collector = _SentenceCollector()
        recognition = Recognition(
            model=self._cfg.model,
            callback=collector,
            format="pcm",
            sample_rate=self._sample_rate,
        )

        recognition.start()
        try:
            for chunk in _iter_chunks(audio, max(1, self._sample_rate * CHUNK_MS // 1000)):
                recognition.send_audio_frame(chunk.tobytes())
        finally:
            # stop() 会 join 内部 worker 线程，是阻塞调用
            recognition.stop()

        if collector.error:
            raise AsrError(collector.error)
        return collector.text


def _iter_chunks(audio: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, audio.size, size):
        yield audio[start : start + size]
