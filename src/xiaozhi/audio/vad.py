"""基于 webrtcvad 的语音分段器。

选 webrtcvad 而不是 silero-vad 的原因：silero 的 pip 包会拖进 torch（约 2GB），
而 webrtcvad 是 200KB 的 C 扩展、无需下载模型、离线可用。它只用来判断
"这 30ms 里有没有人声"，这就够驱动断句和打断了。检测器可注入，便于单测。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..config import VadConfig

SUPPORTED_FRAME_MS = (10, 20, 30)


class Event:
    pass


@dataclass(frozen=True)
class SpeechStart(Event):
    """检测到有人开始说话。"""


@dataclass(frozen=True)
class SpeechEnd(Event):
    """一句话说完了，audio 是这段语音的完整 PCM。"""

    audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))


SpeechDetector = Callable[[bytes], bool]


class VadSegmenter:
    def __init__(
        self,
        cfg: VadConfig,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        detector: SpeechDetector | None = None,
    ) -> None:
        if frame_ms not in SUPPORTED_FRAME_MS:
            raise ValueError(f"webrtcvad 只支持 {SUPPORTED_FRAME_MS} 毫秒的帧")

        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._detector = detector or self._build_detector(cfg.aggressiveness)

        self._preroll_frames = max(1, round(cfg.preroll_ms / frame_ms))
        self._min_speech_frames = max(1, math.ceil(cfg.min_speech_ms / frame_ms))
        self._silence_frames = max(1, math.ceil(cfg.silence_ms / frame_ms))
        self._max_frames = max(1, math.ceil(cfg.max_utterance_s * 1000 / frame_ms))

        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._preroll: list[np.ndarray] = []
        self._collected: list[np.ndarray] = []

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def process(self, frame: np.ndarray) -> list[Event]:
        """喂一帧音频，返回这一帧触发的所有事件（通常为空）。"""
        if frame.size == 0:
            return []

        try:
            has_voice = self._detector(frame.tobytes())
        except Exception:
            has_voice = False

        return self._advance(frame, has_voice)

    def _advance(self, frame: np.ndarray, has_voice: bool) -> list[Event]:
        if not self._in_speech:
            return self._advance_silence(frame, has_voice)
        return self._advance_speech(frame, has_voice)

    def _advance_silence(self, frame: np.ndarray, has_voice: bool) -> list[Event]:
        self._preroll.append(frame)
        if len(self._preroll) > self._preroll_frames:
            self._preroll.pop(0)

        self._speech_run = self._speech_run + 1 if has_voice else 0
        if self._speech_run < self._min_speech_frames:
            return []

        # 进入语音：把前导帧一并带上，避免吃掉第一个字
        self._in_speech = True
        self._collected = list(self._preroll)
        self._preroll = []
        self._silence_run = 0
        return [SpeechStart()]

    def _advance_speech(self, frame: np.ndarray, has_voice: bool) -> list[Event]:
        self._collected.append(frame)
        self._silence_run = 0 if has_voice else self._silence_run + 1

        if self._silence_run < self._silence_frames and len(self._collected) < self._max_frames:
            return []

        audio = np.concatenate(self._collected)
        self._reset_utterance()
        return [SpeechEnd(audio)]

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._preroll = []
        self._collected = []

    def _build_detector(self, aggressiveness: int) -> SpeechDetector:
        import webrtcvad

        vad = webrtcvad.Vad(aggressiveness)
        return lambda pcm_bytes: vad.is_speech(pcm_bytes, self._sample_rate)
