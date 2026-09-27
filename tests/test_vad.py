"""VAD 分段逻辑的单元测试。

检测器是注入的，所以不用真实音频也能验证"前导帧、起始阈值、静音断句、
超长截断"这几个关键行为。
"""

from __future__ import annotations

import numpy as np

from xiaozhi.audio.vad import SpeechEnd, SpeechStart, VadSegmenter
from xiaozhi.config import VadConfig

FRAME_MS = 30
FRAME_SAMPLES = 480


def _frame(tag: int = 0) -> np.ndarray:
    return np.full(FRAME_SAMPLES, tag, dtype=np.int16)


def _segmenter(*, speech_flags: list[bool], **overrides) -> tuple[VadSegmenter, list]:
    """按给定的每帧人声判定跑一遍，返回 (分段器, 事件列表)。"""
    defaults = dict(
        aggressiveness=2,
        preroll_ms=90,      # 3 帧
        min_speech_ms=60,   # 2 帧
        silence_ms=90,      # 3 帧
        max_utterance_s=1,  # 约 33 帧
        idle_timeout_s=0,
    )
    defaults.update(overrides)
    cfg = VadConfig(**defaults)

    flags = iter(speech_flags)
    vad = VadSegmenter(
        cfg, sample_rate=16000, frame_ms=FRAME_MS, detector=lambda _: next(flags, False)
    )

    events: list = []
    for index in range(len(speech_flags)):
        events.extend(vad.process(_frame(index)))
    return vad, events


def test_short_noise_does_not_trigger_speech() -> None:
    # 只有 1 帧人声，未达到 min_speech_ms=2 帧
    _, events = _segmenter(speech_flags=[False, True, False, False])
    assert events == []


def test_speech_start_and_end_are_emitted_once() -> None:
    _, events = _segmenter(
        speech_flags=[False, False, True, True, True, False, False, False, False]
    )
    kinds = [type(event) for event in events]
    assert kinds == [SpeechStart, SpeechEnd]


def test_speech_end_audio_contains_preroll() -> None:
    _, events = _segmenter(
        speech_flags=[False, True, True, True, False, False, False, False]
    )
    end = next(event for event in events if isinstance(event, SpeechEnd))
    # 前导 3 帧 + 语音帧，必须多于纯语音帧数，说明前导被带上了
    assert end.audio.size > 4 * FRAME_SAMPLES


def test_long_speech_is_force_cut() -> None:
    # 一直有人声，超过 max_utterance_s=1s（约 33 帧）必须强制断句
    _, events = _segmenter(speech_flags=[True] * 60)
    assert sum(isinstance(event, SpeechEnd) for event in events) >= 1


def test_two_utterances_in_a_row() -> None:
    flags = [True, True, True, False, False, False, False, True, True, True]
    _, events = _segmenter(speech_flags=flags)
    assert [type(event) for event in events] == [SpeechStart, SpeechEnd, SpeechStart]


def test_empty_frame_is_ignored() -> None:
    cfg = VadConfig(preroll_ms=90, min_speech_ms=60, silence_ms=90, idle_timeout_s=0)
    vad = VadSegmenter(cfg, 16000, FRAME_MS, detector=lambda _: True)
    assert vad.process(np.empty(0, dtype=np.int16)) == []


def test_invalid_frame_size_rejected() -> None:
    cfg = VadConfig()
    try:
        VadSegmenter(cfg, 16000, frame_ms=25, detector=lambda _: True)
    except ValueError as exc:
        assert "10" in str(exc)
    else:
        raise AssertionError("非 10/20/30ms 的帧必须被拒绝")
