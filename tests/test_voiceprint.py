"""声纹模块的单元测试。

用假编码器替代 resemblyzer：不需要 torch、不依赖真实语音，只验证逻辑
（相似度计算、阈值标定、持久化、拿不准时放行）。真实声学准确率必须用真人语音实测。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from xiaozhi.audio.voiceprint import (
    DEFAULT_THRESHOLD,
    VoiceprintError,
    VoiceprintStore,
    cosine_similarity,
)
from xiaozhi.config import VoiceprintConfig

SAMPLE_RATE = 16000

# 假编码器的向量构造：第 7 维是所有人共有的"人声"分量，
# 第 N 维是每个说话人独有的分量。这样同人相似度 = 1.0，异人约 0.73，
# 与 resemblyzer 在真实语音上的量级接近。
_SHARED = 1.0
_DISTINCT = 0.6


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.calls += 1
        if audio.size == 0:
            raise VoiceprintError("音频为空")
        speaker = int(audio[0]) // 1000
        vector = np.zeros(8, dtype=np.float32)
        vector[7] = _SHARED
        vector[speaker % 7] = _DISTINCT
        return vector / np.linalg.norm(vector)


def _audio(speaker: int, seconds: float = 3.0) -> np.ndarray:
    return np.full(int(SAMPLE_RATE * seconds), speaker * 1000, dtype=np.int16)


def _store(tmp_path, **overrides) -> VoiceprintStore:
    cfg = VoiceprintConfig(path=str(tmp_path / "voiceprint"), **overrides)
    return VoiceprintStore(cfg, tmp_path / "voiceprint", encoder=FakeEncoder())


def test_cosine_similarity_basics() -> None:
    same = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(same, same) == pytest.approx(1.0)
    assert cosine_similarity(same, other) == pytest.approx(0.0)


def test_cosine_similarity_handles_zero_vector() -> None:
    assert cosine_similarity(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)) == 0.0


def test_fake_encoder_matches_realistic_similarity_range() -> None:
    encoder = FakeEncoder()
    assert cosine_similarity(encoder.embed(_audio(1)), encoder.embed(_audio(1))) == pytest.approx(1.0)
    cross = cosine_similarity(encoder.embed(_audio(1)), encoder.embed(_audio(2)))
    assert 0.6 < cross < 0.8, "假编码器要让异人相似度落在真实声纹的量级上"


def test_enroll_computes_self_similarity(tmp_path) -> None:
    store = _store(tmp_path)
    enrollment = store.enroll(_audio(1))
    assert enrollment.self_similarity == pytest.approx(1.0)
    assert enrollment.seconds == pytest.approx(3.0, abs=0.1)
    assert store.enrolled is True


def test_enroll_rejects_too_short_audio(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(VoiceprintError):
        store.enroll(_audio(1, seconds=0.2))


def test_enroll_rejects_empty_audio(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(VoiceprintError):
        store.enroll(np.empty(0, dtype=np.int16))


def test_matches_own_voice(tmp_path) -> None:
    store = _store(tmp_path)
    store.enroll(_audio(1))
    assert store.matches(_audio(1)) is True


def test_rejects_other_speaker(tmp_path) -> None:
    store = _store(tmp_path)
    store.enroll(_audio(1))
    assert store.matches(_audio(2)) is False


def test_short_audio_fails_open(tmp_path) -> None:
    """太短的语音判不准，必须放行——宁可漏拦，也别拦掉用户自己的打断。"""
    store = _store(tmp_path)
    store.enroll(_audio(1))
    assert store.similarity(_audio(2, seconds=0.2)) is None
    assert store.matches(_audio(2, seconds=0.2)) is True


def test_not_enrolled_fails_open(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.matches(_audio(1)) is True


def test_save_and_load_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)
    store.enroll(_audio(1))
    store.save()
    assert store.vector_path.exists()
    assert store.meta_path.exists()

    reloaded = _store(tmp_path)
    assert reloaded.enrolled is False
    assert reloaded.load() is True
    assert reloaded.matches(_audio(1)) is True
    assert reloaded.matches(_audio(2)) is False


def test_load_without_files_returns_false(tmp_path) -> None:
    assert _store(tmp_path).load() is False


def test_enrollment_metadata_is_stored(tmp_path) -> None:
    store = _store(tmp_path)
    enrollment = store.enroll(_audio(1))
    store.save()
    meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
    assert meta["segments"] == 3
    assert meta["suggested_threshold"] == enrollment.threshold
    assert meta["self_similarity"] == pytest.approx(1.0)
    assert "created_at" in meta


def test_config_threshold_overrides_suggestion(tmp_path) -> None:
    store = _store(tmp_path, threshold=0.99)
    store.enroll(_audio(1))
    assert store.threshold == pytest.approx(0.99)


def test_loose_threshold_accepts_other_speaker(tmp_path) -> None:
    """阈值调到 0.70 时，假编码器里约 0.73 的异人也算通过——用来验证阈值真的生效。"""
    store = _store(tmp_path, threshold=0.70)
    store.enroll(_audio(1))
    assert store.matches(_audio(2)) is True

    strict = _store(tmp_path, threshold=0.90)
    strict.enroll(_audio(1))
    assert strict.matches(_audio(2)) is False


def test_default_threshold_used_without_meta(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.threshold == pytest.approx(DEFAULT_THRESHOLD)
