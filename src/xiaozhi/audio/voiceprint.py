"""声纹识别：判断正在说话的是不是注册过的本人。

用 resemblyzer（GE2E speaker encoder，模型随包自带约 16MB）。用途是在打断判定里
排除旁人的语音。有两个必须知道的准确率限制：

  1. 蓝牙 HFP 是 16kHz 窄带，还叠加了耳机自身的降噪与自动增益，会削弱声纹特征；
  2. 打断判定能用的语音很短（约 0.6~1 秒），远少于可靠识别通常需要的 2~3 秒。

所以阈值做成注册时自动标定、也可手动覆盖，并且**拿不准时一律放行**——
漏掉一次该有的打断，比误拦一次你自己的打断更烦人。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import VoiceprintConfig

SAMPLE_RATE = 16000
# 注册时把整段录音切成几份分别求声纹再平均，比单次编码稳健
ENROLL_SEGMENTS = 3
DEFAULT_THRESHOLD = 0.75


class VoiceprintError(RuntimeError):
    pass


def to_float_audio(audio: np.ndarray) -> np.ndarray:
    """int16 → [-1, 1] 的 float32，resemblyzer 要求这个格式。"""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class SpeakerEncoder:
    """resemblyzer 编码器的惰性包装：torch 导入很慢，等到真要用了再加载。"""

    def __init__(self) -> None:
        self._encoder = None

    def warmup(self) -> None:
        """提前加载模型。

        torch 首次加载要好几秒，绝不能等到打断判定时才加载——那会直接把
        打断延迟从 0.2 秒拖到 5 秒。所以启动时就要预热。
        """
        self._model()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        wav = self._preprocess(audio)
        vector = np.asarray(self._model().embed_utterance(wav), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise VoiceprintError("声纹向量为空，这段音频可能没有有效语音")
        return vector / norm

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            raise VoiceprintError("音频为空")

        from resemblyzer import preprocess_wav

        wav = preprocess_wav(to_float_audio(audio), source_sr=SAMPLE_RATE)
        if wav.size == 0:
            raise VoiceprintError("这段音频里没有检测到有效语音")
        return wav

    def _model(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder

            self._encoder = VoiceEncoder(device="cpu", verbose=False)
        return self._encoder


@dataclass
class Enrollment:
    seconds: float
    self_similarity: float
    threshold: float
    created_at: str


class VoiceprintStore:
    """声纹的注册、持久化与比对。"""

    def __init__(
        self, cfg: VoiceprintConfig, prefix: Path, encoder: SpeakerEncoder | None = None
    ) -> None:
        self._cfg = cfg
        self._prefix = prefix
        self._encoder = encoder or SpeakerEncoder()
        self._embedding: np.ndarray | None = None
        self._meta: dict = {}

    @property
    def vector_path(self) -> Path:
        return self._prefix.with_suffix(".npy")

    @property
    def meta_path(self) -> Path:
        return self._prefix.with_suffix(".json")

    @property
    def enrolled(self) -> bool:
        return self._embedding is not None

    def warmup(self) -> None:
        """提前把声纹模型加载好，避免第一次判定时卡住。"""
        self._encoder.warmup()

    @property
    def threshold(self) -> float:
        if self._cfg.threshold is not None:
            return float(self._cfg.threshold)
        suggested = self._meta.get("suggested_threshold")
        return float(suggested) if suggested is not None else DEFAULT_THRESHOLD

    def load(self) -> bool:
        """从磁盘载入声纹。没有注册过或文件损坏时返回 False，不抛异常。"""
        try:
            vector = np.load(self.vector_path)
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False

        self._embedding = np.asarray(vector, dtype=np.float32)
        self._meta = meta if isinstance(meta, dict) else {}
        return True

    def enroll(self, audio: np.ndarray) -> Enrollment:
        seconds = audio.size / SAMPLE_RATE
        if audio.size == 0:
            raise VoiceprintError("没有录到任何音频")
        if seconds < self._cfg.min_seconds:
            raise VoiceprintError(
                f"录音只有 {seconds:.1f} 秒，至少需要 {self._cfg.min_seconds:.0f} 秒"
            )

        vectors = [
            self._encoder.embed(piece)
            for piece in np.array_split(audio, ENROLL_SEGMENTS)
            if piece.size > 0
        ]
        if not vectors:
            raise VoiceprintError("没能从录音里提取出声纹，换个安静环境重录")

        stacked = np.stack(vectors)
        embedding = stacked.mean(axis=0)
        embedding /= np.linalg.norm(embedding)

        self_similarity = _mean_pairwise_similarity(stacked)
        suggested = _suggest_threshold(self_similarity, len(vectors))
        created_at = datetime.now().isoformat(timespec="seconds")

        self._embedding = embedding
        self._meta = {
            "created_at": created_at,
            "seconds": round(seconds, 1),
            "segments": len(vectors),
            "self_similarity": round(self_similarity, 4),
            "suggested_threshold": suggested,
        }
        return Enrollment(seconds, self_similarity, suggested, created_at)

    def save(self) -> None:
        if self._embedding is None:
            raise VoiceprintError("还没有可保存的声纹，请先注册")
        self.vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vector_path, self._embedding)
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def matches(self, audio: np.ndarray) -> bool:
        """判断这段语音是不是本人。拿不准时返回 True（放行）。"""
        similarity = self.similarity(audio)
        if similarity is None:
            return True
        return similarity >= self.threshold

    def similarity(self, audio: np.ndarray) -> float | None:
        """与注册声纹的相似度；没注册或音频太短时返回 None。"""
        if self._embedding is None:
            return None
        if audio.size / SAMPLE_RATE < self._cfg.min_seconds:
            return None
        try:
            vector = self._encoder.embed(audio)
        except VoiceprintError:
            return None
        return cosine_similarity(vector, self._embedding)


def _mean_pairwise_similarity(stacked: np.ndarray) -> float:
    if len(stacked) < 2:
        return 1.0
    scores = [
        cosine_similarity(stacked[i], stacked[j])
        for i in range(len(stacked))
        for j in range(i + 1, len(stacked))
    ]
    return float(np.mean(scores))


def _suggest_threshold(self_similarity: float, segments: int) -> float:
    """自相似度越高说明录音越干净，阈值可以定得越严。只有一段时保守取默认值。"""
    if segments < 2:
        return DEFAULT_THRESHOLD
    return round(max(0.60, min(self_similarity - 0.12, 0.85)), 2)
