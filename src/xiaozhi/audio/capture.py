"""麦克风采集：sounddevice 回调把定长帧塞进队列。

只用一个队列、一个消费者，所以不需要额外的环形缓冲——queue.Queue 本身
就是线程安全的定长环形结构。多消费者共享游标的需求在 P1 并不存在：
唤醒由 ASR 关键词完成，VAD 只由编排器单路读取。
"""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .devices import AudioDeviceError, candidate_devices, describe_device


class AudioCapture:
    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self._sample_rate = cfg.sample_rate
        self._frame_samples = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=cfg.queue_frames)
        self._stream = self._open_stream()

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        try:
            self._stream.abort()
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    def frame(self, timeout: float = 0.5) -> np.ndarray | None:
        """取一帧 16-bit 单声道音频，超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _open_stream(self) -> sd.InputStream:
        failures: list[str] = []
        for index in candidate_devices(self._cfg.input_device, "input"):
            try:
                return sd.InputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self._frame_samples,
                    device=index,
                    callback=self._on_audio,
                )
            except Exception as exc:
                failures.append(f"{describe_device('input', index)}：{exc}")

        raise AudioDeviceError(
            f"麦克风打不开（采样率 {self._sample_rate}Hz）。已尝试：\n  "
            + "\n  ".join(failures)
            + "\n\n可以跑 python -m xiaozhi devices --probe 看哪些设备真的能用。"
        )

    def _on_audio(self, indata, frames, time_info, status) -> None:
        # 回调里只做拷贝和入队，任何耗时操作都会导致丢帧
        if frames != self._frame_samples:
            return
        mono = indata[:, 0].copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # 处理不过来时丢最旧的一帧，优先保证低延迟
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono)
            except (queue.Empty, queue.Full):
                pass
