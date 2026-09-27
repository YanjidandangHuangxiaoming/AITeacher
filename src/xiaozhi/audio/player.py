"""流式 PCM 播放器，支持随时打断。"""

from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from .devices import AudioDeviceError, candidate_devices, describe_device


class StreamPlayer:
    def __init__(self, sample_rate: int, device: str | None = None) -> None:
        self._sample_rate = sample_rate
        self._interrupted = threading.Event()
        self._stream = self._open_stream(device)

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    @property
    def active(self) -> bool:
        return bool(self._stream.active)

    def open(self) -> None:
        self._stream.start()

    def close(self) -> None:
        for action in (self._stream.abort, self._stream.close):
            try:
                action()
            except Exception:
                pass

    def reset(self) -> None:
        """开始新一轮播报前清掉打断标记。"""
        self._interrupted.clear()
        self._restart_if_needed()

    def interrupt(self) -> None:
        """立刻停播并丢弃已缓冲的音频，之后不可恢复。"""
        self._interrupted.set()
        try:
            if self._stream.active:
                self._stream.abort()
        except Exception:
            pass

    def silence(self) -> None:
        """立刻停掉当前输出，但不设打断标志，之后还能接着播。"""
        try:
            if self._stream.active:
                self._stream.abort()
        except Exception:
            pass

    def write(self, pcm: bytes) -> bool:
        """写入一块 PCM。返回 False 表示已被打断，调用方应立即停止。"""
        if self._interrupted.is_set():
            return False

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return True

        self._restart_if_needed()
        try:
            self._stream.write(samples)
        except sd.PortAudioError:
            return False
        return not self._interrupted.is_set()

    def _open_stream(self, device: str | None) -> sd.OutputStream:
        failures: list[str] = []
        for index in candidate_devices(device, "output"):
            try:
                return sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype="int16",
                    device=index,
                    latency="low",
                )
            except Exception as exc:
                failures.append(f"{describe_device('output', index)}：{exc}")

        raise AudioDeviceError(
            f"扬声器打不开（采样率 {self._sample_rate}Hz）。已尝试：\n  "
            + "\n  ".join(failures)
            + "\n\n可以跑 python -m xiaozhi devices --probe 看哪些设备真的能用。"
        )

    def _restart_if_needed(self) -> None:
        # interrupt() 用了 abort()，会让流停止；下一轮写入前要先重新启动
        try:
            if not self._stream.active:
                self._stream.start()
        except Exception:
            pass
