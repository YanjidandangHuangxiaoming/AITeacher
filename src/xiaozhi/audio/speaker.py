"""播报出口：把文本变成声音，支持暂停续播与真打断。

speak() 必须在工作线程里调用——TTS 的 complete/cancel 都是阻塞调用，
放进事件循环会卡住整个对话。

三种停止语义必须分清：
  pause()     立刻静音，但保留后续音频，resume() 能接着播（用于声纹待判定）
  resume()    从暂停处继续
  interrupt() 真打断，丢弃剩余音频，不可恢复

为什么需要「代号」（generation）：
  speak() 跑在 to_thread 的工作线程里，取消 asyncio task 并不会让那个线程
  停下来。如果新一轮回复只是清一下标志位，旧线程会继续往同一个输出流里写，
  新旧两段音频交织在一起——听起来就是"两个声音在同时说话"。
  所以每次 begin() 都作废旧代号，旧线程自己发现代号过期就退出。
"""

from __future__ import annotations

import threading
import time

from ..tts.base import TtsClient
from .player import StreamPlayer

# 暂停的兜底上限：万一判定流程出岔子，也不能让播报永远卡住
MAX_PAUSE_S = 5.0


class SpeechOutput:
    def __init__(self, tts: TtsClient, player: StreamPlayer) -> None:
        self._tts = tts
        self._player = player
        self._speaking = threading.Event()
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self._gate = threading.Condition()
        self._generation = 0

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def interrupted(self) -> bool:
        """播报是否已被中止。也要看播放器的标志，否则编排器会继续往已停的流里塞句子。"""
        return self._stopped.is_set() or self._player.interrupted

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def open(self) -> None:
        self._player.open()

    def close(self) -> None:
        self._player.close()

    def begin(self) -> None:
        """新一轮播报开始：作废还在播的旧音频，并清掉暂停与打断状态。"""
        with self._gate:
            self._generation += 1
            self._paused.clear()
            self._stopped.clear()
            self._gate.notify_all()
        self._player.reset()

    def speak(self, text: str) -> bool:
        """朗读一段文本。返回是否完整播完，被打断或被作废则返回 False。"""
        if not text.strip():
            return True

        with self._gate:
            generation = self._generation

        self._speaking.set()
        try:
            for chunk in self._tts.stream(text, lambda: self._is_stale(generation)):
                if not self._wait_until_playable(generation):
                    return False
                if not self._player.write(chunk):
                    return False
            return not self._is_stale(generation)
        finally:
            self._speaking.clear()

    def pause(self) -> None:
        """立刻静音，保留后续音频以便续播。"""
        with self._gate:
            self._paused.set()
        self._player.silence()

    def resume(self) -> None:
        with self._gate:
            self._paused.clear()
            self._gate.notify_all()

    def interrupt(self) -> None:
        with self._gate:
            self._stopped.set()
            self._paused.clear()
            self._gate.notify_all()
        self._player.interrupt()

    # ---------- 内部：代号与暂停闸门 ----------

    def _is_stale(self, generation: int) -> bool:
        with self._gate:
            return self._is_stale_locked(generation)

    def _is_stale_locked(self, generation: int) -> bool:
        """调用前必须持有 self._gate。"""
        return self._stopped.is_set() or generation != self._generation

    def _wait_until_playable(self, generation: int) -> bool:
        """阻塞到可以继续播。返回 False 表示已被打断或被作废。"""
        deadline = time.monotonic() + MAX_PAUSE_S
        with self._gate:
            while self._paused.is_set() and not self._is_stale_locked(generation):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._paused.clear()  # 兜底：暂停超时自动续播，避免卡死
                    break
                self._gate.wait(timeout=min(0.2, remaining))
            return not self._is_stale_locked(generation)
