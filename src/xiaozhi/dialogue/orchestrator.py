"""对话编排：串联采集 → VAD → ASR → LLM → TTS，并处理唤醒与打断。

线程模型：
  - sounddevice 回调线程 → 帧队列（只拷贝，不做别的）
  - audio-reader 线程   → 把帧转投到 asyncio 队列
  - 事件循环            → VAD 分段、状态机、LLM 流
  - to_thread 工作线程  → ASR（同步阻塞）、TTS 播报（同步阻塞）

打断有两条路径，由 interrupt.mode 决定：

  vad        检测到人声立刻停播（快，但旁人说话也会打断）
  voiceprint 先立刻静音，同时收 0.6 秒语音做声纹比对：
             是本人 → 真打断；不是 → 无声续播。
             真打断的响应速度不受影响，只有"认错人"时才会多一个短暂停顿。

打断的本质是一次取消传播：speaker.interrupt() 让播放停下并丢弃缓冲，
再取消 response task，状态回落 LISTENING。
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..asr.base import AsrClient
from ..audio.capture import AudioCapture
from ..audio.speaker import SpeechOutput
from ..audio.vad import Event, SpeechEnd, SpeechStart, VadSegmenter
from ..audio.voiceprint import VoiceprintStore
from ..config import Settings
from ..llm.tutor import Tutor
from ..utils.text import take_sentences
from .state import DialogueState
from .wake import WakeWordMatcher

WAKE_REPLY = "我在，你说。"
# 剥掉唤醒词后短于这个长度，就当只是叫了声名字，不送模型
MIN_UTTERANCE_CHARS = 2


class Reporter(Protocol):
    """编排器向外吐事件的口子，由 CLI 实现。"""

    def state(self, state: DialogueState) -> None: ...
    def heard(self, text: str) -> None: ...
    def say(self, sentence: str) -> None: ...
    def note(self, text: str) -> None: ...
    def error(self, text: str) -> None: ...


class _SilentReporter:
    def state(self, state: DialogueState) -> None:
        pass

    def heard(self, text: str) -> None:
        pass

    def say(self, sentence: str) -> None:
        pass

    def note(self, text: str) -> None:
        pass

    def error(self, text: str) -> None:
        pass


@dataclass
class _BargeGate:
    """一次「疑似打断」：先静音收语音，做声纹判定，再决定真打断还是续播。"""

    frames: list[np.ndarray] = field(default_factory=list)
    decided: bool = False
    verdict: bool | None = None
    utterance: np.ndarray | None = None

    def feed(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def audio(self) -> np.ndarray:
        if not self.frames:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(self.frames)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        capture: AudioCapture,
        vad: VadSegmenter,
        asr: AsrClient,
        tutor: Tutor,
        speaker: SpeechOutput,
        reporter: Reporter | None = None,
        verifier: VoiceprintStore | None = None,
    ) -> None:
        self._settings = settings
        self._capture = capture
        self._vad = vad
        self._asr = asr
        self._tutor = tutor
        self._speaker = speaker
        self._reporter: Reporter = reporter or _SilentReporter()
        self._verifier = verifier
        self._wake = WakeWordMatcher(settings.wake.word, settings.wake.variants)

        self._state = DialogueState.IDLE
        self._frames: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=settings.audio.queue_frames
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._utterance_task: asyncio.Task | None = None
        self._response_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._barge_gate: _BargeGate | None = None
        self._pending: set[asyncio.Task] = set()

        # 判定窗口按"帧数"算，不是采样点数——VAD 是按帧喂进来的
        self._verify_frames = max(
            1,
            math.ceil(
                settings.interrupt.verify_ms / max(1, settings.audio.frame_ms)
            ),
        )

    @property
    def state(self) -> DialogueState:
        return self._state

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._capture.start()
        reader = threading.Thread(target=self._read_frames, name="audio-reader", daemon=True)
        reader.start()

        self._reporter.state(self._state)
        self._reporter.note(f"说「{self._settings.wake.word}」就能叫醒我。")

        try:
            await self._pump()
        finally:
            self._stopping.set()
            self._cancel_tasks()
            self._speaker.interrupt()
            self._capture.stop()
            reader.join(timeout=2)

    # ---------- 音频泵 ----------

    def _read_frames(self) -> None:
        while not self._stopping.is_set():
            frame = self._capture.frame(timeout=0.5)
            loop = self._loop
            if frame is None or loop is None:
                continue
            try:
                loop.call_soon_threadsafe(self._enqueue, frame)
            except RuntimeError:
                return  # 事件循环已关闭

    def _enqueue(self, frame: np.ndarray) -> None:
        try:
            self._frames.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # 处理不过来就丢帧，优先保证低延迟

    async def _pump(self) -> None:
        while True:
            frame = await self._frames.get()
            self._advance_barge_gate(frame)
            for event in self._vad.process(frame):
                self._on_vad_event(event)

    # ---------- VAD 事件 ----------

    def _on_vad_event(self, event: Event) -> None:
        if isinstance(event, SpeechStart):
            self._on_speech_start()
        elif isinstance(event, SpeechEnd):
            self._on_speech_end(event.audio)

    def _interrupt_enabled(self) -> bool:
        return self._settings.interrupt.enabled and self._settings.interrupt.mode != "off"

    def _voiceprint_gate_enabled(self) -> bool:
        return (
            self._settings.interrupt.mode == "voiceprint"
            and self._verifier is not None
            and self._verifier.enrolled
        )

    def _on_speech_start(self) -> None:
        if not self._interrupt_enabled():
            self._cancel_idle_timer()
            return

        if self._state not in (DialogueState.SPEAKING, DialogueState.THINKING):
            self._cancel_idle_timer()
            return

        if self._voiceprint_gate_enabled():
            self._open_barge_gate()
        else:
            self._barge_in()

    def _open_barge_gate(self) -> None:
        """先立刻静音，再收集语音做声纹判定——真打断的响应速度不受影响。"""
        self._barge_gate = _BargeGate()
        self._speaker.pause()

    def _advance_barge_gate(self, frame: np.ndarray) -> None:
        gate = self._barge_gate
        if gate is None or gate.decided:
            return
        gate.feed(frame)
        if len(gate.frames) >= self._verify_frames:
            gate.decided = True
            self._spawn(self._verify_gate(gate))

    async def _verify_gate(self, gate: _BargeGate) -> None:
        try:
            is_owner = await asyncio.to_thread(self._verifier.matches, gate.audio())
        except Exception as exc:
            self._reporter.error(f"声纹比对失败，按你本人处理：{exc}")
            is_owner = True  # 判不了就放行，宁可漏拦也别拦错自己人

        gate.verdict = is_owner
        if is_owner:
            self._reporter.note("确认是你，停下来听你说。")
            self._barge_in()
        else:
            self._reporter.note("听着不像你，继续讲。")
            self._speaker.resume()

        self._consume_gate(gate)

    def _consume_gate(self, gate: _BargeGate) -> None:
        """判定出结果后，决定那句话要不要送识别。"""
        if self._barge_gate is not gate or gate.utterance is None:
            return
        self._barge_gate = None
        if gate.verdict:
            self._start_utterance(gate.utterance)

    def _on_speech_end(self, audio: np.ndarray) -> None:
        gate = self._barge_gate
        if gate is None:
            self._start_utterance(audio)
            return

        gate.utterance = audio
        if gate.verdict is not None:
            # 判定已有结果：不是本人的话，这句话直接丢掉，不送识别
            self._barge_gate = None
            if gate.verdict:
                self._start_utterance(audio)
            return

        if not gate.decided:
            gate.decided = True
            self._spawn(self._verify_gate(gate))

    def _start_utterance(self, audio: np.ndarray) -> None:
        previous = self._utterance_task
        if previous is not None and not previous.done():
            previous.cancel()
        self._utterance_task = asyncio.create_task(self._handle_utterance(audio))

    def _barge_in(self) -> None:
        self._reporter.note("检测到插话，立即停播。")
        self._speaker.interrupt()
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
        self._set_state(DialogueState.LISTENING)

    def _spawn(self, coro) -> None:
        """起后台 task 并持有引用，避免被垃圾回收掉。"""
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    # ---------- 一句话的完整处理 ----------

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        try:
            text = await asyncio.to_thread(self._asr.transcribe, audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reporter.error(f"识别失败：{exc}")
            self._resume_listening()
            return

        text = text.strip()
        if not text:
            return
        self._reporter.heard(text)

        if self._state is DialogueState.IDLE and self._settings.wake.enabled:
            matched, remainder = self._wake.match(text)
            if not matched:
                return
            self._set_state(DialogueState.LISTENING)
            self._cancel_idle_timer()
            if len(remainder) < MIN_UTTERANCE_CHARS:
                # 只叫了名字（或只剩个语气词），别把它当问题塞给模型
                await self._speak_wake_reply()
                self._arm_idle_timer()
                return
            self._reporter.heard(remainder)
            text = remainder

        if self._state is DialogueState.IDLE:
            # 关掉唤醒词时，待唤醒态直接当聆听处理
            self._set_state(DialogueState.LISTENING)

        await self._respond(text)

    # ---------- 生成 + 播报 ----------

    async def _speak_wake_reply(self) -> None:
        """应答一句「我在」。

        走和正常播报一样的路径（含 SPEAKING 状态），这样用户立刻接话时
        它能被正常打断，而不是和用户的声音叠在一起。
        """
        self._speaker.begin()
        self._set_state(DialogueState.SPEAKING)
        try:
            await asyncio.to_thread(self._speaker.speak, WAKE_REPLY)
        finally:
            if not self._stopping.is_set() and not self._speaker.interrupted:
                self._set_state(DialogueState.LISTENING)

    async def _respond(self, user_text: str) -> None:
        self._response_task = asyncio.current_task()
        self._cancel_idle_timer()
        self._speaker.begin()
        try:
            self._set_state(DialogueState.THINKING)
            await self._stream_response(user_text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._reporter.error(f"讲解失败：{exc}")
        finally:
            if self._response_task is asyncio.current_task():
                self._response_task = None
            if not self._stopping.is_set() and not self._speaker.interrupted:
                self._set_state(DialogueState.LISTENING)
                self._arm_idle_timer()

    async def _stream_response(self, user_text: str) -> None:
        """LLM 边生成、TTS 边播报，两边通过句子队列解耦。"""
        pending: asyncio.Queue[str | None] = asyncio.Queue()

        async def produce() -> None:
            buffer = ""
            try:
                async for piece in self._tutor.stream_reply(user_text):
                    buffer += piece
                    ready, buffer = take_sentences(buffer)
                    for sentence in ready:
                        await pending.put(sentence)
                if buffer.strip():
                    await pending.put(buffer.strip())
            finally:
                pending.put_nowait(None)

        async def consume() -> None:
            while True:
                sentence = await pending.get()
                if sentence is None:
                    return
                if self._speaker.interrupted:
                    continue
                self._set_state(DialogueState.SPEAKING)
                self._reporter.say(sentence)
                await asyncio.to_thread(self._speaker.speak, sentence)

        await asyncio.gather(produce(), consume())

    # ---------- 状态与计时 ----------

    def _set_state(self, state: DialogueState) -> None:
        if state is self._state:
            return
        self._state = state
        self._reporter.state(state)

    def _resume_listening(self) -> None:
        if self._stopping.is_set():
            return
        self._set_state(DialogueState.LISTENING)
        self._arm_idle_timer()

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        timeout = self._settings.vad.idle_timeout_s
        if timeout <= 0:
            return
        self._idle_task = asyncio.create_task(self._idle_after(timeout))

    async def _idle_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        if self._stopping.is_set():
            return
        if self._state is DialogueState.LISTENING and not self._speaker.speaking:
            self._tutor.reset()
            self._set_state(DialogueState.IDLE)
            self._reporter.note(
                f"静默超时，回到待唤醒。再叫我一声「{self._settings.wake.word}」。"
            )

    def _cancel_idle_timer(self) -> None:
        task = self._idle_task
        if task is not None and not task.done():
            task.cancel()
        self._idle_task = None

    def _cancel_tasks(self) -> None:
        for task in (self._utterance_task, self._response_task, self._idle_task):
            if task is not None and not task.done():
                task.cancel()
        self._utterance_task = None
        self._response_task = None
        self._idle_task = None
        self._barge_gate = None
        for task in list(self._pending):
            task.cancel()
        self._pending.clear()
