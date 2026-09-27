"""对话编排器的状态流转、打断与声纹拦截测试。

全部用假组件，不碰真实音频设备、不联网，所以可以在 CI 里跑。
测试直接驱动 _frames + _pump，这是唯一能绕开硬件又走完整状态机的方式。
"""

from __future__ import annotations

import asyncio

import numpy as np

from xiaozhi.audio.vad import SpeechEnd, SpeechStart
from xiaozhi.config import Settings
from xiaozhi.dialogue.orchestrator import Orchestrator
from xiaozhi.dialogue.state import DialogueState

FRAME = np.zeros(480, dtype=np.int16)
AUDIO = np.ones(16000, dtype=np.int16)


class FakeCapture:
    """测试不走 run()，这里只是占位，避免碰真实设备。"""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def frame(self, timeout: float = 0.5):
        return None


class ScriptedVad:
    """按脚本逐帧返回预设事件。"""

    def __init__(self) -> None:
        self.script: list[list[object]] = []

    def process(self, frame):  # noqa: ANN001
        return self.script.pop(0) if self.script else []


class FakeAsr:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio):  # noqa: ANN001
        self.calls += 1
        return self.text


class FakeTutor:
    def __init__(self, sentences: list[str], delay: float = 0.01) -> None:
        self._sentences = sentences
        self._delay = delay
        self.replies: list[str] = []
        self.reset_calls = 0

    async def stream_reply(self, user_text: str):
        self.replies.append(user_text)
        for sentence in self._sentences:
            yield sentence
            await asyncio.sleep(self._delay)

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSpeaker:
    def __init__(self) -> None:
        self._interrupted = False
        self._paused = False
        self.spoken: list[str] = []
        self.interrupt_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speaking(self) -> bool:
        return False

    def begin(self) -> None:
        self._interrupted = False
        self._paused = False

    def speak(self, text: str) -> bool:
        if self._interrupted:
            return False
        self.spoken.append(text)
        return True

    def interrupt(self) -> None:
        self._interrupted = True
        self.interrupt_calls += 1

    def pause(self) -> None:
        self._paused = True
        self.pause_calls += 1

    def resume(self) -> None:
        self._paused = False
        self.resume_calls += 1


class FakeVerifier:
    """固定返回预设判定结果，并记录被比对的音频长度。"""

    def __init__(self, verdict: bool) -> None:
        self.enrolled = True
        self._verdict = verdict
        self.lengths: list[int] = []

    def matches(self, audio) -> bool:  # noqa: ANN001
        self.lengths.append(int(audio.size))
        return self._verdict


def _build(
    asr_text: str,
    sentences: list[str],
    delay: float = 0.01,
    verifier: FakeVerifier | None = None,
    interrupt: dict | None = None,
):
    vad = ScriptedVad()
    asr = FakeAsr(asr_text)
    speaker = FakeSpeaker()
    tutor = FakeTutor(sentences, delay)
    settings = Settings(
        audio={"queue_frames": 64},
        wake={"enabled": True, "word": "小智小智"},
        vad={"idle_timeout_s": 0},
        interrupt=interrupt or {"mode": "vad", "verify_ms": 60},
    )
    orchestrator = Orchestrator(
        settings, FakeCapture(), vad, asr, tutor, speaker, verifier=verifier
    )
    return orchestrator, vad, asr, speaker, tutor


async def _wait_for(predicate, timeout: float = 3.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def _emit(orchestrator: Orchestrator, vad: ScriptedVad, events: list[object]) -> None:
    vad.script.append(events)
    orchestrator._frames.put_nowait(FRAME)


async def _feed_frames(orchestrator: Orchestrator, count: int = 10) -> None:
    """喂若干帧，只为把声纹判定窗口撑满。"""
    for _ in range(count):
        orchestrator._frames.put_nowait(FRAME)
        await asyncio.sleep(0.01)


# ---------- 唤醒与回复 ----------


def test_wake_word_then_reply_is_spoken() -> None:
    asyncio.run(_scenario_wake_then_reply())


async def _scenario_wake_then_reply() -> None:
    orchestrator, vad, _, speaker, tutor = _build(
        "小智小智，这道题怎么做", ["先看题干。", "它问的是什么？"]
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(
            lambda: len(speaker.spoken) == 2
            and orchestrator.state is DialogueState.LISTENING
        )
    finally:
        pump.cancel()

    assert speaker.spoken == ["先看题干。", "它问的是什么？"]
    # 唤醒词必须被剥掉，不能原样塞给模型
    assert tutor.replies == ["这道题怎么做"]


def test_speech_without_wake_word_is_ignored() -> None:
    asyncio.run(_scenario_ignored())


async def _scenario_ignored() -> None:
    orchestrator, vad, asr, speaker, tutor = _build("这道题怎么做", ["不该被播出来。"])
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: asr.calls == 1)
        await asyncio.sleep(0.2)
    finally:
        pump.cancel()

    assert speaker.spoken == []
    assert tutor.replies == []
    assert orchestrator.state is DialogueState.IDLE


def test_wake_word_alone_gets_short_ack() -> None:
    asyncio.run(_scenario_ack_only())


async def _scenario_ack_only() -> None:
    orchestrator, vad, _, speaker, tutor = _build("小智小智", ["不该走模型。"])
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: speaker.spoken)
    finally:
        pump.cancel()

    assert speaker.spoken == ["我在，你说。"]
    assert tutor.replies == []


def test_short_remainder_gets_ack_not_model() -> None:
    """叫了名字但没实际内容（"小智小智这" 剥出"这"）时，别把语气词塞给模型。"""
    asyncio.run(_scenario_short_remainder())


async def _scenario_short_remainder() -> None:
    orchestrator, vad, _, speaker, tutor = _build("小智小智这", ["不该走模型。"])
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: speaker.spoken)
    finally:
        pump.cancel()

    assert speaker.spoken == ["我在，你说。"]
    assert tutor.replies == []


def test_homophone_wake_word_is_accepted() -> None:
    """实测 ASR 会把"小智"写成"小志"，必须也能唤醒。"""
    asyncio.run(_scenario_homophone())


async def _scenario_homophone() -> None:
    orchestrator, vad, _, speaker, tutor = _build("小志小志，这道题怎么做", ["先看题干。"])
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: speaker.spoken)
    finally:
        pump.cancel()

    assert speaker.spoken == ["先看题干。"]


def test_wake_disabled_accepts_any_speech() -> None:
    asyncio.run(_scenario_wake_disabled())


async def _scenario_wake_disabled() -> None:
    vad = ScriptedVad()
    asr = FakeAsr("这道题怎么做")
    speaker = FakeSpeaker()
    tutor = FakeTutor(["先看题干。"])
    orchestrator = Orchestrator(
        Settings(
            audio={"queue_frames": 64},
            wake={"enabled": False, "word": "小智小智"},
            vad={"idle_timeout_s": 0},
        ),
        FakeCapture(),
        vad,
        asr,
        tutor,
        speaker,
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: speaker.spoken)
    finally:
        pump.cancel()

    assert speaker.spoken == ["先看题干。"]
    assert tutor.replies == ["这道题怎么做"]


# ---------- 纯 VAD 打断 ----------


def test_vad_barge_in_interrupts_playback() -> None:
    asyncio.run(_scenario_vad_barge_in())


async def _scenario_vad_barge_in() -> None:
    sentences = [f"第{index}句。" for index in range(1, 60)]
    orchestrator, vad, _, speaker, tutor = _build(
        "小智小智，讲讲这道题", sentences, delay=0.01
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: orchestrator.state is DialogueState.SPEAKING)
        assert await _wait_for(lambda: len(speaker.spoken) >= 1)

        spoken_before = len(speaker.spoken)
        _emit(orchestrator, vad, [SpeechStart()])

        assert await _wait_for(lambda: speaker.interrupt_calls >= 1)
        assert await _wait_for(lambda: orchestrator.state is DialogueState.LISTENING)
        await asyncio.sleep(0.15)
    finally:
        pump.cancel()

    assert len(speaker.spoken) <= spoken_before + 1
    assert tutor.replies == ["讲讲这道题"]


# ---------- 声纹打断 ----------


def test_voiceprint_gate_ignores_other_speaker() -> None:
    asyncio.run(_scenario_other_speaker())


async def _scenario_other_speaker() -> None:
    verifier = FakeVerifier(verdict=False)
    sentences = [f"第{index}句。" for index in range(1, 80)]
    orchestrator, vad, asr, speaker, tutor = _build(
        "小智小智，讲讲这道题",
        sentences,
        delay=0.008,
        verifier=verifier,
        interrupt={"mode": "voiceprint", "verify_ms": 60},
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: orchestrator.state is DialogueState.SPEAKING)
        asr_before = asr.calls

        _emit(orchestrator, vad, [SpeechStart()])
        assert await _wait_for(lambda: speaker.pause_calls >= 1)
        assert speaker.interrupt_calls == 0, "声纹模式下不应立即打断"

        await _feed_frames(orchestrator)
        assert await _wait_for(lambda: speaker.resume_calls >= 1)
        assert verifier.lengths, "应该真的做了声纹比对"

        # 旁人的语音说完，也不能被送去识别
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        await asyncio.sleep(0.2)

        assert speaker.interrupt_calls == 0
        assert asr.calls == asr_before
    finally:
        pump.cancel()

    assert tutor.replies == ["讲讲这道题"]


def test_voiceprint_gate_interrupts_for_owner() -> None:
    asyncio.run(_scenario_owner())


async def _scenario_owner() -> None:
    verifier = FakeVerifier(verdict=True)
    sentences = [f"第{index}句。" for index in range(1, 80)]
    orchestrator, vad, asr, speaker, tutor = _build(
        "小智小智，讲讲这道题",
        sentences,
        delay=0.008,
        verifier=verifier,
        interrupt={"mode": "voiceprint", "verify_ms": 60},
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: orchestrator.state is DialogueState.SPEAKING)
        asr_before = asr.calls

        _emit(orchestrator, vad, [SpeechStart()])
        await _feed_frames(orchestrator)

        assert await _wait_for(lambda: speaker.interrupt_calls >= 1)
        assert await _wait_for(lambda: orchestrator.state is DialogueState.LISTENING)
        assert speaker.resume_calls == 0, "确认是本人就不该续播"

        # 本人这句话说完，要正常送识别
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: asr.calls > asr_before)
    finally:
        pump.cancel()

    # 第二次是打断后那句：已在会话态，不再剥唤醒词，所以原样进模型
    assert tutor.replies == ["讲讲这道题", "小智小智，讲讲这道题"]


def test_voiceprint_mode_falls_back_without_enrollment() -> None:
    """配置写了 voiceprint 但没注册声纹时，要退化成纯 VAD 打断而不是完全失效。"""
    asyncio.run(_scenario_no_enrollment())


async def _scenario_no_enrollment() -> None:
    sentences = [f"第{index}句。" for index in range(1, 60)]
    orchestrator, vad, _, speaker, _ = _build(
        "小智小智，讲讲这道题",
        sentences,
        delay=0.01,
        verifier=None,
        interrupt={"mode": "voiceprint", "verify_ms": 60},
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: orchestrator.state is DialogueState.SPEAKING)
        _emit(orchestrator, vad, [SpeechStart()])
        assert await _wait_for(lambda: speaker.interrupt_calls >= 1)
    finally:
        pump.cancel()


def test_interrupt_off_never_interrupts() -> None:
    asyncio.run(_scenario_interrupt_off())


async def _scenario_interrupt_off() -> None:
    sentences = [f"第{index}句。" for index in range(1, 60)]
    orchestrator, vad, _, speaker, _ = _build(
        "小智小智，讲讲这道题",
        sentences,
        delay=0.01,
        interrupt={"mode": "off", "verify_ms": 60},
    )
    pump = asyncio.create_task(orchestrator._pump())
    try:
        _emit(orchestrator, vad, [SpeechEnd(AUDIO)])
        assert await _wait_for(lambda: orchestrator.state is DialogueState.SPEAKING)
        _emit(orchestrator, vad, [SpeechStart()])
        await asyncio.sleep(0.2)
    finally:
        pump.cancel()

    assert speaker.interrupt_calls == 0
    assert speaker.pause_calls == 0
