"""播报出口的暂停续播与真打断测试。"""

from __future__ import annotations

import threading
import time

from xiaozhi.audio.speaker import SpeechOutput


class FakePlayer:
    def __init__(self, interrupt_after: int | None = None) -> None:
        self._interrupted = threading.Event()
        self.interrupt_after = interrupt_after
        self.written: list[bytes] = []
        self.silence_calls = 0

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def reset(self) -> None:
        self._interrupted.clear()
        self.written.clear()

    def interrupt(self) -> None:
        self._interrupted.set()

    def silence(self) -> None:
        self.silence_calls += 1

    def write(self, pcm: bytes) -> bool:
        if self._interrupted.is_set():
            return False
        self.written.append(pcm)
        if self.interrupt_after is not None and len(self.written) >= self.interrupt_after:
            self.interrupt()
        return not self._interrupted.is_set()


class FakeTts:
    def __init__(self, chunks: int = 5, delay: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay

    def stream(self, text, should_stop):
        for index in range(self._chunks):
            if should_stop():
                return
            if self._delay:
                time.sleep(self._delay)
            if should_stop():
                return
            yield f"chunk-{index}".encode()


def _speak_in_thread(speaker: SpeechOutput, text: str) -> tuple[threading.Thread, list[bool], threading.Event]:
    result: list[bool] = []
    done = threading.Event()

    def run() -> None:
        result.append(speaker.speak(text))
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result, done


# ---------- 正常播报 ----------


def test_full_playback_completes() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=4), player)
    assert speaker.speak("你好") is True
    assert len(player.written) == 4
    assert speaker.speaking is False


def test_blank_text_is_noop() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(), player)
    assert speaker.speak("   ") is True
    assert player.written == []


# ---------- 真打断 ----------


def test_interrupt_stops_playback_midway() -> None:
    player = FakePlayer(interrupt_after=2)
    speaker = SpeechOutput(FakeTts(chunks=10), player)
    assert speaker.speak("很长的一段话") is False
    assert len(player.written) == 2
    assert speaker.interrupted is True


def test_speak_reports_false_when_already_interrupted() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=3), player)
    speaker.interrupt()
    assert speaker.speak("你好") is False
    assert player.written == []


def test_begin_clears_previous_state() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=2), player)
    speaker.interrupt()
    assert speaker.interrupted is True
    speaker.begin()
    assert speaker.interrupted is False
    assert speaker.speak("你好") is True


# ---------- 暂停续播 ----------


def test_pause_holds_writing_until_resume() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=5), player)
    speaker.pause()
    assert player.silence_calls == 1

    _, result, done = _speak_in_thread(speaker, "你好")
    time.sleep(0.2)
    assert player.written == [], "暂停期间不应该写入音频"
    assert done.is_set() is False

    speaker.resume()
    assert done.wait(2.0) is True
    assert result == [True]
    assert len(player.written) == 5
    assert speaker.paused is False


def test_speak_after_pause_waits_at_first_chunk() -> None:
    """在思考阶段被暂停时，第一句话也不应该抢着播。"""
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=2), player)
    speaker.pause()

    _, result, done = _speak_in_thread(speaker, "先看题干。")
    time.sleep(0.15)
    assert player.written == []

    speaker.resume()
    assert done.wait(2.0) is True
    assert result == [True]


def test_interrupt_wins_over_paused() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=50), player)
    speaker.pause()

    _, result, done = _speak_in_thread(speaker, "一段很长的话")
    time.sleep(0.15)
    speaker.interrupt()

    assert done.wait(2.0) is True
    assert result == [False]
    assert speaker.interrupted is True
    assert player.written == []


def test_pause_times_out_and_resumes(monkeypatch) -> None:
    """兜底：判定流程万一卡住，暂停也不能让播报永远停在那儿。"""
    monkeypatch.setattr("xiaozhi.audio.speaker.MAX_PAUSE_S", 0.3)
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=3), player)
    speaker.pause()

    assert speaker.speak("你好") is True
    assert len(player.written) == 3


def test_resume_without_pause_is_safe() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=2), player)
    speaker.resume()
    assert speaker.speak("你好") is True
    assert len(player.written) == 2


# ---------- 代号作废（防"两个声音"） ----------


def test_begin_invalidates_inflight_speak() -> None:
    """回归：旧播报线程必须被作废。

    speak() 跑在 to_thread 里，取消 asyncio task 并不会让线程停下。
    若新一轮回复只是清标志位，旧线程会继续往同一输出流写，
    两段音频交织——听起来就是"两个声音在同时说话"。
    """
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=50, delay=0.02), player)

    _, result, done = _speak_in_thread(speaker, "第一段")
    time.sleep(0.12)
    assert player.written, "第一段应该已经开始播了"

    speaker.begin()  # 新一轮播报开始，旧播报作废
    assert done.wait(2.0) is True
    assert result == [False], "旧播报必须被判定为已作废"

    settled = len(player.written)
    time.sleep(0.2)
    assert len(player.written) == settled, "作废之后不能再往输出流里写"


def test_second_response_does_not_overlap_first() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=40, delay=0.01), player)

    _, first_result, first_done = _speak_in_thread(speaker, "第一段")
    time.sleep(0.08)

    speaker.begin()
    _, second_result, second_done = _speak_in_thread(speaker, "第二段")

    assert first_done.wait(2.0) is True
    assert second_done.wait(2.0) is True
    assert first_result == [False], "第一段应被作废"
    assert second_result == [True], "第二段应完整播完"

    settled = len(player.written)
    time.sleep(0.2)
    assert len(player.written) == settled, "两段播完后不应再有写入"


def test_interrupt_still_wins_after_begin() -> None:
    player = FakePlayer()
    speaker = SpeechOutput(FakeTts(chunks=40, delay=0.01), player)
    speaker.begin()

    _, result, done = _speak_in_thread(speaker, "一段话")
    time.sleep(0.08)
    speaker.interrupt()

    assert done.wait(2.0) is True
    assert result == [False]
