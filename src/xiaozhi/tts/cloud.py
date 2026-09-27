"""阿里 CosyVoice 流式合成。"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from contextlib import suppress

import dashscope
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

from ..config import TtsConfig
from .base import StopCheck, TtsClient, TtsError

_AUDIO_FORMATS = {
    8000: AudioFormat.PCM_8000HZ_MONO_16BIT,
    16000: AudioFormat.PCM_16000HZ_MONO_16BIT,
    22050: AudioFormat.PCM_22050HZ_MONO_16BIT,
    24000: AudioFormat.PCM_24000HZ_MONO_16BIT,
    44100: AudioFormat.PCM_44100HZ_MONO_16BIT,
    48000: AudioFormat.PCM_48000HZ_MONO_16BIT,
}


class CosyVoiceTts(TtsClient):
    def __init__(self, cfg: TtsConfig, sample_rate: int = 16000) -> None:
        self._cfg = cfg
        self._sample_rate = sample_rate
        if cfg.api_key:
            dashscope.api_key = cfg.api_key

    def stream(self, text: str, should_stop: StopCheck) -> Iterator[bytes]:
        if not text.strip():
            return

        chunks: queue.Queue[bytes | None] = queue.Queue()
        failures: list[str] = []

        class _Callback(ResultCallback):
            def on_data(self, data: bytes) -> None:
                chunks.put(data)

            def on_complete(self) -> None:
                chunks.put(None)

            def on_close(self) -> None:
                chunks.put(None)

            def on_error(self, message) -> None:
                failures.append(str(message))
                chunks.put(None)

            def on_open(self) -> None:
                pass

            def on_event(self, message) -> None:
                pass

        synthesizer = SpeechSynthesizer(
            model=self._cfg.model,
            voice=self._cfg.voice,
            format=self._audio_format(),
            speech_rate=self._cfg.speed,
            callback=_Callback(),
        )
        synthesizer.streaming_call(text)

        # streaming_complete() 会一直阻塞到所有音频下发完毕，必须扔到独立线程，
        # 否则消费音频的这条线程会被卡死，流式就退化成同步了。
        threading.Thread(
            target=_finish_quietly,
            args=(synthesizer,),
            name="tts-complete",
            daemon=True,
        ).start()

        interrupted = False
        try:
            while True:
                if should_stop():
                    interrupted = True
                    break
                try:
                    item = chunks.get(timeout=self._cfg.timeout_s)
                except queue.Empty as exc:
                    raise TtsError("语音合成超时，没有收到音频数据") from exc
                if item is None:
                    break
                yield item
        finally:
            _release(synthesizer, interrupted)

        if failures:
            raise TtsError(failures[0])

    def _audio_format(self) -> AudioFormat:
        audio_format = _AUDIO_FORMATS.get(self._sample_rate)
        if audio_format is None:
            raise TtsError(
                f"不支持的合成采样率 {self._sample_rate}，可选：{sorted(_AUDIO_FORMATS)}"
            )
        return audio_format


def _finish_quietly(synthesizer: SpeechSynthesizer) -> None:
    with suppress(Exception):
        synthesizer.streaming_complete()


def _release(synthesizer: SpeechSynthesizer, interrupted: bool) -> None:
    with suppress(Exception):
        if interrupted:
            synthesizer.streaming_cancel()
        else:
            synthesizer.close()
