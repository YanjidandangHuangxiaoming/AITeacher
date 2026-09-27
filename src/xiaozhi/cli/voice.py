"""P1 语音模式入口：蓝牙耳机 + 关键词唤醒 + ASR + TTS + 打断。"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..asr.cloud import ParaformerAsr
from ..audio.capture import AudioCapture
from ..audio.devices import (
    describe_device,
    effective_device,
    looks_like_headset,
    looks_like_loopback,
    refresh_devices,
)
from ..audio.player import StreamPlayer
from ..audio.speaker import SpeechOutput
from ..audio.vad import VadSegmenter
from ..audio.voiceprint import VoiceprintStore
from ..config import Settings, get_settings, load_prompt, missing_keys
from ..dialogue.orchestrator import Orchestrator
from ..dialogue.state import DialogueState
from ..llm.client import DeepSeekClient
from ..llm.tutor import Tutor
from ..tts.cloud import CosyVoiceTts

console = Console()

_STATE_STYLES = {
    DialogueState.IDLE: "dim",
    DialogueState.LISTENING: "cyan",
    DialogueState.THINKING: "yellow",
    DialogueState.SPEAKING: "green",
}


class RichReporter:
    def state(self, state: DialogueState) -> None:
        console.print(f"[{_STATE_STYLES[state]}]· {state.label}[/]")

    def heard(self, text: str) -> None:
        console.print(f"[bold cyan]你[/]：{escape(text)}")

    def say(self, sentence: str) -> None:
        console.print(f"[bold green]小智[/]：{escape(sentence)}")

    def note(self, text: str) -> None:
        console.print(f"[dim]{escape(text)}[/dim]")

    def error(self, text: str) -> None:
        console.print(f"[red]{escape(text)}[/red]")


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        console.print(f"[red]读取配置失败：{exc}[/red]")
        raise SystemExit(1) from exc

    # 必须在打开任何音频流之前重扫设备：PortAudio 只在初始化时枚举一次，
    # 后连上的蓝牙耳机不重扫就不会出现在设备列表里。
    refresh_devices()

    verifier = _load_verifier(settings)

    # 先报设备情况，这样即使还没配好 Key，也能马上确认音频设备选对了
    console.print(_banner(settings, verifier))
    _warn_about_devices(settings)

    missing = missing_keys(settings)
    if missing:
        console.print("[red]缺少 API Key，无法启动。[/red]")
        console.print(f"请在项目根目录的 .env 里填入：{', '.join(missing)}")
        raise SystemExit(1)

    try:
        speaker, orchestrator = _build(settings, verifier)
    except Exception as exc:
        console.print(f"[red]音频设备初始化失败：{exc}[/red]")
        raise SystemExit(1) from exc

    speaker.open()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n再见，同学。")
    finally:
        speaker.close()


def _load_verifier(settings: Settings) -> VoiceprintStore | None:
    """载入声纹。没注册过就退回纯 VAD 打断，并提示用户去注册。"""
    if not settings.voiceprint.enabled or settings.interrupt.mode != "voiceprint":
        return None

    store = VoiceprintStore(settings.voiceprint, settings.voiceprint_path())
    if not store.load():
        console.print(
            "[yellow]还没注册声纹，打断会退化成「检测到人声就打断」，"
            "旁人说话也会打断我。先跑 [bold]python -m xiaozhi enroll[/bold] "
            "注册你的声纹。[/yellow]"
        )
        return None

    # 模型加载要好几秒，必须在这里做掉，否则第一次打断会卡住
    with console.status("[cyan]正在加载声纹模型…[/cyan]"):
        store.warmup()
    return store


def _build(settings: Settings, verifier: VoiceprintStore | None) -> tuple[SpeechOutput, Orchestrator]:
    sample_rate = settings.audio.sample_rate

    capture = AudioCapture(settings.audio)
    vad = VadSegmenter(settings.vad, sample_rate, settings.audio.frame_ms)
    asr = ParaformerAsr(settings.asr, sample_rate)
    speaker = SpeechOutput(
        CosyVoiceTts(settings.tts, sample_rate),
        StreamPlayer(sample_rate, settings.audio.output_device),
    )
    tutor = Tutor(
        settings.tutor,
        load_prompt("tutor_system.md"),
        DeepSeekClient(settings.llm),
    )
    orchestrator = Orchestrator(
        settings,
        capture,
        vad,
        asr,
        tutor,
        speaker,
        reporter=RichReporter(),
        verifier=verifier,
    )
    return speaker, orchestrator


def _interrupt_label(settings: Settings, verifier: VoiceprintStore | None) -> str:
    if not settings.interrupt.enabled or settings.interrupt.mode == "off":
        return "已关闭"
    if settings.interrupt.mode == "vad":
        return "检测到人声即打断"
    if verifier is not None and verifier.enrolled:
        return f"声纹确认（阈值 {verifier.threshold:.2f}）"
    return "检测到人声即打断（未注册声纹）"


def _banner(settings: Settings, verifier: VoiceprintStore | None) -> Panel:
    try:
        input_name = describe_device(
            "input", effective_device("input", settings.audio.input_device)
        )
        output_name = describe_device(
            "output", effective_device("output", settings.audio.output_device)
        )
    except Exception as exc:
        input_name = output_name = f"解析失败（{exc}）"

    return Panel.fit(
        "[bold]小智 AI 家教[/bold] · P1 语音版\n"
        f"唤醒词 {settings.wake.word}　打断 {_interrupt_label(settings, verifier)}\n"
        f"麦克风 {input_name}\n"
        f"扬声器 {output_name}\n"
        f"识别 {settings.asr.model}　合成 {settings.tts.model} / {settings.tts.voice}\n\n"
        "按 Ctrl+C 退出",
        border_style="green",
    )


def _warn_about_devices(settings: Settings) -> None:
    """把两类"会静默失效"的设备误配当面拦下来。"""
    try:
        input_name = describe_device(
            "input", effective_device("input", settings.audio.input_device)
        )
        output_name = describe_device(
            "output", effective_device("output", settings.audio.output_device)
        )
    except Exception:
        return

    if looks_like_loopback(input_name):
        console.print(
            f"[yellow]提醒：当前麦克风是「{input_name}」，这是录电脑自身声音的环回设备，"
            "听不到你说话。请把 config/default.yaml 的 audio.input_device "
            "改成真实麦克风或耳机麦克风。[/yellow]"
        )
    if settings.interrupt.require_headset and not looks_like_headset(output_name):
        console.print(
            f"[yellow]提醒：当前扬声器是「{output_name}」，不像耳机。"
            "外放时我的声音会被麦克风拾到，可能自己打断自己。"
            "建议戴上蓝牙耳机，或把 audio.output_device 指向耳机。[/yellow]"
        )
