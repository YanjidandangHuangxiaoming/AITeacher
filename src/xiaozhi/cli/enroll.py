"""录制并注册声纹。

    python -m xiaozhi enroll

注册后，语音模式会在打断前先做一次声纹比对，把旁人的说话声挡掉。
"""

from __future__ import annotations

import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from ..audio.capture import AudioCapture
from ..audio.devices import describe_device, effective_device, refresh_devices
from ..audio.voiceprint import Enrollment, VoiceprintError, VoiceprintStore
from ..config import Settings, get_settings

console = Console()

SCRIPT_HINT = (
    "随便说点什么，或者照着念下面这段。重要的是用你平时跟我说话的音量和语速：\n"
    "[dim]「今天数学课讲的是一元二次方程，我觉得配方法比公式法好记一点，"
    "不过系数比较大的时候还是公式法更稳妥。晚上我打算先把作业写完，"
    "再把上周做错的那几道几何题重做一遍。」[/dim]"
)


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        console.print(f"[red]读取配置失败：{exc}[/red]")
        raise SystemExit(1) from exc

    refresh_devices()
    console.print(
        Panel.fit(
            "[bold]声纹注册[/bold]\n"
            f"接下来录音 {settings.voiceprint.enroll_seconds:.0f} 秒，"
            "请在安静环境里用平时的音量和语速说话\n"
            f"录音设备：{_input_name(settings)}",
            border_style="green",
        )
    )
    console.print(SCRIPT_HINT)

    try:
        console.input("\n准备好了按回车开始录音…")
    except (EOFError, KeyboardInterrupt):
        return

    try:
        audio = _record(settings)
    except Exception as exc:
        console.print(f"[red]录音失败：{exc}[/red]")
        raise SystemExit(1) from exc

    store = VoiceprintStore(settings.voiceprint, settings.voiceprint_path())
    try:
        with console.status("[cyan]正在分析声纹…[/cyan]"):
            enrollment = store.enroll(audio)
            store.save()
    except VoiceprintError as exc:
        console.print(f"[red]注册失败：{exc}[/red]")
        raise SystemExit(1) from exc

    _report(enrollment, store)


def _record(settings: Settings) -> np.ndarray:
    capture = AudioCapture(settings.audio)
    frame_ms = settings.audio.frame_ms
    target = int(settings.voiceprint.enroll_seconds * 1000 / frame_ms)
    # 给足 1.5 倍墙钟时间，容忍偶发丢帧
    deadline = time.monotonic() + settings.voiceprint.enroll_seconds * 1.5
    frames: list[np.ndarray] = []

    capture.start()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("录音中…", total=target)
            while len(frames) < target and time.monotonic() < deadline:
                frame = capture.frame(timeout=0.2)
                if frame is None:
                    continue
                frames.append(frame)
                progress.update(task, completed=len(frames))
    finally:
        capture.stop()

    if not frames:
        raise RuntimeError("一帧音频都没收到，检查麦克风是否被别的程序占用")
    return np.concatenate(frames)


def _input_name(settings: Settings) -> str:
    try:
        index = effective_device("input", settings.audio.input_device)
        return describe_device("input", index)
    except Exception as exc:
        return f"解析失败（{exc}）"


def _report(enrollment: Enrollment, store: VoiceprintStore) -> None:
    if enrollment.self_similarity >= 0.90:
        quality = "[green]良好[/green]"
    elif enrollment.self_similarity >= 0.80:
        quality = "[yellow]一般[/yellow]"
    else:
        quality = "[red]偏低[/red]"

    console.print(
        Panel(
            "\n".join(
                [
                    f"录音时长　{enrollment.seconds:.1f} 秒",
                    f"自相似度　{enrollment.self_similarity:.3f}　{quality}",
                    f"判定阈值　{enrollment.threshold:.2f}",
                    f"保存位置　{store.vector_path}",
                ]
            ),
            title="注册完成",
            border_style="green",
        )
    )

    if enrollment.self_similarity < 0.80:
        console.print(
            "[yellow]自相似度偏低，说明录音里混了噪声，或者你说话方式变化太大。"
            "建议换个安静环境重跑一次 enroll，否则打断时容易认不出你。[/yellow]"
        )

    console.print(
        "\n现在可以跑 [bold]python -m xiaozhi voice[/bold] 了。"
        "觉得太严或太松，就改 config/default.yaml 里的 "
        f"voiceprint.threshold（留 null 表示用刚标定的 {enrollment.threshold:.2f}）。"
    )
