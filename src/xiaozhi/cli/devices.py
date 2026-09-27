"""列出音频设备，并可选地逐个试开，确认哪些真的能用。

每次运行都会让 PortAudio 重新枚举，所以刚连上的蓝牙耳机也会出现在列表里。

    python -m xiaozhi devices
    python -m xiaozhi devices --probe
"""

from __future__ import annotations

from rich.console import Console

from ..audio.devices import list_devices, probe_report, refresh_devices

console = Console()


def main(argv: list[str] | None = None) -> None:
    args = argv or []

    # PortAudio 只在初始化时枚举一次设备，刚连上的蓝牙耳机不重扫就看不到
    refresh_devices()

    console.print("[bold]可用音频设备[/bold]")
    console.print(list_devices())

    if "--probe" in args:
        console.print("\n[bold]逐个试开（约 10 秒，会短暂占用声卡）[/bold]")
        console.print(probe_report(refresh=True))
    else:
        console.print("\n加 [bold]--probe[/bold] 可以逐个试开，确认哪些真的能打开。")

    console.print(
        "\n把设备名的一部分填到 config/default.yaml 的 "
        "audio.input_device / audio.output_device 即可（填 null 表示系统默认）。"
    )
