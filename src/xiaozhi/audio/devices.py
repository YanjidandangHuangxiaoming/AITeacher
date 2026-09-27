"""音频设备查询、解析与可用性探测。"""

from __future__ import annotations

import time

import sounddevice as sd


class AudioDeviceError(RuntimeError):
    pass


# WDM-KS 是内核级接口，PortAudio 下经常以 "Invalid device" 打开失败，只作兜底
LOW_LEVEL_HOST_APIS = ("WDM-KS",)


def refresh_devices() -> None:
    """强制 PortAudio 重新枚举设备。

    PortAudio 只在初始化时抓一次设备列表，此后才连上的蓝牙耳机不会出现在
    sd.query_devices() 里——这正是"耳机已连上、但程序列表里没有"的原因。
    terminate + initialize 会重新枚举（sounddevice 没有公开的重扫接口，
    这是它的既定用法）。只能在没有任何流打开时调用。
    """
    sd._terminate()
    sd._initialize()


# host API 优先级：WASAPI 延迟最低，其次 DirectSound / MME。
# WDM-KS 是内核级接口，PortAudio 下几乎必然以 Invalid device 失败，只作兜底。
HOST_API_PRIORITY = (
    "Windows WASAPI",
    "Windows DirectSound",
    "MME",
    "Windows WDM-KS",
)


def _host_api_names() -> list[str]:
    return [api["name"] for api in sd.query_hostapis()]


def candidate_devices(name: str | None, kind: str) -> list[int | None]:
    """按 host API 优先级列出候选设备索引。name 为空时返回 [None]（系统默认）。

    同一个物理设备会在多个 host API 下各出现一次，能打开与否差别很大
    （比如蓝牙耳机的 WDM-KS 条目必然失败、WASAPI 条目正常），
    所以这里返回有序候选，由调用方逐个试开，而不是赌第一个能成。
    """
    if not name:
        return [None]

    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    lowered = name.lower()
    devices = sd.query_devices()
    apis = _host_api_names()

    def rank(index: int) -> tuple[int, int]:
        api = apis[devices[index]["hostapi"]]
        priority = (
            HOST_API_PRIORITY.index(api)
            if api in HOST_API_PRIORITY
            else len(HOST_API_PRIORITY)
        )
        return (priority, index)

    matches = sorted(
        (
            index
            for index, device in enumerate(devices)
            if device[channel_key] > 0 and lowered in device["name"].lower()
        ),
        key=rank,
    )
    if not matches:
        direction = "输入" if kind == "input" else "输出"
        raise AudioDeviceError(
            f"找不到{direction}设备「{name}」。可用设备：\n{list_devices()}"
        )
    return list(matches)


def resolve_device(name: str | None, kind: str) -> int | None:
    """返回首选设备索引，主要用于启动横幅和提示。"""
    return candidate_devices(name, kind)[0]


def effective_device(kind: str, name: str | None) -> int | None:
    """返回实际生效的设备索引：配置了就用配置的，否则回落到系统默认。"""
    resolved = resolve_device(name, kind)
    if resolved is not None:
        return resolved

    index = sd.default.device[0] if kind == "input" else sd.default.device[1]
    if index is None or index < 0:
        return None
    return int(index)


def list_devices(refresh: bool = False) -> str:
    """格式化输出所有音频设备，含 host API 与默认采样率，供复制设备名到配置。

    refresh=True 时先重新枚举，这样刚连上的蓝牙耳机也能立刻出现在列表里。
    """
    if refresh:
        refresh_devices()
    devices = sd.query_devices()
    apis = _host_api_names()
    default_in, default_out = sd.default.device[0], sd.default.device[1]

    lines = []
    for index, device in enumerate(devices):
        api = apis[device["hostapi"]]
        marks = []
        if index == default_in:
            marks.append("默认输入")
        if index == default_out:
            marks.append("默认输出")
        if api in LOW_LEVEL_HOST_APIS:
            marks.append("低层接口，可能打不开")
        suffix = f"　[{'，'.join(marks)}]" if marks else ""
        name = " ".join(device["name"].split())
        lines.append(
            f"  {index:>3}  {api:<22}"
            f"输入 {device['max_input_channels']} / 输出 {device['max_output_channels']}"
            f"　{device['default_samplerate']:>6.0f}Hz  {name[:64]}{suffix}"
        )
    return "\n".join(lines)


HEADPHONE_HINTS = ("headphone", "headset", "耳机", "蓝牙", "bluetooth", "airpods", "hands-free")
# 这些是"录电脑自己声音"的环回设备，当麦克风用会完全听不到用户说话
LOOPBACK_HINTS = (
    "立体声混音",
    "stereo mix",
    "what u hear",
    "loopback",
    "主声音捕获",
    "monitor",
)


def looks_like_headset(name: str | None) -> bool:
    """粗略判断设备名是否像耳机，用于提示回声自激风险。"""
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in HEADPHONE_HINTS)


def looks_like_loopback(name: str | None) -> bool:
    """判断输入设备是不是环回设备（会把电脑自身的声音当成"人声"）。"""
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in LOOPBACK_HINTS)


def describe_device(kind: str, index: int | None) -> str:
    """把设备索引转成人能看的名字。"""
    if index is None:
        return "系统默认"
    try:
        return sd.query_devices(index)["name"]
    except Exception:
        return f"设备 {index}"


def probe_device(index: int, kind: str) -> str | None:
    """试着打开一个设备。成功返回 None，失败返回错误说明。

    设备能在列表里看到，不代表能打开——Windows 上 WDM-KS 的条目几乎必然失败，
    所以配置前必须先试开，否则要等到运行时才炸。
    """
    device = sd.query_devices(index)
    rate = max(1, int(device["default_samplerate"]))
    try:
        if kind == "input":
            stream = sd.InputStream(
                device=index,
                samplerate=rate,
                channels=1,
                dtype="int16",
                blocksize=int(rate * 0.03),
                callback=lambda *_: None,
            )
        else:
            stream = sd.OutputStream(
                device=index,
                samplerate=rate,
                channels=1,
                dtype="int16",
                blocksize=int(rate * 0.03),
                callback=lambda outdata, *_: outdata.fill(0),
            )
        stream.start()
        time.sleep(0.15)
        stream.stop()
        stream.close()
        return None
    except Exception as exc:
        return str(exc)


def probe_report(refresh: bool = False) -> str:
    """逐个试开所有设备，列出哪些真的能用。"""
    if refresh:
        refresh_devices()
    devices = sd.query_devices()
    apis = _host_api_names()
    lines: list[str] = []

    for kind, label in (("input", "输入"), ("output", "输出")):
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        lines.append(f"=== {label}设备 ===")
        usable = 0
        for index, device in enumerate(devices):
            if device[channel_key] <= 0:
                continue
            error = probe_device(index, kind)
            name = " ".join(device["name"].split())[:42]
            api = apis[device["hostapi"]]
            if error is None:
                usable += 1
                lines.append(f"  [可用]   #{index:>3}  {api:<20}{name}")
            else:
                lines.append(f"  [不可用] #{index:>3}  {api:<20}{name} — {error}")
        lines.append(f"  小计 {usable} 个可用\n")

    return "\n".join(lines)
