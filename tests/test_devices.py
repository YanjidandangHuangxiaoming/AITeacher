"""音频设备名判定的单元测试（纯字符串逻辑，不访问硬件）。"""

from __future__ import annotations

import pytest

from xiaozhi.audio.devices import looks_like_headset, looks_like_loopback


@pytest.mark.parametrize(
    "name",
    [
        "耳机 ()",
        "AriPlus Pro Hands-Free AG Audio",
        "Headphones (Realtek Audio)",
        "WH-1000XM4 Hands-Free",
        "Bluetooth Audio",
    ],
)
def test_headset_detection_positive(name: str) -> None:
    assert looks_like_headset(name)


@pytest.mark.parametrize(
    "name",
    [
        "扬声器/听筒 (Realtek Audio)",
        "LG HDR 4K (NVIDIA High Definition Audio)",
        "Speakers (Realtek HD Audio output)",
    ],
)
def test_headset_detection_negative(name: str) -> None:
    assert looks_like_headset(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "立体声混音 (Realtek Audio)",
        "Stereo Mix (Realtek Audio)",
        "主声音捕获驱动程序",
        "What U Hear (Sound Blaster)",
    ],
)
def test_loopback_detection_positive(name: str) -> None:
    assert looks_like_loopback(name)


@pytest.mark.parametrize(
    "name",
    [
        "麦克风 (Realtek HD Audio Mic input)",
        "AriPlus Pro Hands-Free AG Audio",
        "线路输入 (Realtek HD Audio Line input)",
    ],
)
def test_loopback_detection_negative(name: str) -> None:
    assert looks_like_loopback(name) is False


def test_none_and_empty_are_safe() -> None:
    assert looks_like_headset(None) is False
    assert looks_like_headset("") is False
    assert looks_like_loopback(None) is False
    assert looks_like_loopback("") is False
