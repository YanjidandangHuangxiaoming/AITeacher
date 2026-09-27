"""环境自检：配置、屏幕截图、LLM、视觉识题、音频设备、语音链路。

用法：
    python scripts/check_env.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from xiaozhi.asr.cloud import ParaformerAsr  # noqa: E402
from xiaozhi.audio.capture import AudioCapture  # noqa: E402
from xiaozhi.audio.voiceprint import VoiceprintStore  # noqa: E402
from xiaozhi.audio.devices import (  # noqa: E402
    describe_device,
    effective_device,
    looks_like_headset,
    refresh_devices,
)
from xiaozhi.capture.screen import ScreenCapturer  # noqa: E402
from xiaozhi.config import PROJECT_ROOT, Settings, get_settings, missing_keys  # noqa: E402
from xiaozhi.ingest.wechat import resolve_watch_dirs  # noqa: E402
from xiaozhi.llm.client import DeepSeekClient  # noqa: E402
from xiaozhi.tts.cloud import CosyVoiceTts  # noqa: E402
from xiaozhi.vision.cloud_vl import QwenVlClient  # noqa: E402

SELF_CHECK_SENTENCE = "你好，这是一次自检。"


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def report(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        return ok


def check_config(checker: Checker, settings: Settings) -> bool:
    missing = missing_keys(settings)
    return checker.report(
        "配置加载",
        not missing,
        f"缺少 {', '.join(missing)}" if missing else str(PROJECT_ROOT),
    )


def check_screenshot(checker: Checker, settings: Settings) -> None:
    try:
        shot = ScreenCapturer(settings.capture, settings.output_dir()).capture(delay_s=0)
        checker.report("屏幕截图", True, shot.name)
    except Exception as exc:
        checker.report("屏幕截图", False, repr(exc))


def check_audio_devices(checker: Checker, settings: Settings) -> None:
    for kind, label in (("input", "麦克风"), ("output", "扬声器")):
        try:
            index = effective_device(kind, getattr(settings.audio, f"{kind}_device"))
            name = describe_device(kind, index)
            hint = ""
            if kind == "output" and not looks_like_headset(name):
                hint = "　注意：这不像耳机，外放时可能自己打断自己"
            checker.report(f"{label}设备", True, f"{name}{hint}")
        except Exception as exc:
            checker.report(f"{label}设备", False, str(exc))

    try:
        capture = AudioCapture(settings.audio)
        capture.start()
        frames = 0
        try:
            for _ in range(40):
                if capture.frame(timeout=0.05) is not None:
                    frames += 1
                    if frames >= 5:
                        break
        finally:
            capture.stop()
        checker.report("麦克风采集", frames > 0, f"收到 {frames} 帧")
    except Exception as exc:
        checker.report("麦克风采集", False, repr(exc))


async def check_llm(checker: Checker, settings: Settings) -> None:
    try:
        reply = ""
        async for piece in DeepSeekClient(settings.llm).stream_chat(
            [{"role": "user", "content": "只回复两个字：收到"}]
        ):
            reply += piece
        checker.report("DeepSeek 对话", bool(reply.strip()), reply.strip()[:30])
    except Exception as exc:
        checker.report("DeepSeek 对话", False, repr(exc))


async def check_vision(checker: Checker, settings: Settings) -> None:
    try:
        test_image = _make_test_image(PROJECT_ROOT / "data" / "cache" / "check_env.png")
        raw = await QwenVlClient(settings.vision).recognize(test_image)
        checker.report("VL 识题", bool(raw.strip()), " ".join(raw.split())[:60])
    except Exception as exc:
        checker.report("VL 识题", False, repr(exc))


def check_voice_roundtrip(checker: Checker, settings: Settings) -> None:
    """合成一句话再识别回来，一次验证 TTS 与 ASR 两条链路。"""
    try:
        tts = CosyVoiceTts(settings.tts, settings.audio.sample_rate)
        pcm = b"".join(tts.stream(SELF_CHECK_SENTENCE, lambda: False))
    except Exception as exc:
        checker.report("语音合成", False, repr(exc))
        return

    if not checker.report("语音合成", bool(pcm), f"收到 {len(pcm)} 字节音频"):
        return

    try:
        audio = np.frombuffer(pcm, dtype=np.int16)
        text = ParaformerAsr(settings.asr, settings.audio.sample_rate).transcribe(audio)
        checker.report("语音识别", bool(text.strip()), f"识别回来「{text.strip()[:40]}」")
    except Exception as exc:
        checker.report("语音识别", False, repr(exc))


def check_watch_dirs(checker: Checker, settings: Settings) -> None:
    """确认图片监控目录可用（不存在的会在启动时自动创建）。"""
    paths = resolve_watch_dirs(settings)
    pending = [str(path) for path in paths if not path.exists()]
    wechat = [path for path in paths if "msg" in path.parts and "file" in path.parts]
    detail = f"{len(paths)} 个目录"
    if wechat:
        detail += f"，其中微信 {len(wechat)} 个"
    if pending:
        detail += f"；启动时会创建：{', '.join(pending)}"
    checker.report("图片监控目录", True, detail)


def check_voiceprint(checker: Checker, settings: Settings) -> None:
    """验证声纹模型能加载。模型加载慢，这里也顺便帮启动预热一次。"""
    if not settings.voiceprint.enabled or settings.interrupt.mode != "voiceprint":
        return
    try:
        store = VoiceprintStore(settings.voiceprint, settings.voiceprint_path())
        store.warmup()
        if store.load():
            checker.report("声纹模型", True, f"已注册，判定阈值 {store.threshold:.2f}")
        else:
            checker.report(
                "声纹模型", True, "模型可用；还没注册，跑 python -m xiaozhi enroll"
            )
    except Exception as exc:
        checker.report("声纹模型", False, repr(exc))


def _make_test_image(path: Path) -> Path:
    """生成一张白底黑字的测试图，用来验证识题链路。"""
    image = Image.new("RGB", (420, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 30), "1/2 + 1/3 = ?", fill="black")
    draw.text((24, 70), "A. 2/5   B. 5/6   C. 1/6", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path


async def main() -> int:
    settings = get_settings()
    checker = Checker()

    refresh_devices()

    if not check_config(checker, settings):
        print("\n请复制 .env.example 为 .env 并填入上述 Key 后重试。")
        return 1

    check_screenshot(checker, settings)
    check_watch_dirs(checker, settings)
    check_audio_devices(checker, settings)
    check_voiceprint(checker, settings)
    await check_llm(checker, settings)
    await check_vision(checker, settings)
    check_voice_roundtrip(checker, settings)

    print()
    if checker.failed:
        print(f"自检未通过：{checker.passed} 项通过，{checker.failed} 项失败。")
        return 1
    print(f"自检全部通过（{checker.passed} 项）。可以跑 python -m xiaozhi voice 了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
