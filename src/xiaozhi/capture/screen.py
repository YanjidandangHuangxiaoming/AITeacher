"""屏幕截图。"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image

from ..config import CaptureConfig


class ScreenCapturer:
    def __init__(self, cfg: CaptureConfig, output_dir: Path) -> None:
        self._cfg = cfg
        self._output_dir = output_dir

    def capture(self, delay_s: float | None = None) -> Path:
        """延迟若干秒后截屏，返回保存的 PNG 路径。"""
        wait = self._cfg.delay_s if delay_s is None else delay_s
        if wait > 0:
            time.sleep(wait)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        target = self._output_dir / f"shot_{datetime.now():%Y%m%d_%H%M%S}.png"

        with mss.mss() as sct:
            shot = sct.grab(self._resolve_monitor(sct))
            Image.frombytes("RGB", shot.size, shot.rgb).save(target, "PNG")
        return target

    def _resolve_monitor(self, sct: "mss.base.MSSBase") -> dict[str, int]:
        region = self._cfg.region
        if region and len(region) == 4:
            left, top, width, height = region
            return {"left": left, "top": top, "width": width, "height": height}
        return dict(sct.monitors[1])  # 主显示器全屏
