"""图片处理。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

MAX_EDGE = 1600
JPEG_QUALITY = 88


def encode_image_data_uri(path: Path, max_edge: int = MAX_EDGE) -> str:
    """压缩到合理尺寸后编码为 data URI，降低上传体积与识别费用。"""
    with Image.open(path) as image:
        frame = image.convert("RGB")
        longest = max(frame.size)
        if longest > max_edge:
            scale = max_edge / longest
            frame = frame.resize(
                (round(frame.width * scale), round(frame.height * scale)),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
