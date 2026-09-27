"""从模型输出里抠 JSON。

云端模型经常在 JSON 外面裹一层解释文字或者 ``` 代码块，直接 json.loads 会炸。
"""

from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """返回文本里的第一个 JSON 对象；抠不出来或格式不对时返回 None。"""
    if not text:
        return None

    cleaned = _CODE_FENCE.sub("", text.strip())
    match = _JSON_BLOCK.search(cleaned)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
