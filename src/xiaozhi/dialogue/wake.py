"""ASR 关键词唤醒。

开源唤醒词方案（openWakeWord / Porcupine）没有现成的中文模型，自训成本很高，
所以这里改成：本地 VAD 检测到有人说话 → 送一次 ASR → 转写里出现唤醒词就唤醒。
好处是无需训练、中文准确；代价是待机时每说一句话会消耗一次识别额度，
所以 VAD 门控是必须的——静音永远不会送到云端。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TRAILING_PUNCT = " \t，。、,.!?！？：:；;"
_SEPARATORS = " \t，。、,.!?！？：:；;"


def _compact(text: str) -> str:
    return re.sub(f"[{re.escape(_SEPARATORS)}]", "", text)


class WakeWordMatcher:
    def __init__(self, word: str, variants: Sequence[str] = ()) -> None:
        seeds = [word, *variants]
        self._candidates = sorted(_build_candidates(seeds), key=len, reverse=True)

    def match(self, text: str) -> tuple[bool, str]:
        """返回 (是否命中唤醒词, 去掉唤醒词后的内容)。"""
        if not text:
            return False, text

        for candidate in self._candidates:
            # 允许识别结果在唤醒词中间插入标点或空格
            pattern = f"[{re.escape(_SEPARATORS)}]*".join(
                re.escape(char) for char in candidate
            )
            found = re.search(pattern, text)
            if found:
                remaining = (text[: found.start()] + text[found.end() :]).strip(
                    _TRAILING_PUNCT
                )
                return True, remaining
        return False, text


def _build_candidates(seeds: Sequence[str]) -> set[str]:
    candidates: set[str] = set()
    for seed in seeds:
        compact = _compact(seed)
        if not compact:
            continue
        # 单字变体（小志）也要能配上叠词形式（小志小志）
        for form in (compact, compact + compact):
            candidates.add(form)
            half, remainder = divmod(len(form), 2)
            # 叠词只说一半（"小智小智" → "小智"）也算唤醒
            if remainder == 0 and half > 0 and form[:half] == form[half:]:
                candidates.add(form[:half])
    return candidates
