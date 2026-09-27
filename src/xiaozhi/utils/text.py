"""文本切分：把流式生成的回复切成适合送 TTS 的短句。"""

from __future__ import annotations

# 句号级断点：出现即切
HARD_ENDERS = "。！？!?；;\n"
# 逗号级断点：只有缓冲区够长才切，用来压低首字延迟
SOFT_ENDERS = "，,、：:"
SOFT_BREAK_MIN = 30
# 逗号切分时，切出的片段至少要有这么长，避免产出"好，"这种碎片
MIN_PIECE = 10

_FILLER = set(" \t\r\n" + HARD_ENDERS + SOFT_ENDERS)


def take_sentences(buffer: str) -> tuple[list[str], str]:
    """从流式缓冲区里切出可以送 TTS 的句子，返回 (句子列表, 剩余缓冲区)。"""
    sentences: list[str] = []
    start = 0

    for index, char in enumerate(buffer):
        if char in HARD_ENDERS:
            _flush(buffer[start : index + 1], sentences)
            start = index + 1

    return sentences, _split_long_remainder(buffer[start:], sentences)


def _flush(piece: str, sentences: list[str]) -> None:
    """只有包含实际内容的片段才值得送去合成，纯标点/空白直接丢掉。"""
    stripped = piece.strip()
    if any(char not in _FILLER for char in stripped):
        sentences.append(stripped)


def _split_long_remainder(rest: str, sentences: list[str]) -> str:
    while len(rest) >= SOFT_BREAK_MIN:
        cut = next(
            (index for index, char in enumerate(rest) if index >= MIN_PIECE and char in SOFT_ENDERS),
            -1,
        )
        if cut < 0:
            break
        _flush(rest[: cut + 1], sentences)
        rest = rest[cut + 1 :]
    return rest
