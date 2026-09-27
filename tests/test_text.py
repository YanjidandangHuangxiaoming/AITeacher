"""流式文本切句的单元测试。"""

from __future__ import annotations

from xiaozhi.utils.text import MIN_PIECE, SOFT_BREAK_MIN, take_sentences


def test_splits_on_hard_enders() -> None:
    sentences, rest = take_sentences("先看题干。它问的是什么？")
    assert sentences == ["先看题干。", "它问的是什么？"]
    assert rest == ""


def test_keeps_incomplete_tail() -> None:
    sentences, rest = take_sentences("先看题干。它问的是")
    assert sentences == ["先看题干。"]
    assert rest == "它问的是"


def test_skips_blank_pieces() -> None:
    sentences, rest = take_sentences("。\n。先看题干。")
    assert sentences == ["先看题干。"]
    assert rest == ""


def test_long_text_breaks_at_comma() -> None:
    text = "好，这道题给了一个直角三角形，然后要求我们算出它斜边的长度是多少"
    sentences, rest = take_sentences(text)
    assert sentences == ["好，这道题给了一个直角三角形，"]
    assert rest == "然后要求我们算出它斜边的长度是多少"


def test_short_clause_is_not_split_on_comma() -> None:
    sentences, rest = take_sentences("我们先看，再看")
    assert sentences == []
    assert rest == "我们先看，再看"


def test_comma_break_ignores_leading_fragment() -> None:
    # 逗号太靠前时不能切，否则会产出"好，"这种碎片
    text = "嗯，这道题给了一个直角三角形，然后要求我们算出斜边的长度是多少"
    sentences, _ = take_sentences(text)
    assert sentences
    assert all(len(sentence) >= MIN_PIECE for sentence in sentences)


def test_incremental_streaming_accumulates() -> None:
    buffer = ""
    collected: list[str] = []
    for piece in ["先看", "题干。", "它问", "的是什么？"]:
        buffer += piece
        ready, buffer = take_sentences(buffer)
        collected.extend(ready)
    if buffer.strip():
        collected.append(buffer.strip())
    assert collected == ["先看题干。", "它问的是什么？"]


def test_buffer_exactly_at_threshold_does_not_split_yet() -> None:
    sentences, rest = take_sentences("短" * (SOFT_BREAK_MIN - 1))
    assert sentences == []
    assert rest == "短" * (SOFT_BREAK_MIN - 1)
