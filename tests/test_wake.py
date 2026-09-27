"""唤醒词匹配的单元测试。"""

from __future__ import annotations

import pytest

from xiaozhi.dialogue.wake import WakeWordMatcher


@pytest.fixture
def matcher() -> WakeWordMatcher:
    return WakeWordMatcher("小智小智")


@pytest.fixture
def homophone_matcher() -> WakeWordMatcher:
    # 实测中 ASR 会把"小智"写成"小志"、"这小志"，必须也能唤醒
    return WakeWordMatcher("小智小智", ("小志", "小知", "小致", "小至"))


@pytest.mark.parametrize(
    "text",
    [
        "小智小智",
        "小智小智，这道题怎么做",
        "小智小智这道题怎么做",
        "小 智 小 智，在吗",
        "喂，小智小智，帮我看下题",
    ],
)
def test_matches_full_wake_word(matcher: WakeWordMatcher, text: str) -> None:
    matched, _ = matcher.match(text)
    assert matched


@pytest.mark.parametrize("text", ["小智", "小智，这道题怎么做", "小智帮我看题"])
def test_half_of_doubled_word_also_wakes(matcher: WakeWordMatcher, text: str) -> None:
    matched, _ = matcher.match(text)
    assert matched


@pytest.mark.parametrize(
    "text",
    ["这道题怎么做", "老师你好", "嗯", "小题大做"],
)
def test_does_not_wake_on_unrelated_speech(matcher: WakeWordMatcher, text: str) -> None:
    matched, _ = matcher.match(text)
    assert matched is False


def test_remainder_strips_wake_word_and_punctuation(matcher: WakeWordMatcher) -> None:
    matched, remainder = matcher.match("小智小智，这道题怎么做")
    assert matched
    assert remainder == "这道题怎么做"


def test_remainder_prefers_longest_candidate(matcher: WakeWordMatcher) -> None:
    # 命中「小智小智」时不应该只吃掉「小智」
    _, remainder = matcher.match("小智小智，帮我讲讲")
    assert remainder == "帮我讲讲"


def test_empty_text_is_safe(matcher: WakeWordMatcher) -> None:
    assert matcher.match("") == (False, "")


def test_custom_wake_word() -> None:
    custom = WakeWordMatcher("小爱小爱")
    assert custom.match("小爱小爱，你好")[0] is True
    assert custom.match("小智小智，你好")[0] is False


# ---------- 同音字变体 ----------


@pytest.mark.parametrize(
    "text",
    [
        "小志小志",
        "小志小志，这道题怎么做",
        "小志，这道题怎么做",
        "这小志。",
        "小知小知",
        "小致小致",
        "小至，帮我看题",
    ],
)
def test_homophone_variants_wake(homophone_matcher: WakeWordMatcher, text: str) -> None:
    matched, _ = homophone_matcher.match(text)
    assert matched, f"{text!r} 应该能唤醒"


def test_variant_remainder_is_stripped(homophone_matcher: WakeWordMatcher) -> None:
    matched, remainder = homophone_matcher.match("小志小志，帮我看看这道题")
    assert matched
    assert remainder == "帮我看看这道题"


def test_variant_without_variants_does_not_wake() -> None:
    """没配变体时，同音字不应误唤醒——变体是显式配置的。"""
    plain = WakeWordMatcher("小智小智")
    assert plain.match("小志小志")[0] is False


def test_variants_do_not_cause_false_positive(homophone_matcher: WakeWordMatcher) -> None:
    for text in ["这道题怎么做", "小题大做", "老师你好", "嗯"]:
        assert homophone_matcher.match(text)[0] is False, f"{text!r} 不该唤醒"
