"""结题归档解析的测试（纯逻辑，不联网）。"""

from __future__ import annotations

from xiaozhi.llm.summarizer import parse_summary, render_transcript


def test_parses_clean_json() -> None:
    summary = parse_summary(
        '{"topics": ["配方法"], "mastery": "部分掌握", "summary": "卡在配方"}'
    )
    assert summary.topics == ["配方法"]
    assert summary.mastery == "部分掌握"
    assert summary.summary == "卡在配方"


def test_tolerates_code_fence_and_prose() -> None:
    summary = parse_summary(
        '好的，总结如下：\n```json\n{"topics": ["受力分析"], "mastery": "掌握"}\n```'
    )
    assert summary.topics == ["受力分析"]
    assert summary.mastery == "掌握"


def test_rejects_invalid_mastery() -> None:
    assert parse_summary('{"mastery": "还行"}').mastery == ""


def test_accepts_string_topics() -> None:
    assert parse_summary('{"topics": "配方法"}').topics == ["配方法"]


def test_limits_topics_to_three() -> None:
    summary = parse_summary('{"topics": ["a", "b", "c", "d", "e"]}')
    assert summary.topics == ["a", "b", "c"]


def test_drops_blank_topics() -> None:
    assert parse_summary('{"topics": ["配方法", "", "  "]}').topics == ["配方法"]


def test_truncates_long_summary() -> None:
    summary = parse_summary('{"summary": "' + "字" * 200 + '"}')
    assert len(summary.summary) == 80


def test_garbage_returns_empty_summary() -> None:
    summary = parse_summary("这道题他学得还行")
    assert summary.topics == []
    assert summary.mastery == ""
    assert summary.summary == ""


def test_empty_text_is_safe() -> None:
    assert parse_summary("").topics == []


def test_render_transcript_labels_speakers() -> None:
    text = render_transcript(
        [
            {"role": "user", "content": "怎么算"},
            {"role": "assistant", "content": "先看题干"},
        ]
    )
    assert text == "学生：怎么算\n小智：先看题干"


def test_render_transcript_caps_length() -> None:
    messages = [{"role": "user", "content": "字" * 5000}]
    assert len(render_transcript(messages)) <= 6000
