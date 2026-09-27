"""周报聚合与渲染的测试（纯逻辑，不联网）。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from xiaozhi.report.weekly import aggregate, build_report, describe_stats, render_plain
from xiaozhi.store.db import Database
from xiaozhi.store.models import (
    MASTERY_FAILED,
    MASTERY_MASTERED,
    MASTERY_PARTIAL,
    QuestionRecord,
    QuestionSummary,
)
from xiaozhi.store.repository import Repository
from xiaozhi.vision.question_parser import Question

NOW = datetime(2026, 9, 27, 20, 0)
START = datetime(2026, 9, 21, 0, 0)
DAYS = 7


def _record(
    day_offset: int,
    subject: str = "数学",
    mastery: str = "",
    topics: list[str] | None = None,
    turns: int = 2,
) -> QuestionRecord:
    return QuestionRecord(
        id=day_offset + 1,
        created_at=NOW - timedelta(days=day_offset),
        subject=subject,
        grade="初二",
        stem=f"{subject}的一道题",
        student_answer="",
        source="shot",
        topics=topics or [],
        mastery=mastery,
        summary="",
        turns=turns,
        closed=True,
    )


def _build(records: list[QuestionRecord]):
    return aggregate(records, START, NOW, DAYS)


def test_build_report_includes_just_created_question() -> None:
    """回归：用真实时钟建一条记录，立刻生成周报，必须统计到它。

    固定时间戳测不出这个 bug——真实的 created_at 和窗口右界会落在同一秒。
    """
    database = Database(":memory:")
    try:
        repo = Repository(database)
        question_id = repo.create_question(Question(subject="数学", stem="刚讲完的题"))
        repo.close_question(
            question_id, turns=2, summary=QuestionSummary(mastery=MASTERY_MASTERED)
        )
        report = build_report(repo, 7)
        assert report.total == 1
        assert report.mastered == 1
    finally:
        database.close()


def test_counts_by_subject_and_mastery() -> None:
    report = _build(
        [
            _record(0, "数学", MASTERY_MASTERED),
            _record(1, "数学", MASTERY_PARTIAL, ["配方法"]),
            _record(2, "物理", MASTERY_FAILED, ["受力分析"]),
        ]
    )
    assert report.total == 3
    assert (report.mastered, report.partial, report.failed) == (1, 1, 1)
    assert [item.subject for item in report.by_subject] == ["数学", "物理"]
    assert report.by_subject[0].total == 2
    assert report.by_subject[0].weak == 1


def test_mastery_rate_ignores_unknown() -> None:
    report = _build(
        [
            _record(0, "数学", MASTERY_MASTERED),
            _record(0, "数学", MASTERY_MASTERED),
            _record(0, "数学", ""),
        ]
    )
    assert report.judged == 2
    assert report.unknown == 1
    assert report.mastery_rate == pytest.approx(1.0)


def test_mastery_rate_is_none_without_judgements() -> None:
    report = _build([_record(0, "数学", "")])
    assert report.mastery_rate is None


def test_weak_topics_only_count_unmastered_questions() -> None:
    report = _build(
        [
            _record(0, "数学", MASTERY_MASTERED, ["已经会的"]),
            _record(0, "数学", MASTERY_PARTIAL, ["配方法"]),
            _record(0, "数学", MASTERY_FAILED, ["配方法", "判别式"]),
        ]
    )
    topics = dict(report.weak_topics)
    assert topics["配方法"] == 2
    assert topics["判别式"] == 1
    assert "已经会的" not in topics


def test_turns_are_summed() -> None:
    report = _build([_record(0, turns=3), _record(1, turns=4)])
    assert report.turns == 7


def test_daily_counts_fill_empty_days() -> None:
    report = _build(
        [_record(0, mastery=MASTERY_MASTERED), _record(2, mastery=MASTERY_MASTERED)]
    )
    assert len(report.daily_counts) == DAYS
    assert report.daily_counts[-1][1] == 1  # 今天
    assert report.daily_counts[-3][1] == 1
    assert sum(count for _, count in report.daily_counts) == 2


def test_empty_report_is_safe() -> None:
    report = _build([])
    assert report.total == 0
    assert report.mastery_rate is None
    assert report.by_subject == []
    assert report.weak_topics == []


def test_render_empty_report() -> None:
    text = render_plain(_build([]))
    assert "讲题总数　0 道" in text
    assert "暂无判定" in text


def test_render_includes_subject_and_topics() -> None:
    report = _build(
        [
            _record(0, "数学", MASTERY_PARTIAL, ["配方法"]),
            _record(1, "物理", MASTERY_FAILED, ["受力分析"]),
        ]
    )
    text = render_plain(report)
    assert "学科分布" in text
    assert "数学" in text and "物理" in text
    assert "薄弱知识点" in text
    assert "配方法" in text


def test_render_includes_narrative() -> None:
    text = render_plain(_build([_record(0)]), "这周练得不错。")
    assert "本周小结" in text
    assert "这周练得不错。" in text


def test_render_omits_narrative_section_when_absent() -> None:
    assert "本周小结" not in render_plain(_build([_record(0)]))


def test_describe_stats_mentions_key_numbers() -> None:
    report = _build([_record(0, "数学", MASTERY_FAILED, ["配方法"])])
    text = describe_stats(report)
    assert "讲题总数：1 道" in text
    assert "未掌握：1 道" in text
    assert "配方法" in text
