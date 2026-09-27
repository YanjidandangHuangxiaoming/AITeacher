"""学习记录存储的测试。用内存库，不碰磁盘。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from xiaozhi.store.db import Database
from xiaozhi.store.models import (
    MASTERY_FAILED,
    MASTERY_MASTERED,
    MASTERY_PARTIAL,
    QuestionSummary,
)
from xiaozhi.store.repository import Repository, recent_window
from xiaozhi.vision.question_parser import Question


@pytest.fixture
def repo() -> Repository:
    database = Database(":memory:")
    yield Repository(database)
    database.close()


def _question(stem: str = "解方程 x+1=2", subject: str = "数学") -> Question:
    return Question(subject=subject, stem=stem, options=["A. 1"], student_answer="A")


def test_create_and_read_back(repo: Repository) -> None:
    question_id = repo.create_question(_question(), source="shot", image_path="a.png")
    record = repo.get_question(question_id)
    assert record is not None
    assert record.stem == "解方程 x+1=2"
    assert record.subject == "数学"
    assert record.source == "shot"
    assert record.closed is False
    assert record.mastery == ""
    assert record.topics == []


def test_get_missing_question_returns_none(repo: Repository) -> None:
    assert repo.get_question(9999) is None


def test_messages_keep_order(repo: Repository) -> None:
    question_id = repo.create_question(_question())
    repo.append_message(question_id, "user", "第一句")
    repo.append_message(question_id, "assistant", "第二句")
    assert repo.messages_for(question_id) == [("user", "第一句"), ("assistant", "第二句")]


def test_close_question_records_summary(repo: Repository) -> None:
    question_id = repo.create_question(_question())
    repo.close_question(
        question_id,
        turns=3,
        summary=QuestionSummary(
            topics=["一元二次方程-配方法"], mastery=MASTERY_PARTIAL, summary="卡在配方"
        ),
    )
    record = repo.get_question(question_id)
    assert record is not None
    assert record.closed is True
    assert record.turns == 3
    assert record.topics == ["一元二次方程-配方法"]
    assert record.mastery == MASTERY_PARTIAL
    assert record.summary == "卡在配方"


def test_close_without_summary_still_closes(repo: Repository) -> None:
    question_id = repo.create_question(_question())
    repo.close_question(question_id, turns=0)
    record = repo.get_question(question_id)
    assert record is not None
    assert record.closed is True
    assert record.mastery == ""


def test_weak_questions_only_include_partial_and_failed(repo: Repository) -> None:
    good = repo.create_question(_question("题A"))
    partial = repo.create_question(_question("题B"))
    failed = repo.create_question(_question("题C"))
    unknown = repo.create_question(_question("题D"))
    repo.close_question(good, 1, QuestionSummary(mastery=MASTERY_MASTERED))
    repo.close_question(partial, 1, QuestionSummary(mastery=MASTERY_PARTIAL))
    repo.close_question(failed, 1, QuestionSummary(mastery=MASTERY_FAILED))
    repo.close_question(unknown, 1)

    assert {record.stem for record in repo.weak_questions()} == {"题B", "题C"}


def test_recent_questions_newest_first(repo: Repository) -> None:
    for index in range(3):
        repo.create_question(_question(f"题{index}"))
    assert [record.stem for record in repo.recent_questions(2)] == ["题2", "题1"]


def test_questions_between_respects_window(repo: Repository) -> None:
    repo.create_question(_question())
    now = datetime.now()
    assert len(repo.questions_between(now - timedelta(hours=1), now + timedelta(hours=1))) == 1
    assert repo.questions_between(now - timedelta(days=10), now - timedelta(days=5)) == []


def test_question_created_now_is_inside_window(repo: Repository) -> None:
    """回归：刚讲完立刻查周报，这道题必须出现。

    时间戳按秒截断，created_at 和窗口右界会落在同一秒；
    右界若用严格小于，今天的题会被整批丢掉（报告显示 0 道）。
    """
    repo.create_question(_question())
    start, end = recent_window(7)
    assert len(repo.questions_between(start, end)) == 1


def test_deleting_question_cascades_to_messages() -> None:
    database = Database(":memory:")
    try:
        repo = Repository(database)
        question_id = repo.create_question(_question())
        repo.append_message(question_id, "user", "x")
        database.connection().execute("DELETE FROM questions WHERE id = ?", (question_id,))
        database.connection().commit()
        assert repo.messages_for(question_id) == []
    finally:
        database.close()


def test_count_questions(repo: Repository) -> None:
    assert repo.count_questions() == 0
    repo.create_question(_question())
    assert repo.count_questions() == 1


def test_recent_window_starts_at_midnight() -> None:
    now = datetime(2026, 9, 27, 15, 30)
    start, end = recent_window(7, now)
    assert start == datetime(2026, 9, 21, 0, 0)
    assert end == now


def test_recent_window_of_one_day_covers_today() -> None:
    now = datetime(2026, 9, 27, 15, 30)
    start, _ = recent_window(1, now)
    assert start == datetime(2026, 9, 27, 0, 0)


def test_memory_database_does_not_touch_disk(tmp_path) -> None:
    """内存库不应该在磁盘上留下文件。"""
    database = Database(":memory:")
    database.connection()
    database.close()
    assert list(tmp_path.iterdir()) == []
