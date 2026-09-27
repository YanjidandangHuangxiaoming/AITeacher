"""学习记录的读写。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from sqlite3 import Row

from ..vision.question_parser import Question
from .db import Database
from .models import QuestionRecord, QuestionSummary


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_record(row: Row) -> QuestionRecord:
    return QuestionRecord(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        subject=row["subject"],
        grade=row["grade"],
        stem=row["stem"],
        student_answer=row["student_answer"],
        source=row["source"],
        topics=json.loads(row["topics_json"] or "[]"),
        mastery=row["mastery"],
        summary=row["summary"],
        turns=row["turns"],
        closed=bool(row["closed_at"]),
    )


class Repository:
    def __init__(self, database: Database) -> None:
        self._db = database

    # ---------- 写入 ----------

    def create_question(
        self, question: Question, source: str = "", image_path: str = ""
    ) -> int:
        conn = self._db.connection()
        cursor = conn.execute(
            """
            INSERT INTO questions
                (created_at, subject, grade, stem, options_json, figure,
                 student_answer, notes, image_path, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                question.subject,
                question.grade,
                question.stem,
                json.dumps(question.options, ensure_ascii=False),
                question.figure,
                question.student_answer,
                question.notes,
                image_path,
                source,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def append_message(self, question_id: int, role: str, content: str) -> None:
        conn = self._db.connection()
        conn.execute(
            """
            INSERT INTO messages (question_id, created_at, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (question_id, _now(), role, content),
        )
        conn.commit()

    def close_question(
        self, question_id: int, turns: int, summary: QuestionSummary | None = None
    ) -> None:
        payload = summary or QuestionSummary()
        conn = self._db.connection()
        conn.execute(
            """
            UPDATE questions
               SET turns = ?, topics_json = ?, mastery = ?, summary = ?, closed_at = ?
             WHERE id = ?
            """,
            (
                turns,
                json.dumps(payload.topics, ensure_ascii=False),
                payload.mastery,
                payload.summary,
                _now(),
                question_id,
            ),
        )
        conn.commit()

    # ---------- 查询 ----------

    def get_question(self, question_id: int) -> QuestionRecord | None:
        row = self._db.connection().execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
        return _to_record(row) if row else None

    def recent_questions(self, limit: int = 10) -> list[QuestionRecord]:
        rows = self._db.connection().execute(
            "SELECT * FROM questions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_to_record(row) for row in rows]

    def weak_questions(self, limit: int = 10) -> list[QuestionRecord]:
        rows = self._db.connection().execute(
            """
            SELECT * FROM questions
             WHERE mastery IN ('部分掌握', '未掌握')
             ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_to_record(row) for row in rows]

    def questions_since(self, since: datetime) -> list[QuestionRecord]:
        rows = self._db.connection().execute(
            "SELECT * FROM questions WHERE created_at >= ? ORDER BY created_at",
            (since.isoformat(timespec="seconds"),),
        ).fetchall()
        return [_to_record(row) for row in rows]

    def questions_between(self, start: datetime, end: datetime) -> list[QuestionRecord]:
        """取时间窗内的题，闭区间。

        右界必须用 `<=`：时间戳按秒截断，刚讲完立刻生成周报时
        created_at 和 end 会落在同一秒，用 `<` 会把今天的题全丢掉。
        """
        rows = self._db.connection().execute(
            """
            SELECT * FROM questions
             WHERE created_at >= ? AND created_at <= ?
             ORDER BY created_at
            """,
            (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
        ).fetchall()
        return [_to_record(row) for row in rows]

    def messages_for(self, question_id: int) -> list[tuple[str, str]]:
        rows = self._db.connection().execute(
            "SELECT role, content FROM messages WHERE question_id = ? ORDER BY id",
            (question_id,),
        ).fetchall()
        return [(row["role"], row["content"]) for row in rows]

    def count_questions(self) -> int:
        row = self._db.connection().execute("SELECT COUNT(*) AS n FROM questions").fetchone()
        return int(row["n"]) if row else 0


def recent_window(days: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    """返回最近 days 天的时间窗（含今天的起点）。"""
    end = now or datetime.now()
    start = (end - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end
