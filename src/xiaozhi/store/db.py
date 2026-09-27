"""SQLite 连接与建表。

学习记录是本地唯一的持久化数据。表结构用 IF NOT EXISTS + user_version 管理，
不引入 ORM——这个规模用不上，理解成本反而更高。

连接只在事件循环线程里用（写入都是毫秒级的同步操作），所以不需要跨线程共享连接。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL,
    subject        TEXT    NOT NULL DEFAULT '',
    grade          TEXT    NOT NULL DEFAULT '',
    stem           TEXT    NOT NULL DEFAULT '',
    options_json   TEXT    NOT NULL DEFAULT '[]',
    figure         TEXT    NOT NULL DEFAULT '',
    student_answer TEXT    NOT NULL DEFAULT '',
    notes          TEXT    NOT NULL DEFAULT '',
    image_path     TEXT    NOT NULL DEFAULT '',
    source         TEXT    NOT NULL DEFAULT '',
    topics_json    TEXT    NOT NULL DEFAULT '[]',
    mastery        TEXT    NOT NULL DEFAULT '',
    summary        TEXT    NOT NULL DEFAULT '',
    turns          INTEGER NOT NULL DEFAULT 0,
    closed_at      TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    created_at  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at);
CREATE INDEX IF NOT EXISTS idx_questions_mastery ON questions(mastery);
CREATE INDEX IF NOT EXISTS idx_messages_question ON messages(question_id);
"""

IN_MEMORY = ":memory:"


class Database:
    def __init__(self, path: Path | str) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> Path | str:
        return self._path

    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if str(self._path) != IN_MEMORY:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(self._path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
