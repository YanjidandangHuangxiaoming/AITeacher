"""学习记录的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# 掌握情况：结题时由模型根据整段对话判断
MASTERY_MASTERED = "掌握"
MASTERY_PARTIAL = "部分掌握"
MASTERY_FAILED = "未掌握"
MASTERY_UNKNOWN = ""
# 这两种算「错题」，要进错题本
WEAK_MASTERY = (MASTERY_PARTIAL, MASTERY_FAILED)


@dataclass
class QuestionSummary:
    """一道题讲完后归档出来的信息。"""

    topics: list[str] = field(default_factory=list)
    mastery: str = MASTERY_UNKNOWN
    summary: str = ""


@dataclass
class QuestionRecord:
    id: int
    created_at: datetime
    subject: str
    grade: str
    stem: str
    student_answer: str
    source: str
    topics: list[str]
    mastery: str
    summary: str
    turns: int
    closed: bool

    @property
    def is_weak(self) -> bool:
        return self.mastery in WEAK_MASTERY

    @property
    def is_unknown(self) -> bool:
        return self.mastery == MASTERY_UNKNOWN

    def brief(self, width: int = 40) -> str:
        text = " ".join(self.stem.split())
        if len(text) > width:
            text = text[: width - 1] + "…"
        return text or "（题干为空）"
