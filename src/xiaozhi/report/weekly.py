"""家长周报：把学习记录聚合成一份能给家长看的报告。

统计部分是确定性的，叙述段落交给模型写——家长要的是人话，不是一堆数字。
模型写不出来时，统计部分照样输出。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..store.models import (
    MASTERY_FAILED,
    MASTERY_MASTERED,
    MASTERY_PARTIAL,
    QuestionRecord,
)
from ..store.repository import Repository, recent_window


@dataclass
class SubjectStat:
    subject: str
    total: int = 0
    mastered: int = 0
    partial: int = 0
    failed: int = 0

    @property
    def weak(self) -> int:
        return self.partial + self.failed


@dataclass
class WeeklyReport:
    start: datetime
    end: datetime
    days: int
    total: int = 0
    by_subject: list[SubjectStat] = field(default_factory=list)
    weak_topics: list[tuple[str, int]] = field(default_factory=list)
    daily_counts: list[tuple[str, int]] = field(default_factory=list)
    mastered: int = 0
    partial: int = 0
    failed: int = 0
    unknown: int = 0
    turns: int = 0

    @property
    def judged(self) -> int:
        return self.mastered + self.partial + self.failed

    @property
    def mastery_rate(self) -> float | None:
        """掌握率只按"已判定"的题算，否则未知题会把比例摊薄得没意义。"""
        if not self.judged:
            return None
        return self.mastered / self.judged


def build_report(
    repo: Repository, days: int = 7, now: datetime | None = None
) -> WeeklyReport:
    start, end = recent_window(days, now)
    return aggregate(repo.questions_between(start, end), start, end, days)


def aggregate(
    records: list[QuestionRecord], start: datetime, end: datetime, days: int
) -> WeeklyReport:
    subjects: dict[str, SubjectStat] = {}
    topics: Counter[str] = Counter()
    daily: Counter[str] = Counter()
    report = WeeklyReport(start=start, end=end, days=days)

    for record in records:
        report.total += 1
        report.turns += record.turns

        stat = subjects.setdefault(record.subject or "其他", SubjectStat(record.subject or "其他"))
        stat.total += 1

        if record.mastery == MASTERY_MASTERED:
            stat.mastered += 1
            report.mastered += 1
        elif record.mastery == MASTERY_PARTIAL:
            stat.partial += 1
            report.partial += 1
            _count_topics(record, topics)
        elif record.mastery == MASTERY_FAILED:
            stat.failed += 1
            report.failed += 1
            _count_topics(record, topics)
        else:
            report.unknown += 1

        daily[record.created_at.strftime("%m-%d")] += 1

    report.by_subject = sorted(subjects.values(), key=lambda item: (-item.total, item.subject))
    report.weak_topics = topics.most_common(8)
    report.daily_counts = _fill_days(daily, start, end)
    return report


def _count_topics(record: QuestionRecord, topics: Counter[str]) -> None:
    """只统计"没掌握好"的题涉及的知识点——那才是要补的地方。"""
    for topic in record.topics:
        topics[topic] += 1


def _fill_days(daily: Counter[str], start: datetime, end: datetime) -> list[tuple[str, int]]:
    """补齐没练的日子，否则看不出间断。"""
    filled: list[tuple[str, int]] = []
    cursor = start
    while cursor.date() <= end.date():
        key = cursor.strftime("%m-%d")
        filled.append((key, daily.get(key, 0)))
        cursor += timedelta(days=1)
    return filled


def describe_stats(report: WeeklyReport) -> str:
    """把统计摊平成给模型看的文本。"""
    lines = [
        f"统计区间：{report.start:%Y-%m-%d} 至 {report.end:%Y-%m-%d}，共 {report.days} 天",
        f"讲题总数：{report.total} 道",
        f"掌握：{report.mastered} 道，部分掌握：{report.partial} 道，"
        f"未掌握：{report.failed} 道，未判定：{report.unknown} 道",
    ]
    if report.by_subject:
        subjects = "、".join(
            f"{item.subject} {item.total} 道（未掌握好 {item.weak} 道）"
            for item in report.by_subject
        )
        lines.append(f"学科分布：{subjects}")
    if report.weak_topics:
        topics = "、".join(f"{name}（{count} 次）" for name, count in report.weak_topics)
        lines.append(f"薄弱知识点：{topics}")
    empty_days = [day for day, count in report.daily_counts if count == 0]
    lines.append(f"没有练习的天数：{len(empty_days)} 天")
    return "\n".join(lines)


def render_plain(report: WeeklyReport, narrative: str = "") -> str:
    """纯文本报告，方便直接复制给家长。"""
    lines = [
        f"学习周报　{report.start:%Y-%m-%d} ~ {report.end:%Y-%m-%d}",
        "",
        f"讲题总数　{report.total} 道　（累计对话 {report.turns} 轮）",
    ]

    if report.judged:
        rate = report.mastery_rate or 0.0
        lines.append(
            f"掌握情况　掌握 {report.mastered}　部分掌握 {report.partial}　"
            f"未掌握 {report.failed}　→ 掌握率 {rate:.0%}"
        )
    else:
        lines.append("掌握情况　暂无判定（结题时才会归档）")
    if report.unknown:
        lines.append(f"　　　　　另有 {report.unknown} 道未判定")

    if report.by_subject:
        lines += ["", "学科分布"]
        for item in report.by_subject:
            detail = f"掌握 {item.mastered} / 部分 {item.partial} / 未掌握 {item.failed}"
            lines.append(f"　{item.subject}　{item.total} 道　（{detail}）")

    if report.weak_topics:
        lines += ["", "薄弱知识点"]
        for name, count in report.weak_topics:
            lines.append(f"　{name}　{count} 次")

    if report.daily_counts:
        lines += ["", "每日练习"]
        lines.append("　" + "　".join(f"{day} {count}" for day, count in report.daily_counts))

    if narrative:
        lines += ["", "本周小结", f"　{narrative}"]

    return "\n".join(lines)
