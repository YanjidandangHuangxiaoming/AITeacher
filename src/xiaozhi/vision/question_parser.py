"""把视觉模型的输出规整成结构化题目。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..utils.json import extract_json_object


class Question(BaseModel):
    """一张试卷图片被识别后的结构化结果。"""

    subject: str = "其他"
    grade: str = ""
    stem: str = ""
    options: list[str] = Field(default_factory=list)
    figure: str = ""
    student_answer: str = ""
    notes: str = ""

    def is_empty(self) -> bool:
        return not self.stem.strip()

    def to_prompt(self) -> str:
        """转成投喂给对话模型的题目描述。"""
        lines = [f"【学科】{self.subject or '未知'}"]
        if self.grade:
            lines.append(f"【年级】{self.grade}")
        lines.append(f"【题干】{self.stem}")
        if self.options:
            lines.append("【选项】" + " ".join(self.options))
        if self.figure:
            lines.append(f"【图形信息】{self.figure}")
        if self.student_answer:
            lines.append(f"【学生已作答】{self.student_answer}")
        if self.notes:
            lines.append(f"【识别备注】{self.notes}")
        return "\n".join(lines)


def _coerce_options(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def parse_question(raw: str) -> Question:
    """解析视觉模型输出。模型没按 JSON 返回时，退化成把整段当作题干。"""
    payload = extract_json_object(raw)
    if payload is None:
        return Question(stem=raw.strip())

    payload["options"] = _coerce_options(payload.get("options"))
    known = {key: value for key, value in payload.items() if key in Question.model_fields}
    for key, value in known.items():
        if value is None:
            known[key] = [] if key == "options" else ""

    try:
        return Question.model_validate(known)
    except Exception:
        return Question(stem=raw.strip())
