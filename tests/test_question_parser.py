"""识题结果解析的单元测试（纯逻辑，不联网）。"""

from __future__ import annotations

from xiaozhi.vision.question_parser import parse_question


def test_parses_plain_json() -> None:
    question = parse_question(
        '{"subject": "数学", "stem": "解方程 x+1=2", "options": []}'
    )
    assert question.subject == "数学"
    assert question.stem == "解方程 x+1=2"
    assert question.options == []


def test_strips_markdown_code_fence() -> None:
    question = parse_question(
        '```json\n{"subject": "物理", "stem": "求小车加速度"}\n```'
    )
    assert question.subject == "物理"
    assert question.stem == "求小车加速度"


def test_ignores_prose_around_json() -> None:
    question = parse_question(
        '好的，我识别到：{"subject": "英语", "stem": "Fill in the blank."} 完毕'
    )
    assert question.stem == "Fill in the blank."


def test_null_options_becomes_empty_list() -> None:
    question = parse_question('{"stem": "题目", "options": null}')
    assert question.options == []


def test_string_options_becomes_list() -> None:
    question = parse_question('{"stem": "题目", "options": "A. 1"}')
    assert question.options == ["A. 1"]


def test_falls_back_to_raw_text_when_not_json() -> None:
    question = parse_question("这张图太模糊了，读不出来。")
    assert question.stem == "这张图太模糊了，读不出来。"
    assert question.is_empty() is False


def test_unknown_fields_are_dropped() -> None:
    question = parse_question('{"stem": "题目", "difficulty": "hard"}')
    assert not hasattr(question, "difficulty")


def test_to_prompt_contains_all_sections() -> None:
    question = parse_question(
        '{"subject": "数学", "grade": "初二", "stem": "题干内容", '
        '"options": ["A. 1"], "figure": "一个直角三角形", '
        '"student_answer": "选 A", "notes": "右下角有折痕"}'
    )
    prompt = question.to_prompt()
    for fragment in ["【学科】数学", "【年级】初二", "【题干】题干内容",
                     "【选项】A. 1", "【图形信息】一个直角三角形",
                     "【学生已作答】选 A", "【识别备注】右下角有折痕"]:
        assert fragment in prompt
