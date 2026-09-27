"""对话状态机的状态定义。"""

from __future__ import annotations

from enum import StrEnum


class DialogueState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    DialogueState.IDLE: "待唤醒",
    DialogueState.LISTENING: "聆听中",
    DialogueState.THINKING: "思考中",
    DialogueState.SPEAKING: "播报中",
}
