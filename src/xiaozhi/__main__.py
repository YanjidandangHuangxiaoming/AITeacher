"""小智 AI 家教入口。

    python -m xiaozhi            纯文字模式（截图讲题）
    python -m xiaozhi voice      语音模式（唤醒词 + 打断）
    python -m xiaozhi devices    列出音频设备
"""

from __future__ import annotations

import sys

USAGE = """用法：
  python -m xiaozhi                  纯文字模式（截图 / 图片讲题）
  python -m xiaozhi voice            语音模式（唤醒词 + 语音打断）
  python -m xiaozhi enroll           录制并注册声纹（用于排除旁人说话）
  python -m xiaozhi report [天数]    生成家长周报（默认最近 7 天）
  python -m xiaozhi devices          列出可用音频设备
  python -m xiaozhi devices --probe  逐个试开设备，确认哪些真的能用
"""


def main() -> None:
    command = sys.argv[1].lower() if len(sys.argv) > 1 else ""

    if not command:
        from .cli.app import main as run_text

        run_text()
    elif command == "voice":
        from .cli.voice import main as run_voice

        run_voice()
    elif command == "enroll":
        from .cli.enroll import main as run_enroll

        run_enroll()
    elif command == "report":
        from .cli.report import main as run_report

        run_report(sys.argv[2:])
    elif command == "devices":
        from .cli.devices import main as run_devices

        run_devices(sys.argv[2:])
    elif command in {"-h", "--help", "help"}:
        print(USAGE)
    else:
        print(f"未知命令：{command}\n\n{USAGE}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
