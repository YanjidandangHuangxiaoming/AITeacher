"""启动菜单。

做成 Python 而不是直接写在 .cmd 里：cmd.exe 解析含中文的批处理会字节偏移错位，
把命令行从中间劈开（`chcp 65001` 之后偏移算错）。所以批处理里只留 ASCII，
中文界面放在这里——Python 的源码和输出都是 UTF-8 原生支持，没这个问题。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (按键, 名称, 说明, 传给 python 的参数)
MENU: list[tuple[str, str, str, list[str]]] = [
    ("1", "文字模式", "截图 / 手机上传 / 微信传题", ["-m", "xiaozhi"]),
    ("2", "语音模式", "唤醒词 + 语音打断（先连耳机）", ["-m", "xiaozhi", "voice"]),
    ("3", "家长周报", "最近 7 天的学习情况", ["-m", "xiaozhi", "report"]),
    ("4", "注册声纹", "排除旁人说话造成的误打断", ["-m", "xiaozhi", "enroll"]),
    ("5", "环境自检", "逐项检查配置和各条链路", ["scripts/check_env.py"]),
    ("6", "设备诊断", "确认麦克风和扬声器能用", ["-m", "xiaozhi", "devices", "--probe"]),
]


def _draw() -> None:
    print()
    print("  ================================")
    print("     小智 AI 家教")
    print("  ================================")
    print()
    for key, label, hint, _ in MENU:
        print(f"    {key}. {label}　　{hint}")
    print("    0. 退出")
    print()


def _ask(prompt: str) -> str | None:
    """读一行输入；stdin 关掉时返回 None，交给调用方收尾。"""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def main() -> int:
    if not sys.executable:
        print("[错误] 拿不到 Python 解释器路径。")
        return 1

    while True:
        _draw()
        choice = _ask("  请选择：")
        if choice is None:
            return 0
        if choice in {"0", "q", "Q"}:
            return 0

        action = next((item for item in MENU if item[0] == choice), None)
        if action is None:
            continue

        label, args = action[1], action[3]
        print()
        try:
            # 用同一个解释器跑子命令，避免误用系统 Python
            subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=False)
        except KeyboardInterrupt:
            print("\n  [已中断]")

        print(f"\n  ---- {label} 已结束 ----")
        if _ask("  按回车回到菜单…") is None:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
