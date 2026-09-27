"""生成家长周报。

    python -m xiaozhi report        最近 7 天
    python -m xiaozhi report 14     最近 14 天

统计部分是本地算的，不联网也能出；只有"本周小结"那段叙述需要 DeepSeek，
没有可用 Key 时会自动跳过。
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..config import Settings, get_settings, missing_keys
from ..llm.client import DeepSeekClient
from ..report.narrator import WeeklyNarrator
from ..report.weekly import WeeklyReport, build_report, render_plain
from ..store.db import Database
from ..store.repository import Repository

console = Console()
DEFAULT_DAYS = 7


def main(argv: list[str] | None = None) -> None:
    days = _parse_days(argv or [])

    try:
        settings = get_settings()
    except Exception as exc:
        console.print(f"[red]读取配置失败：{exc}[/red]")
        raise SystemExit(1) from exc

    database = Database(settings.store_path())
    try:
        report = build_report(Repository(database), days)
        if not report.total:
            console.print(
                f"[yellow]最近 {days} 天还没有讲题记录。[/yellow]\n"
                "[dim]用 python -m xiaozhi 讲几道题，结题时会自动归档。[/dim]"
            )
            return
        narrative = asyncio.run(_narrate(settings, report))
    finally:
        database.close()

    console.print(
        Panel(
            escape(render_plain(report, narrative)),
            title="家长周报",
            border_style="green",
        )
    )
    if not narrative:
        console.print(
            "[dim]（没生成「本周小结」——需要配置 DeepSeek Key）[/dim]"
        )


def _parse_days(args: list[str]) -> int:
    if not args:
        return DEFAULT_DAYS
    try:
        return max(1, int(args[0]))
    except ValueError:
        console.print(
            f"[yellow]用法：python -m xiaozhi report [天数]，例如 report 14[/yellow]"
        )
        raise SystemExit(1) from None


async def _narrate(settings: Settings, report: WeeklyReport) -> str:
    if missing_keys(settings):
        return ""
    return await WeeklyNarrator(DeepSeekClient(settings.llm)).narrate(report)
