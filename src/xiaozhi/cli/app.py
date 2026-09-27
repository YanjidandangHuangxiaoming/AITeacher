"""P0 命令行交互主循环。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from openai import APIError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..capture.screen import ScreenCapturer
from ..config import Settings, get_settings, load_prompt, missing_keys
from ..ingest.pipeline import IngestPipeline
from ..ingest.upload import UploadServer
from ..ingest.watcher import ImageFolderWatcher
from ..ingest.wechat import resolve_watch_dirs
from ..llm.client import DeepSeekClient
from ..llm.summarizer import QuestionSummarizer
from ..llm.tutor import Tutor
from ..store.db import Database
from ..store.repository import Repository
from ..vision.cloud_vl import QwenVlClient
from ..vision.question_parser import Question

console = Console()
MODEL_STYLE = "bold green"
USER_STYLE = "bold cyan"
# 待讲图片队列上限，防止被刷屏时无限堆积
MAX_PENDING_IMAGES = 5

HELP_TEXT = """[bold]可用命令[/bold]
  [cyan]/shot[/cyan] [等待秒数]    截取当前屏幕并开始讲题（默认 3 秒切屏时间）
  [cyan]/img[/cyan] <图片路径>     用本地已有的图片开始讲题
  [cyan]/review[/cyan] [数量]      查看错题本（默认 10 道）
  [cyan]/dirs[/cyan]              查看正在监控的图片目录
  [cyan]/new[/cyan]               结束当前题目，归档并清空上下文
  [cyan]/history[/cyan]           查看本轮对话记录
  [cyan]/help[/cyan]              显示这份帮助
  [cyan]/quit[/cyan]              退出

把图片丢进监控目录就会自动识别；也可以直接输入文字追问，比如"这一步没听懂"。
每道题结束时会自动归档掌握情况和知识点，周报用 python -m xiaozhi report 生成。"""


def _render_question(question: Question) -> str:
    lines = [f"学科：{question.subject}"]
    if question.grade:
        lines.append(f"年级：{question.grade}")
    lines.append(f"题干：{question.stem}")
    if question.options:
        lines.append("选项：" + "　".join(question.options))
    if question.figure:
        lines.append(f"图形：{question.figure}")
    if question.student_answer:
        lines.append(f"学生作答：{question.student_answer}")
    if question.notes:
        lines.append(f"备注：{question.notes}")
    return "\n".join(lines)


class TutorApp:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline = IngestPipeline(QwenVlClient(settings.vision))
        self._capturer = ScreenCapturer(settings.capture, settings.output_dir())

        llm = DeepSeekClient(settings.llm)
        self._tutor = Tutor(settings.tutor, load_prompt("tutor_system.md"), llm)
        self._summarizer = QuestionSummarizer(llm)

        self._database = Database(settings.store_path())
        self._repo = Repository(self._database)
        # 当前这道题在库里的 id，以及已经落库的消息条数
        self._open_question_id: int | None = None
        self._recorded_messages = 0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._watcher: ImageFolderWatcher | None = None
        self._uploader: UploadServer | None = None
        # 监控目录收到图时先排队，等用户按回车再讲——
        # 直接在流式输出中间插话会把终端界面搅乱。
        # 用列表而不是单个变量：连发几张时不能只留下最后一张。
        self._pending_images: list[Path] = []

    # ---------- 启动 ----------

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._print_banner()
        self._start_watcher()
        self._start_uploader()
        try:
            await self._main_loop()
        finally:
            self._stop_watcher()
            self._stop_uploader()
            await self._close_current_question()
            self._database.close()

    async def _main_loop(self) -> None:
        while True:
            try:
                raw = await asyncio.to_thread(console.input, f"\n[{USER_STYLE}]你 > [/]")
            except (EOFError, KeyboardInterrupt):
                self._farewell()
                return

            line = raw.strip()
            if not line:
                if self._pending_images:
                    path = self._pending_images.pop(0)
                    await self._start_from_image(path)
                    if self._pending_images:
                        console.print(
                            f"[dim]还有 {len(self._pending_images)} 张待讲，"
                            "按回车继续。[/dim]"
                        )
                continue
            if line.startswith("/"):
                if not await self._handle_command(line):
                    self._farewell()
                    return
                continue
            if self._tutor.question is None:
                console.print("[yellow]还没有题目。先用 /shot 截张卷子，或用 /img <路径> 指定一张。[/yellow]")
                continue
            await self._stream(self._tutor.stream_reply(line))

    def _print_banner(self) -> None:
        settings = self._settings
        console.print(
            Panel.fit(
                "[bold]小智 AI 家教[/bold] · P0 纯文字版\n"
                f"年级 {settings.tutor.grade}　模式 分步引导\n"
                "输入 [bold]/help[/bold] 查看命令",
                border_style="green",
            )
        )

    def _farewell(self) -> None:
        console.print("\n再见，同学。")

    # ---------- 目录监控 ----------

    def _start_watcher(self) -> None:
        paths = resolve_watch_dirs(self._settings)
        self._watcher = ImageFolderWatcher(
            paths,
            on_image=self._on_image_found,
            extensions=self._settings.ingest.extensions,
            settle_timeout_s=self._settings.ingest.settle_timeout_s,
        )
        self._watcher.start()

        listed = "\n".join(f"    {path}" for path in paths)
        console.print(f"[dim]正在监控这些目录，图片放进去就会自动识别：\n{listed}[/dim]")

    def _stop_watcher(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def _start_uploader(self) -> None:
        upload = self._settings.ingest.upload
        if not upload.enabled:
            return
        try:
            self._uploader = UploadServer(
                self._settings.inbox_dir(),
                port=upload.port,
                extensions=set(self._settings.ingest.extensions),
                max_bytes=int(upload.max_mb * 1024 * 1024),
            )
            self._uploader.start()
        except Exception as exc:
            self._uploader = None
            console.print(f"[yellow]局域网上传服务没起来：{exc}[/yellow]")
            return

        console.print(
            f"[green]手机传题：[/green]让手机连同一个 WiFi，浏览器打开 "
            f"[bold]{self._uploader.url}[/bold]"
        )

    def _stop_uploader(self) -> None:
        if self._uploader is not None:
            self._uploader.stop()
            self._uploader = None

    def _on_image_found(self, path: Path) -> None:
        """在监控线程里被调用，必须转回事件循环再动界面状态。"""
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._mark_pending_image, path)

    def _mark_pending_image(self, path: Path) -> None:
        if path not in self._pending_images:
            self._pending_images.append(path)
        # 上限保护：真被刷屏也不会无限堆积
        if len(self._pending_images) > MAX_PENDING_IMAGES:
            dropped = self._pending_images[:-MAX_PENDING_IMAGES]
            self._pending_images = self._pending_images[-MAX_PENDING_IMAGES:]
            for item in dropped:
                console.print(f"[dim]待讲队列已满，丢弃 {item.name}[/dim]")

        more = f"（还有 {len(self._pending_images) - 1} 张）" if len(self._pending_images) > 1 else ""
        console.print(
            f"\n[green]收到新图片[/green] {escape(path.name)}{more}"
            "　[dim]按回车开始讲题[/dim]"
        )

    def _print_watch_dirs(self) -> None:
        paths = (
            self._watcher.paths
            if self._watcher is not None
            else resolve_watch_dirs(self._settings)
        )
        lines = "\n".join(
            f"  {path}　[dim]{'存在' if path.exists() else '不存在'}[/dim]"
            for path in paths
        )
        console.print(f"[bold]正在监控的图片目录[/bold]\n{lines}")
        if self._uploader is not None:
            console.print(f"\n[bold]手机上传地址[/bold]　{self._uploader.url}")

    # ---------- 命令分发 ----------

    async def _handle_command(self, line: str) -> bool:
        parts = line[1:].split()
        name = parts[0].lower() if parts else ""
        args = parts[1:]

        if name in {"quit", "exit", "q"}:
            return False
        if name == "help":
            console.print(HELP_TEXT)
        elif name == "shot":
            await self._shot(args)
        elif name == "img":
            await self._img(args)
        elif name == "new":
            await self._close_current_question()
            self._tutor.reset()
            console.print("[green]已结束当前题目，归档并清空上下文。[/green]")
        elif name == "review":
            self._print_review(args)
        elif name == "dirs":
            self._print_watch_dirs()
        elif name == "history":
            self._print_history()
        else:
            console.print(f"[yellow]未知命令 /{name}，输入 /help 查看全部命令。[/yellow]")
        return True

    async def _shot(self, args: list[str]) -> None:
        delay: float | None = None
        if args:
            try:
                delay = float(args[0])
            except ValueError:
                console.print("[yellow]用法：/shot [等待秒数]，例如 /shot 5[/yellow]")
                return

        wait = self._settings.capture.delay_s if delay is None else delay
        if wait > 0:
            console.print(f"[dim]请在 {wait:g} 秒内把卷子切到屏幕上…[/dim]")
        try:
            path = await asyncio.to_thread(self._capturer.capture, wait)
        except Exception as exc:
            console.print(f"[red]截图失败：{exc}[/red]")
            return

        console.print(f"[dim]已保存截图 {path.name}[/dim]")
        await self._start_from_image(path, source="shot")

    async def _img(self, args: list[str]) -> None:
        if not args:
            console.print("[yellow]用法：/img <图片路径>[/yellow]")
            return

        path = Path(args[0].strip('"')).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            console.print(f"[red]找不到图片：{path}[/red]")
            return
        await self._start_from_image(path, source="img")

    # ---------- 讲题 ----------

    async def _start_from_image(self, path: Path, source: str = "unknown") -> None:
        try:
            with console.status("[cyan]正在识别题目…[/cyan]"):
                question = await self._pipeline.to_question(path)
        except APIError as exc:
            console.print(f"[red]识题请求失败：{exc}[/red]")
            return
        except Exception as exc:
            console.print(f"[red]识题出错：{exc}[/red]")
            return

        if question.is_empty():
            console.print("[yellow]没能从这张图里读出题干，换一张更清晰、更正的照片再试。[/yellow]")
            if question.notes:
                console.print(f"[dim]{question.notes}[/dim]")
            return

        # 上一道题先归档，再开新的
        await self._close_current_question()

        console.print(
            Panel(escape(_render_question(question)), title="识别到的题目", border_style="cyan")
        )
        self._tutor.start_question(question)
        self._open_question_id = self._repo.create_question(
            question, source=source, image_path=str(path)
        )
        self._recorded_messages = 0
        await self._stream(self._tutor.stream_reply(self._tutor.opening_prompt()))

    # ---------- 学习记录 ----------

    def _sync_messages(self) -> None:
        """把还没落库的对话增量写进去（按游标，不重复写）。"""
        if self._open_question_id is None:
            return
        messages = self._tutor.messages
        for message in messages[self._recorded_messages :]:
            self._repo.append_message(
                self._open_question_id, message["role"], message["content"]
            )
        self._recorded_messages = len(messages)

    async def _close_current_question(self) -> None:
        """结题归档：判断掌握情况和知识点，然后落库。"""
        question_id = self._open_question_id
        if question_id is None:
            return
        self._open_question_id = None
        self._sync_messages()

        messages = self._tutor.messages
        if not messages:
            self._repo.close_question(question_id, turns=0)
            return

        with console.status("[cyan]正在归档这道题…[/cyan]"):
            summary = await self._summarizer.summarize(messages)
        self._repo.close_question(question_id, turns=len(messages) // 2, summary=summary)

        if summary.mastery:
            topics = "、".join(summary.topics) or "未标知识点"
            console.print(
                f"[dim]已归档：{summary.mastery}　{topics}[/dim]"
            )

    def _print_review(self, args: list[str]) -> None:
        limit = 10
        if args:
            try:
                limit = max(1, int(args[0]))
            except ValueError:
                console.print("[yellow]用法：/review [数量]，例如 /review 20[/yellow]")
                return

        records = self._repo.weak_questions(limit)
        if not records:
            console.print(
                '[dim]错题本是空的。判定为"部分掌握"或"未掌握"的题才会进来，'
                "结题归档后才有记录。[/dim]"
            )
            return

        console.print(f"[bold]错题本（最近 {len(records)} 道）[/bold]")
        for record in records:
            color = "red" if record.mastery == "未掌握" else "yellow"
            topics = "、".join(record.topics) or "未标知识点"
            console.print(
                f"\n[bold]{record.created_at:%m-%d %H:%M}[/bold]　"
                f"{escape(record.subject)}　[{color}]{record.mastery}[/]"
            )
            console.print(f"  {escape(record.brief(60))}")
            if record.summary:
                console.print(f"  [dim]{escape(record.summary)}[/dim]")
            console.print(f"  [dim]知识点：{escape(topics)}[/dim]")

    async def _stream(self, stream: AsyncIterator[str]) -> None:
        print()
        print("小智：", end="", flush=True)
        try:
            async for piece in stream:
                print(piece, end="", flush=True)
        except APIError as exc:
            console.print(f"[red]对话请求失败：{exc}[/red]")
        except Exception as exc:
            console.print(f"[red]对话出错：{exc}[/red]")
        finally:
            print()
            # 无论成功失败，都把已经产生的对话落库
            self._sync_messages()

    def _print_history(self) -> None:
        messages = self._tutor.messages
        if not messages:
            console.print("[dim]当前还没有对话记录。[/dim]")
            return
        for message in messages:
            tag = USER_STYLE if message["role"] == "user" else MODEL_STYLE
            speaker = "你" if message["role"] == "user" else "小智"
            console.print(f"[{tag}]{speaker}[/]：", end="")
            console.print(message["content"], markup=False, highlight=False)


def main() -> None:
    try:
        settings = get_settings()
    except Exception as exc:
        console.print(f"[red]读取配置失败：{exc}[/red]")
        raise SystemExit(1) from exc

    # 先校验密钥，再构造各客户端，避免 SDK 抛出难懂的异常
    missing = missing_keys(settings)
    if missing:
        console.print("[red]缺少 API Key，无法启动。[/red]")
        console.print(f"请在项目根目录创建 [bold].env[/bold]，填入：{', '.join(missing)}")
        console.print("可以复制 [bold].env.example[/bold] 作为模板。")
        raise SystemExit(1)

    app = TutorApp(settings)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("\n再见，同学。")
