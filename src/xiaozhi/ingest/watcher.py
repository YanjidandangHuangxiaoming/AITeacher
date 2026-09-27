"""目录监控：目录里出现新图片就自动送进讲题流水线。

看起来简单，但有三件事不做就会误触发：
  1. 只认图片扩展名——监控目录里可能混着 pdf、docx、数据库文件
  2. 等文件写完——微信/浏览器是分块落盘的，写一半就去读会得到坏图
  3. 去重——同一个文件在创建、改名、修改时会来好几个事件

监控线程与事件循环是两套东西，所以回调只往队列里塞路径，由消费方决定怎么处理。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from PIL import Image

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# 文件大小连续两次采样不变，就认为写完了
POLL_INTERVAL_S = 0.25
# 同一文件的去重窗口
DEDUP_WINDOW_S = 10.0


def _is_readable_image(path: Path) -> bool:
    """尝试解码，判断文件是否已经写完。截断的图片会在这里被挡掉。"""
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


class _ImageEventHandler(FileSystemEventHandler):
    def __init__(self, sink: Callable[[Path], None], extensions: set[str]) -> None:
        self._sink = sink
        self._extensions = extensions

    def on_created(self, event: FileSystemEvent) -> None:
        self._submit(event.src_path, event.is_directory)

    def on_moved(self, event: FileSystemEvent) -> None:
        # 不少程序先写临时文件再改名，所以 moved 的目标也要看
        self._submit(getattr(event, "dest_path", ""), event.is_directory)

    def _submit(self, raw_path: str, is_directory: bool) -> None:
        if is_directory or not raw_path:
            return
        path = Path(raw_path)
        if path.suffix.lower() in self._extensions:
            self._sink(path)


class ImageFolderWatcher:
    """监听若干目录（含子目录），把稳定下来的新图片交给 on_image 回调。"""

    def __init__(
        self,
        paths: Iterable[Path],
        on_image: Callable[[Path], None],
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
        settle_timeout_s: float = 5.0,
    ) -> None:
        self._paths = [Path(path) for path in paths]
        self._on_image = on_image
        self._extensions = {ext.lower() for ext in extensions}
        self._settle_timeout_s = settle_timeout_s

        self._candidates: queue.Queue[Path] = queue.Queue()
        self._handled: dict[str, float] = {}
        self._observer: Observer | None = None
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._consume, name="image-watcher", daemon=True
        )
        self._worker.start()

        observer = Observer()
        handler = _ImageEventHandler(self._candidates.put, self._extensions)
        for path in self._paths:
            path.mkdir(parents=True, exist_ok=True)
            observer.schedule(handler, str(path), recursive=True)
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        self._stopping.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        if self._worker is not None:
            self._worker.join(timeout=2)
            self._worker = None

    # ---------- 内部 ----------

    def _consume(self) -> None:
        while not self._stopping.is_set():
            try:
                candidate = self._candidates.get(timeout=0.25)
            except queue.Empty:
                continue

            settled = self._wait_until_settled(candidate)
            if settled is None or self._already_handled(settled):
                continue

            try:
                self._on_image(settled)
            except Exception:
                # 单个文件处理失败不能拖垮整个监听
                continue

    def _wait_until_settled(self, path: Path) -> Path | None:
        """等文件写完并且真的能解码成图片。

        只看「文件大小不变」是不够的——写入方可能在中间停顿很久。
        真正的判据是这张图能被正常打开，截断的 JPEG/PNG 会在这里被挡掉。
        """
        deadline = time.monotonic() + self._settle_timeout_s
        last_size = -1
        while time.monotonic() < deadline:
            if self._stopping.is_set():
                return None
            try:
                size = path.stat().st_size
            except OSError:
                return None  # 文件没了（临时文件被删）
            if size > 0 and size == last_size and _is_readable_image(path):
                return path
            last_size = size
            time.sleep(POLL_INTERVAL_S)
        return None

    def _already_handled(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return True

        key = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
        now = time.monotonic()
        for stale in [k for k, seen in self._handled.items() if now - seen > DEDUP_WINDOW_S]:
            self._handled.pop(stale, None)

        if key in self._handled:
            return True
        self._handled[key] = now
        return False
