"""目录监控的单元测试。

这里骗不了人：必须走真实文件系统和 watchdog 事件，
因为要验证的正是「分块落盘、重复事件、非图片文件」这些真实世界的行为。
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

from xiaozhi.ingest.watcher import ImageFolderWatcher


def _image_bytes(fmt: str = "PNG", size: tuple[int, int] = (160, 120)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, fmt)
    return buffer.getvalue()


def _write_image(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _wait_for(predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _watcher(tmp_path, found: list, **kwargs) -> ImageFolderWatcher:
    watcher = ImageFolderWatcher([tmp_path], on_image=found.append, **kwargs)
    watcher.start()
    return watcher


def test_detects_new_image(tmp_path) -> None:
    found: list[Path] = []
    watcher = _watcher(tmp_path, found)
    try:
        target = tmp_path / "question.png"
        _write_image(target, _image_bytes())
        assert _wait_for(lambda: found == [target])
    finally:
        watcher.stop()


def test_ignores_non_image_files(tmp_path) -> None:
    found: list[Path] = []
    watcher = _watcher(tmp_path, found)
    try:
        (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4" + b"x" * 400)
        (tmp_path / "plan.docx").write_bytes(b"x" * 400)
        (tmp_path / "message.db").write_bytes(b"x" * 400)
        time.sleep(1.5)
        assert found == []
    finally:
        watcher.stop()


def test_detects_image_in_subdirectory(tmp_path) -> None:
    """微信按月建子目录，所以必须递归监控。"""
    found: list[Path] = []
    watcher = _watcher(tmp_path, found)
    try:
        nested = tmp_path / "2026-09"
        nested.mkdir()
        target = nested / "question.jpg"
        _write_image(target, _image_bytes("JPEG"))
        assert _wait_for(lambda: found == [target])
    finally:
        watcher.stop()


def test_same_file_reported_once(tmp_path) -> None:
    found: list[Path] = []
    watcher = _watcher(tmp_path, found)
    try:
        target = tmp_path / "once.png"
        _write_image(target, _image_bytes())
        assert _wait_for(lambda: found)
        time.sleep(1.5)
        assert len(found) == 1
    finally:
        watcher.stop()


def test_partial_write_is_not_reported(tmp_path) -> None:
    """文件还在写的时候不能回调，否则会读到半张图。

    故意在中间停顿 0.8 秒——光看「文件大小不变」会误判成写完了，
    所以判据必须是「这张图能解码」。
    """
    payload = _image_bytes("PNG")
    found: list[Path] = []
    watcher = _watcher(tmp_path, found, settle_timeout_s=5.0)
    try:
        target = tmp_path / "slow.png"
        with target.open("wb") as handle:
            handle.write(payload[: len(payload) // 3])
            handle.flush()
            time.sleep(0.8)
            assert found == [], "都没写完就不该回调"
            handle.write(payload[len(payload) // 3 :])
        assert _wait_for(lambda: found == [target])
    finally:
        watcher.stop()


def test_truncated_image_is_never_reported(tmp_path) -> None:
    """永远写不完的坏图不能回调。"""
    payload = _image_bytes("PNG")
    found: list[Path] = []
    watcher = _watcher(tmp_path, found, settle_timeout_s=1.5)
    try:
        (tmp_path / "broken.png").write_bytes(payload[: len(payload) // 2])
        time.sleep(2.5)
        assert found == []
    finally:
        watcher.stop()


def test_callback_failure_does_not_stop_watching(tmp_path) -> None:
    seen: list[Path] = []

    def boom(path: Path) -> None:
        seen.append(path)
        raise RuntimeError("故意失败")

    watcher = ImageFolderWatcher([tmp_path], on_image=boom, settle_timeout_s=2.0)
    watcher.start()
    try:
        (tmp_path / "a.png").write_bytes(_image_bytes())
        assert _wait_for(lambda: len(seen) == 1)
        (tmp_path / "b.png").write_bytes(_image_bytes())
        assert _wait_for(lambda: len(seen) == 2), "一个文件出错后仍要继续监听"
    finally:
        watcher.stop()


def test_missing_directory_is_created(tmp_path) -> None:
    target = tmp_path / "not" / "yet"
    watcher = ImageFolderWatcher([target], on_image=lambda _: None)
    watcher.start()
    try:
        assert target.exists()
    finally:
        watcher.stop()


def test_stop_is_idempotent(tmp_path) -> None:
    found: list[Path] = []
    watcher = _watcher(tmp_path, found)
    watcher.stop()
    watcher.stop()
    (tmp_path / "after.png").write_bytes(_image_bytes())
    time.sleep(0.8)
    assert found == []
