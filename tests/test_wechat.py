"""微信目录自动定位的测试。

用伪造的目录树，不碰真实微信数据——真实数据里全是私人聊天文件。
"""

from __future__ import annotations

from pathlib import Path

from xiaozhi.config import Settings
from xiaozhi.ingest.wechat import (
    discover_file_dirs,
    resolve_watch_dirs,
    wechat_root_candidates,
)


def _fake_v4_root(tmp_path: Path) -> Path:
    """仿造微信 4.x 的目录结构。"""
    root = tmp_path / "xwechat_files"
    (root / "wxid_aaa_1111" / "msg" / "file" / "2026-09").mkdir(parents=True)
    (root / "wxid_bbb_2222" / "msg" / "file" / "2026-08").mkdir(parents=True)
    (root / "all_users" / "config").mkdir(parents=True)  # 这个账号没有 msg\file
    return root


def test_discovers_v4_file_dirs(tmp_path) -> None:
    root = _fake_v4_root(tmp_path)
    found = discover_file_dirs([root])
    assert len(found) == 2
    assert all(path.name == "file" and path.parent.name == "msg" for path in found)


def test_discovers_v3_file_storage(tmp_path) -> None:
    root = tmp_path / "WeChat Files"
    (root / "wxid_ccc" / "FileStorage" / "File").mkdir(parents=True)
    found = discover_file_dirs([root])
    assert len(found) == 1
    assert found[0].parent.name == "FileStorage"


def test_ignores_accounts_without_file_dir(tmp_path) -> None:
    root = tmp_path / "xwechat_files"
    (root / "all_users" / "config").mkdir(parents=True)
    assert discover_file_dirs([root]) == []


def test_missing_root_is_skipped() -> None:
    assert discover_file_dirs([Path("Z:/definitely/not/here")]) == []


def test_root_candidates_cover_both_names() -> None:
    names = [str(path) for path in wechat_root_candidates()]
    assert any("xwechat_files" in name for name in names)
    assert any("WeChat Files" in name for name in names)


def test_resolve_watch_dirs_always_includes_inbox(tmp_path) -> None:
    settings = Settings(ingest={"watch_wechat": False})
    paths = resolve_watch_dirs(settings)
    assert settings.inbox_dir() in paths


def test_resolve_watch_dirs_adds_configured_dirs(tmp_path) -> None:
    extra = tmp_path / "extra"
    settings = Settings(
        ingest={"watch_wechat": False, "watch_dirs": [str(extra)]}
    )
    paths = resolve_watch_dirs(settings)
    assert extra in paths
    assert len(paths) == 2


def test_resolve_watch_dirs_does_not_duplicate(tmp_path) -> None:
    settings = Settings(
        ingest={"watch_wechat": False, "watch_dirs": [str(tmp_path), str(tmp_path)]}
    )
    paths = resolve_watch_dirs(settings)
    assert len(paths) == len(set(paths))
