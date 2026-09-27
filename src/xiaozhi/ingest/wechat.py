r"""自动定位微信「文件」通道的存盘目录。

微信 4.x 把**收到的文件**原样放在 `<数据目录>\<账号>\msg\file\<年-月>\` 下，
是可读的 jpg/png/pdf；而**收到的图片**会被存成 `msg\attach` 下的加密 .dat，
读不出来。所以想让电脑收到可读的照片，手机上必须用「文件」方式发送。

数据目录位置由用户安装时自己选（本机就放在 E 盘），所以这里靠注册表拿到
Documents 的真实位置，再穷举几个常见落点。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from ..config import Settings


def wechat_root_candidates() -> list[Path]:
    """列出可能的微信数据根目录。"""
    roots: list[Path] = []

    documents = documents_dir()
    if documents is not None:
        roots.append(documents / "xwechat_files")
        roots.append(documents / "WeChat Files")  # 3.x 旧版

    home = Path.home()
    roots.append(home / "Documents" / "xwechat_files")
    roots.append(home / "Documents" / "WeChat Files")

    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def discover_file_dirs(roots: Iterable[Path] | None = None) -> list[Path]:
    r"""找出所有可用的微信文件目录（4.x 的 msg\file 和 3.x 的 FileStorage\File）。"""
    found: list[Path] = []
    for root in roots if roots is not None else wechat_root_candidates():
        if not root.is_dir():
            continue
        for account in _safe_iterdir(root):
            for candidate in (
                account / "msg" / "file",  # 4.x
                account / "FileStorage" / "File",  # 3.x
            ):
                if candidate.is_dir() and candidate not in found:
                    found.append(candidate)
    return found


def resolve_watch_dirs(settings: Settings) -> list[Path]:
    r"""合并「固定监控目录」与「自动发现的微信目录」。

    注意微信的 msg\file 里混着**所有聊天**收到的文件，同学发来的图片也会进来。
    这是选择微信通道的代价，启动时会明确提示。
    """
    paths = list(settings.watch_dirs())
    if settings.ingest.watch_wechat:
        for found in discover_file_dirs():
            if found not in paths:
                paths.append(found)
    return paths


def documents_dir() -> Path | None:
    """取 Documents 的真实路径。

    光看 %USERPROFILE%\\Documents 是不够的——用户可以把 Documents 挪到别的盘
    （本机就在 E 盘），真实位置记在注册表里。
    """
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            raw, _ = winreg.QueryValueEx(key, "Personal")
    except OSError:
        return None

    expanded = os.path.expandvars(raw)
    return Path(expanded) if expanded else None


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return [item for item in path.iterdir() if item.is_dir()]
    except OSError:
        return []
