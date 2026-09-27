"""局域网上传服务的测试。

真的起 HTTP 服务、真的发请求，不 mock——这里的重点正是「网络边界上的校验」。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from xiaozhi.ingest.upload import UploadServer


def _image_bytes(fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (160, 120), "white").save(buffer, fmt)
    return buffer.getvalue()


def _server(tmp_path: Path, **kwargs) -> UploadServer:
    server = UploadServer(tmp_path, port=0, **kwargs)
    server.start()
    return server


def _post(server: UploadServer, name: str, payload: bytes) -> tuple[int, dict]:
    # 文件名要按 URL 规则编码——页面那边用的是 encodeURIComponent，测试要一致
    url = f"http://127.0.0.1:{server.port}/upload?name={urllib.parse.quote(name)}"
    return _request(url, payload)


def _request(url: str, payload: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_saves_uploaded_image(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        status, body = _post(server, "卷子.png", _image_bytes())
        assert status == 200
        assert body["ok"] is True

        saved = list(tmp_path.glob("*.png"))
        assert len(saved) == 1
        Image.open(saved[0]).verify()
        # 文件名里要保留原名的可读部分，方便辨认
        assert "卷子" in saved[0].name
    finally:
        server.stop()


def test_serves_upload_page(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/", timeout=5
        ) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
    finally:
        server.stop()

    assert "上传题目" in html
    assert 'accept="image/*"' in html
    # 手机上要能直接调起相机
    assert "capture" in html


def test_rejects_non_image_extension(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        status, body = _post(server, "notes.pdf", b"%PDF-1.4" + b"x" * 100)
        assert status == 400
        assert body["ok"] is False
        assert list(tmp_path.iterdir()) == []
    finally:
        server.stop()


def test_rejects_unreadable_image(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        status, body = _post(server, "broken.png", _image_bytes()[:20])
        assert status == 400
        assert body["ok"] is False
        assert list(tmp_path.iterdir()) == []
    finally:
        server.stop()


def test_rejects_oversized_upload(tmp_path) -> None:
    payload = _image_bytes()
    server = _server(tmp_path, max_bytes=32)
    try:
        assert len(payload) > 32, "测试前提：这张图要比上限大"
        status, body = _post(server, "big.png", payload)
        assert status == 413
        assert body["ok"] is False
        assert list(tmp_path.iterdir()) == []
    finally:
        server.stop()


def test_unknown_path_returns_404(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        status, body = _request(
            f"http://127.0.0.1:{server.port}/nope?name=a.png", b"x"
        )
        assert status == 404
        assert body["ok"] is False
    finally:
        server.stop()


def test_url_points_at_lan_address(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        assert server.url.startswith("http://")
        assert server.url.endswith(str(server.port))
        assert server.port != 0, "port=0 时也要把真实端口暴露出来"
    finally:
        server.stop()


def test_stop_is_idempotent(tmp_path) -> None:
    server = _server(tmp_path)
    server.stop()
    server.stop()


def test_uses_ephemeral_port_when_requested(tmp_path) -> None:
    server = _server(tmp_path)
    try:
        assert 1024 < server.port < 65536
    finally:
        server.stop()
