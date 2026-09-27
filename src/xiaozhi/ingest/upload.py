"""局域网网页上传：手机浏览器打开一个地址，直接拍照上传。

这是最可靠的手机→电脑通道：不依赖任何第三方 App，不会连带看到别人发来的文件，
iOS 和 Android 都能用。上传的图片落到 data/inbox，由目录监控接走。

安全边界：只监听局域网、只收图片、限制大小、校验能否解码。
不做身份认证——家用 WiFi 内网场景下够用，但要知道同网段的设备都能上传。
"""

from __future__ import annotations

import socket
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

DEFAULT_PORT = 8000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>小智 AI 家教 · 上传题目</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 20px;
    font: 16px/1.6 "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
    background: #f7f7f8; color: #171717;
  }
  h1 { margin: 0 0 8px; font-size: 20px; }
  p { margin: 0 0 20px; color: #52525b; font-size: 14px; }
  label {
    display: block; padding: 40px 20px; margin-bottom: 16px;
    text-align: center; color: #52525b; font-size: 14px;
    background: #fff; border: 2px dashed rgba(23,23,23,.18); border-radius: 12px;
  }
  input[type=file] { display: block; width: 100%; margin-bottom: 16px; font-size: 14px; }
  button {
    width: 100%; padding: 14px; font-size: 16px; font-weight: 500;
    color: #004d26; background: #0fdc78; border: 0; border-radius: 8px;
  }
  button:disabled { opacity: .5; }
  #status { margin: 16px 0 0; font-size: 14px; min-height: 22px; }
  .ok { color: #02743b; }
  .err { color: #e8463a; }
  img { max-width: 100%; margin-top: 12px; border-radius: 8px; display: none; }
  @media (prefers-color-scheme: dark) {
    body { background: #171717; color: #e5e5e5; }
    p, label { color: #a1a1aa; }
    label { background: #262626; border-color: rgba(229,229,229,.18); }
    button { color: #004d26; }
  }
</style>
</head>
<body>
  <h1>上传题目</h1>
  <p>拍一张卷子，或者从相册里选一张。上传后电脑上按回车就开始讲。</p>
  <label>点这里选照片 / 拍照
    <input id="file" type="file" accept="image/*" capture="environment" style="margin-top:12px">
  </label>
  <button id="send" disabled>上传</button>
  <p id="status"></p>
  <img id="preview" alt="预览">
<script>
  const input = document.getElementById('file');
  const button = document.getElementById('send');
  const status = document.getElementById('status');
  const preview = document.getElementById('preview');

  input.addEventListener('change', () => {
    const file = input.files[0];
    button.disabled = !file;
    status.textContent = '';
    status.className = '';
    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    } else {
      preview.style.display = 'none';
    }
  });

  button.addEventListener('click', async () => {
    const file = input.files[0];
    if (!file) return;
    button.disabled = true;
    status.textContent = '正在上传…';
    status.className = '';
    try {
      const res = await fetch('/upload?name=' + encodeURIComponent(file.name), {
        method: 'POST',
        body: file,
      });
      const data = await res.json();
      if (res.ok && data.ok) {
        status.textContent = '上传成功，去电脑上按回车开始讲题。';
        status.className = 'ok';
        input.value = '';
        preview.style.display = 'none';
      } else {
        status.textContent = data.error || '上传失败';
        status.className = 'err';
      }
    } catch (err) {
      status.textContent = '上传失败：' + err;
      status.className = 'err';
    } finally {
      button.disabled = false;
    }
  });
</script>
</body>
</html>
"""


class UploadServer:
    """在后台线程里跑一个只收图片的 HTTP 服务。"""

    def __init__(
        self,
        inbox: Path,
        port: int = DEFAULT_PORT,
        extensions: set[str] | None = None,
        max_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self._inbox = inbox
        self._port = port
        self._extensions = extensions or {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self._max_bytes = max_bytes
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bound_port = port

    @property
    def url(self) -> str:
        return f"http://{lan_ip()}:{self._bound_port}"

    @property
    def port(self) -> int:
        return self._bound_port

    def start(self) -> None:
        self._inbox.mkdir(parents=True, exist_ok=True)
        server = _UploadHTTPServer(
            ("0.0.0.0", self._port),
            inbox=self._inbox,
            extensions=self._extensions,
            max_bytes=self._max_bytes,
        )
        # 传 0 时由系统分配端口，这里要把真实端口取回来
        self._bound_port = server.server_address[1]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="upload-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class _UploadHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, inbox: Path, extensions: set[str], max_bytes: int) -> None:
        super().__init__(address, _UploadHandler)
        self.inbox = inbox
        self.extensions = extensions
        self.max_bytes = max_bytes


class _UploadHandler(BaseHTTPRequestHandler):
    server_version = "XiaozhiTutor"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler 的既定命名)
        if urlparse(self.path).path in ("/", "/index.html"):
            self._respond(200, "text/html; charset=utf-8", UPLOAD_PAGE.encode("utf-8"))
        else:
            self._respond(404, "text/plain; charset=utf-8", "没这个页面".encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self._json(404, {"ok": False, "error": "未知路径"})
            return

        raw_name = (parse_qs(parsed.query).get("name") or [""])[0]
        suffix = Path(raw_name).suffix.lower()

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"ok": False, "error": "没有收到内容"})
            return
        if length > self.server.max_bytes:
            # 超出上限时不再读 body，必须关掉连接，否则残留数据会污染后续请求
            self.close_connection = True
            limit = self.server.max_bytes // (1024 * 1024)
            self._json(413, {"ok": False, "error": f"图片超过 {limit} MB"})
            return

        # 先把 body 读干净，再校验，保持连接整洁
        payload = self.rfile.read(length)

        if suffix not in self.server.extensions:
            self._json(400, {"ok": False, "error": f"只收图片，收到的是 {suffix or '无扩展名'}"})
            return

        try:
            Image.open(BytesIO(payload)).verify()
        except Exception:
            self._json(400, {"ok": False, "error": "这不是一张能打开的图片"})
            return

        target = self._target_path(raw_name, suffix)
        target.write_bytes(payload)
        self._json(200, {"ok": True, "saved": target.name})

    def _target_path(self, raw_name: str, suffix: str) -> Path:
        stem = Path(raw_name).stem.strip() or "photo"
        safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_")[:24] or "photo"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.server.inbox / f"{stamp}_{safe_stem}{suffix}"

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        import json

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._respond(code, "application/json; charset=utf-8", body)

    def log_message(self, format: str, *args) -> None:
        """默认会往 stderr 刷日志，这里静音，避免搅乱命令行界面。"""


def lan_ip() -> str:
    """取本机在局域网里的地址。

    连一个外部 UDP 地址只是为了让系统选出实际出口网卡，不会真的发包。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()
