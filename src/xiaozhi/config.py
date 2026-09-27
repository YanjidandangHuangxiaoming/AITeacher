"""配置加载：读取 config/default.yaml，展开 ${ENV} 占位符并校验。"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AudioConfig(BaseModel):
    input_device: str | None = None
    output_device: str | None = None
    sample_rate: int = 16000
    frame_ms: int = 30
    queue_frames: int = 64


class WakeConfig(BaseModel):
    enabled: bool = True
    word: str = "小智小智"
    # ASR 常把"小智"写成同音字，这些变体同样算命中（否则十次唤醒有八次不灵）
    variants: list[str] = Field(
        default_factory=lambda: ["小志", "小知", "小致", "小至", "小制"]
    )


class VadConfig(BaseModel):
    # webrtcvad 激进度：0 最宽松、3 最严格
    aggressiveness: int = 3
    # 进入语音前保留的前导帧时长，防止吃掉第一个字
    preroll_ms: int = 300
    min_speech_ms: int = 250
    silence_ms: int = 800
    max_utterance_s: int = 30
    idle_timeout_s: int = 15


class InterruptConfig(BaseModel):
    enabled: bool = True
    # voiceprint = 声纹确认是本人后才真打断；vad = 检测到人声立即打断；off = 关闭
    mode: str = "voiceprint"
    require_headset: bool = True
    # 声纹模式下，先收满这么多语音再做身份判定
    verify_ms: int = 600


class VoiceprintConfig(BaseModel):
    enabled: bool = True
    # 声纹文件前缀，实际会写成 .npy + .json 两个文件
    path: str = "data/voiceprint"
    # 余弦相似度阈值。留 null = 用注册时自动标定的值（推荐）
    threshold: float | None = None
    # 注册时要求的录音时长
    enroll_seconds: float = 15.0
    # 短于该时长的语音不做判定，直接放行（太短不可靠，宁可放行也别拦错人）
    min_seconds: float = 0.5


class AsrConfig(BaseModel):
    provider: str = "aliyun"
    model: str = "paraformer-realtime-v2"
    api_key: str = ""


class LlmConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key: str = ""
    temperature: float = 0.5
    timeout_s: float = 60.0


class VisionConfig(BaseModel):
    provider: str = "aliyun"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-vl-max"
    api_key: str = ""
    timeout_s: float = 120.0


class TtsConfig(BaseModel):
    provider: str = "aliyun"
    model: str = "cosyvoice-v2"
    voice: str = "longxiaochun_v2"
    speed: float = 1.0
    api_key: str = ""
    timeout_s: float = 30.0


class CaptureConfig(BaseModel):
    delay_s: float = 3.0
    region: list[int] | None = None
    # 截图单独放一个目录，不能和 ingest 的投递目录混用，
    # 否则 /shot 存下的图会被目录监控当成新图片再处理一次
    output_dir: str = "data/shots"
    hotkey: str = "ctrl+alt+q"


class UploadConfig(BaseModel):
    # 局域网网页上传：手机浏览器打开一个地址就能拍照传题
    enabled: bool = True
    port: int = 8000
    max_mb: float = 20


class IngestConfig(BaseModel):
    # 自动把微信「文件」通道的存盘目录加入监控
    watch_wechat: bool = True
    # 额外监控的图片目录（含子目录）。data/inbox 始终会被监控，不用写在这里。
    watch_dirs: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )
    # 等文件写完的最长时间；微信/浏览器是分块落盘的
    settle_timeout_s: float = 5.0
    upload: UploadConfig = Field(default_factory=UploadConfig)


class TutorConfig(BaseModel):
    grade: str = "初二"
    style: str = "socratic"
    max_followups: int = 5


class StoreConfig(BaseModel):
    path: str = "data/xiaozhi.db"


class Settings(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    interrupt: InterruptConfig = Field(default_factory=InterruptConfig)
    voiceprint: VoiceprintConfig = Field(default_factory=VoiceprintConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    tutor: TutorConfig = Field(default_factory=TutorConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)

    def output_dir(self) -> Path:
        """截图输出目录，相对路径按项目根目录解析。"""
        path = Path(self.capture.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def inbox_dir(self) -> Path:
        """外部投递目录：手机上传、手动拖入、微信另存为都放这里。"""
        return PROJECT_ROOT / "data" / "inbox"

    def watch_dirs(self) -> list[Path]:
        """实际监控的目录：始终包含 inbox，再加上配置里额外指定的。"""
        candidates = [self.inbox_dir(), *(Path(raw) for raw in self.ingest.watch_dirs)]
        unique: list[Path] = []
        for item in candidates:
            path = item if item.is_absolute() else PROJECT_ROOT / item
            if path not in unique:
                unique.append(path)
        return unique

    def voiceprint_path(self) -> Path:
        """声纹文件前缀，相对路径按项目根目录解析。"""
        path = Path(self.voiceprint.path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def store_path(self) -> Path:
        """学习记录数据库路径，相对路径按项目根目录解析。"""
        path = Path(self.store.path)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _expand_env(value: Any) -> Any:
    """把 yaml 里的 ${VAR} 替换成环境变量，缺失时替换为空串。"""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_settings(config_path: Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    path = config_path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(_expand_env(raw))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def missing_keys(settings: Settings) -> list[str]:
    """返回缺失的 API Key 环境变量名，用于启动前自检。"""
    missing = []
    if not settings.llm.api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not settings.vision.api_key:
        missing.append("DASHSCOPE_API_KEY")
    return missing


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")
