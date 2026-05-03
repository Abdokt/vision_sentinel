from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List

# Resolve .env relative to this file, not the CWD — so the script can be run
# from any directory and settings will always load correctly.
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=()
    )

    # Camera sources — accept int (webcam index) or str (RTSP URL)
    camera_0_source: str = "0"
    camera_1_source: str = "1"
    camera_0_name: str = "Camera 0"
    camera_1_name: str = "Camera 1"

    # Camera hardware settings
    camera_backend: str = "dshow"   # Windows: dshow | Linux/Mac: ""
    camera_width: int = 640
    camera_height: int = 480

    # Detection
    model_path: str = "models/yolo11n.pt"
    confidence_threshold: float = 0.75
    inference_size: int = 320
    device: str = "cpu"

    # API security
    api_key: str = "change_me"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:8000"

    # Performance
    frame_skip: int = 1
    jpeg_quality: int = 75
    max_occupancy: int = 3

    # Camera enable / disable — set false to skip a camera at startup
    # Can also be toggled at runtime via POST /api/cameras/{cam_id}/enable|disable
    camera_0_enabled: bool = False
    camera_1_enabled: bool = False

    # Windows device watchdog
    enable_device_watchdog: bool = True
    camera_0_device_name: str = "HD Pro Webcam C920"
    camera_1_device_name: str = "Integrated Camera"

    @field_validator("camera_backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"dshow", "v4l2", ""}
        if v not in allowed:
            raise ValueError(f"camera_backend must be one of {allowed}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    def get_camera_source(self, index: int) -> int | str:
        """Returns int for webcam index, str for RTSP URL."""
        raw = self.camera_0_source if index == 0 else self.camera_1_source
        try:
            return int(raw)
        except ValueError:
            return raw  # It's an RTSP URL string

    def get_allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()