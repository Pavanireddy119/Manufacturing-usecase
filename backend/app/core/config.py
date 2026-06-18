from functools import lru_cache
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_cors_origins(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [origin.strip() for origin in value.split(",") if origin.strip()]


CorsOrigins = Annotated[list[str], BeforeValidator(parse_cors_origins)]


class Settings(BaseSettings):
    app_name: str = "QualityVision AI Backend"
    app_version: str = "1.0.0"
    database_url: str = "sqlite:///./qualityvision.db"
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    backend_cors_origins: CorsOrigins = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    upload_dir: str = "uploads"

    # ML inference settings. When ml_enabled is True the backend uses the real
    # YOLOv8 damage model from the ml/ package; it transparently falls back to a
    # placeholder if torch/ultralytics or the weights are unavailable.
    ml_enabled: bool = True
    # Empty string => use the default path resolved by YoloDamagePredictor.
    ml_model_path: str = ""
    ml_conf_threshold: float = 0.25
    ml_iou_threshold: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
