"""Application settings loaded from environment / .env (Rules.md §3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Black_Box backend.

    All values are resolved from environment variables or a `.env` file.
    Nothing is hardcoded (Rules.md §3 + §5).
    """

    app_name: str = "Black_Box"
    version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Ingested document / working storage
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    chunk_cache_dir: Path = Path("data/chunks")

    # Ingestion guardrails (Rules.md §5)
    max_upload_size: int = 25 * 1024 * 1024  # bytes
    max_num_pages: int = 500
    # Comma-separated allowlist of Docling InputFormat names we accept.
    allowed_formats: str = "pdf,docx,html,md"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database (Block B+ state checkpointing)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/black_box"

    @property
    def allowed_format_list(self) -> list[str]:
        return [f.strip().upper() for f in self.allowed_formats.split(",") if f.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
