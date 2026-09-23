"""Application settings loaded from environment / .env (Rules.md §3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/black_box"
    )

    # ------------------------------------------------------------------
    # Block A — Alpha Feasibility & Strategy Ingestion Engine
    # ------------------------------------------------------------------
    #: LLM backend: "ollama" (local, default) | "fake" (deterministic tests).
    block_a_llm_backend: Literal["ollama", "fake"] = "ollama"
    #: Ollama model used by Stage A1–A5 structured extraction (override via
    #: OLLAMA_MODEL — surfaced as BLOCK_A_OLLAMA_MODEL to avoid clobbering the
    #: model already pulled by the local server config).
    block_a_ollama_model: str = "llama3.2"
    block_a_ollama_base_url: str = "http://localhost:11434/v1"
    #: Instructor Pydantic validation retries (spec §6 item 3: max 3).
    block_a_max_retries: int = 3
    #: Directory holding primitives_registry.json + data_catalog.parquet.
    block_a_config_dir: Path = (
        Path(__file__).resolve().parent.parent / "block_a" / "config"
    )
    #: Output directory for the validated spec matrix (block_a_specs.json).
    block_a_out_dir: Path = Path("out")

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
