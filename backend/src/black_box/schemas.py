"""API response schemas for Black_Box (Rules.md §3: pydantic schemas)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkOut(BaseModel):
    """A structure-aware semantic chunk returned by the ingestion endpoint.

    Mirrors the service-level `Chunk` dataclass but is the serialization
    contract other blocks (B/C/D) consume — keep it stable.
    """

    id: str = Field(description="Stable chunk identifier (document_id + index)")
    text: str = Field(description="Raw chunk text content")
    page: int | None = Field(
        default=None, description="Source page number (None for HTML/MD)"
    )
    heading: str | None = Field(
        default=None, description="Nearest heading context, ' | ' joined"
    )
    tokens: int = Field(default=0, description="Approximate token count")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Structure flags (table/equation/section)"
    )


class IngestResponse(BaseModel):
    """Payload of `POST /api/v1/documents/ingest`."""

    document_id: str
    format: str
    chunk_count: int
    total_tokens: int
    chunks: list[ChunkOut]
