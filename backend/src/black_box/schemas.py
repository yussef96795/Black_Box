"""API response schemas for Black_Box (Rules.md §3: pydantic schemas)."""

from __future__ import annotations

from typing import Any, Literal

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


class TableValidationReport(BaseModel):
    """Per-table validation outcome (Block A Step 1d)."""

    table_id: str
    status: str = Field(description="fit | orphan | parse-error")
    rows: int = 0
    columns: int = 0
    errors: list[str] = Field(default_factory=list)


class MathResolution(BaseModel):
    """Resolved math/symbol occurrence (Block A Step 1c)."""

    original: str
    resolved: str
    symbol_table: dict[str, str] = Field(default_factory=dict)
    context: str = Field(default="")


class IngestResponse(BaseModel):
    """Payload of `POST /api/v1/documents/ingest`."""

    document_id: str
    format: str
    chunk_count: int
    total_tokens: int
    chunks: list[ChunkOut]
    tables: list[TableValidationReport] = Field(default_factory=list)
    math: list[MathResolution] = Field(default_factory=list)


# --- Block A Step 2: Feasibility Auditor -------------------------------------


class DataDependency(BaseModel):
    """One data-source requirement detected in the strategy spec."""

    name: str
    status: Literal["satisfied", "missing", "partial"]
    latency_ms: int | None = Field(
        default=None, description="null until real data feeds are measured"
    )
    evidence: str = Field(default="")


class ModelCheck(BaseModel):
    """One model/compute requirement flagged by the compute checker."""

    name: str
    supported: bool
    reason: str = Field(default="")


class DomainMapping(BaseModel):
    """A TradFi concept mapped to its crypto analog (with confidence)."""

    tradfi: str
    crypto: str
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = Field(default="")


class FeasibilityAuditRequest(BaseModel):
    """Input to `POST /api/v1/feasibility/audit`.

    `text` is the extracted strategy specification (concatenated chunks or raw
    document text); `tables` optionally carry the 1d validation reports so the
    auditor can gate on unreadable parameter tables.
    """

    text: str = Field(min_length=1, description="Strategy specification text")
    tables: list[TableValidationReport] = Field(default_factory=list)


class FeasibilityCheck(BaseModel):
    """Per-checker verdict inside a FeasibilityResult."""

    name: str
    status: Literal["PASSED", "REQUIRES_HITL", "REJECTED"]
    details: dict[str, Any] = Field(default_factory=dict)


class FeasibilityResult(BaseModel):
    """Aggregated verdict of `POST /api/v1/feasibility/audit`."""

    status: Literal["PASSED", "REQUIRES_HITL", "REJECTED"]
    checks: list[FeasibilityCheck]
    summary: str


# --- Block B — Formulation Engine -------------------------------------------


class StrategySubmitRequest(BaseModel):
    """Input to `POST /api/v1/strategy/submit`.

    Either `spec_id` (a curated A6 spec from `out/block_a_specs.json`) or an
    inline `spec` payload. Exactly one must be provided.
    """

    spec_id: str | None = Field(
        default=None, description="spec_id of a curated A6 spec (block_a_specs.json)"
    )
    spec: dict[str, Any] | None = Field(
        default=None, description="Inline ExecutableStrategySpec payload"
    )


class StrategyResumeRequest(BaseModel):
    """Input to `POST /api/v1/strategy/{id}/resume`.

    `answers` maps card_id → `{"action": "accept|override|reject|approve",
    "value": {...}}` — deterministic keyed mutations, never free text.
    """

    answers: dict[str, Any] = Field(default_factory=dict)
