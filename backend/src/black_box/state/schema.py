"""StrategyState schema (Block B Step 1).

Typed Pydantic model defining the full strategy formulation state.
All sub-models use only msgpack-serializable primitive types
(dict, list, str, int, float) so the state can be checkpointed
by LangGraph without custom serializers.

This is the single source of truth — the TypeScript counterpart at
frontend/src/app/strategy/strategy-state.ts must mirror this model.

State flows through a LangGraph DAG:
  idle → extracting → building → auditing → [hitl | complete | rejected]

The iteration_count is capped at max_iterations (default 2) via a
LangGraph ConditionalEdge, enforcing the deterministic N ≤ 2 retry
loop described in plan.md Block B Step 1.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Sub-models — all fields must be msgpack-serializable primitives
# ---------------------------------------------------------------------------


class Quantity(BaseModel):
    """A numerical quantity extracted from a document chunk."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Parameter name, e.g. 'lookback'")
    value: float = Field(description="Extracted numeric value")
    unit: str = Field(default="", description="Unit of measurement")
    ambiguity_level: Literal["clear", "ambiguous", "NULL_AMBIGUOUS"] = Field(
        default="clear",
        description="'NULL_AMBIGUOUS' when value cannot be reliably determined",
    )
    source_chunk_id: str = Field(default="", description="Origin chunk identifier")


class SchemaDefinition(BaseModel):
    """Auto-generated Pydantic model definition from extracted quantities.

    All field values are stored as strings to ensure msgpack
    serializability for LangGraph checkpointing.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="StrategyParams", description="Pydantic model name")
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Field name → JSON-serialized type/constraints (string)",
    )
    sympy_expressions: list[str] = Field(
        default_factory=list,
        description="SymPy expression strings for validation",
    )


class AuditFlag(BaseModel):
    """A single issue flagged by the Cynical Auditor."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="e.g. 'lookahead_bias', 'missing_exit', 'unbounded_param'")
    severity: Literal["low", "medium", "high", "critical"] = Field(description="Severity level")
    description: str = Field(description="Human-readable description of the issue")
    suggestion: str = Field(default="", description="Suggested fix")


# ---------------------------------------------------------------------------
# StrategyState — the LangGraph state graph carries through this model
# ---------------------------------------------------------------------------


class StrategyState(BaseModel):
    """Full strategy formulation state, persisted via PostgreSQL checkpoints.

    All fields use msgpack-serializable primitives so LangGraph
    can checkpoint the state without custom serializers.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Unique session identifier")
    round: int = Field(default=0, description="HITL round counter")
    extracted_quantities: list[Quantity] = Field(default_factory=list)
    schema_definition: dict[str, Any] | None = Field(
        default=None,
        description="Auto-generated schema as a dict for checkpoint serializability",
    )
    audit_flags: list[AuditFlag] = Field(default_factory=list)
    user_responses: dict[str, str] = Field(default_factory=dict)
    status: Literal["idle", "extracting", "building", "auditing", "hitl", "complete", "rejected"] = Field(
        default="idle",
        description="Current pipeline stage",
    )
    iteration_count: int = Field(default=0, description="Monotonically increasing retry counter")
    max_iterations: int = Field(default=2, description="Hard retry cap enforced by ConditionalEdge")
