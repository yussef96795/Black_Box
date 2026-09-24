"""StrategyState schema (Block B — Formulation Engine).

Typed Pydantic state for the formulation LangGraph DAG. All list/dict
fields use msgpack-serializable primitives so the graph checkpoints without
custom serializers (house convention, mirrors ``BlockAState``).

Flow (see ``graph.py``):

    hydrate_spec → structural_gate → cynical_auditor ── router ──► complete
                                                  │             └─► rejected
                                                  └─► hitl_payload → hitl_gate (interrupt)
                                                     resume: apply_answers → structural_gate (loop)

Retry semantics live on ConditionalEdges, not the graph-level
``max_iterations`` dead-man:
  * ``round``  — HITL answer rounds completed (incremented on resume),
    capped by ``max_rounds`` (= 2).
  * ``iteration_count`` — monotonic stage counter for observability.

This file is the single source of truth; the TypeScript mirror at
frontend/src/app/strategy/strategy-state.ts must track it (contract-first,
per plan.md guardrail 2).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClarificationCard(BaseModel):
    """One typed human-in-the-loop question emitted at the interrupt.

    `mutation` is the keyed state path a resume answer patches (e.g.
    ``variants.0.sizing``) — answers are deterministic merges, not text.
    """

    model_config = ConfigDict(extra="forbid")

    card_id: str = Field(description="Batch-unique id, referenced by resume answers")
    type: Literal[
        "missingPillar", "missingPrimitive", "remediation", "universe", "riskOverride"
    ] = Field(description="Card kind; drives the mutation semantics on resume")
    pillar: str = Field(
        default="", description="entry | exit | sizing | risk | universe | codebase"
    )
    title: str = Field(default="")
    evidence: str = Field(
        default="", description="Paper/chunk evidence backing the question"
    )
    proposal: dict[str, Any] | None = Field(
        default=None, description="Machine proposal applied on accept"
    )
    options: list[str] = Field(
        default_factory=list, description="Allowed answer actions / candidates"
    )
    mutation: str = Field(default="", description="Keyed state path this card patches")
    required: bool = Field(
        default=True, description="Rejected required cards keep the session blocked"
    )
    state: Literal["pending", "answered", "skipped"] = Field(default="pending")


class AuditFlag(BaseModel):
    """Contract for one Cynical Auditor finding (see ``auditor.py``)."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        description="e.g. lookahead_bias | missing_exit | no_sizing | unbounded_param"
    )
    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(default="")
    suggestion: str = Field(default="")


class StrategyState(BaseModel):
    """Full formulation state, checkpointed by LangGraph (MemorySaver) and
    snapshotted to disk (``block_b_out_dir``) for restart resilience."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(description="Unique session (thread_id)")
    status: Literal[
        "idle",
        "extracting",
        "building",
        "auditing",
        "hitl",
        "complete",
        "rejected",
        "llm_failed",
    ] = Field(default="idle")
    spec: dict[str, Any] | None = Field(
        default=None, description="The A6 ExecutableStrategySpec being formulated"
    )
    variants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Formulations: entry/exit/sizing triples + sweep grids (≤ MAX_VARIANTS)",
    )
    structural_errors: list[dict[str, Any]] = Field(default_factory=list)
    audit_flags: list[dict[str, Any]] = Field(default_factory=list)
    hitl_cards: list[dict[str, Any]] = Field(
        default_factory=list, description="Current pending card batch"
    )
    answers_applied: list[str] = Field(
        default_factory=list, description="Card ids already applied (idempotence)"
    )
    user_responses: dict[str, Any] = Field(
        default_factory=dict, description="card_id → {action, value} from resume"
    )
    resume_key: str = Field(
        default="",
        description="Hash of the last applied answers batch (idempotent resume)",
    )
    round: int = Field(default=0, description="HITL answer rounds completed")
    max_rounds: int = Field(
        default=2, description="Round cap enforced by ConditionalEdge"
    )
    iteration_count: int = Field(
        default=0, description="Monotonic stage counter (observability)"
    )
    max_iterations: int = Field(
        default=2, description="Synthesis-retry bound inside nodes (dead-man safety)"
    )


__all__ = ["AuditFlag", "ClarificationCard", "StrategyState"]
