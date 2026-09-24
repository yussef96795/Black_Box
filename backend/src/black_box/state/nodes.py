"""Block B DAG nodes (LangGraph) — thin wrappers over the formulation engine.

Nodes stay thin: the logic lives in ``black_box.block_b`` (testable without
the graph); these map the engine onto ``StrategyState`` and the HITL
interrupt. Every return value uses msgpack-serializable primitives.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import interrupt

from black_box.block_b.auditor import audit_variants
from black_box.block_b.formulation import hydrate_spec as run_hydrate
from black_box.block_b.formulation import structural_gate as run_gate
from black_box.block_b.hitl import apply_answers as run_apply
from black_box.block_b.hitl import build_cards
from black_box.state.schema import StrategyState

# ---------------------------------------------------------------------------
# Pre-audit chain
# ---------------------------------------------------------------------------


def hydrate_spec(state: StrategyState) -> dict[str, Any]:
    """Formulate the A6 spec into variants (entry/exit/sizing triples)."""
    if not state.spec:
        return {
            "structural_errors": [
                {
                    "type": "bad_data",
                    "pillar": "universe",
                    "message": "No spec to formulate.",
                }
            ],
            "status": "auditing",
        }
    return {**run_hydrate(state.spec), "status": "auditing"}


def structural_gate(state: StrategyState) -> dict[str, Any]:
    """Deterministic well-posedness gate (registry + data + grid sanity)."""
    return {"structural_errors": run_gate(state), "status": "auditing"}


def cynical_auditor(state: StrategyState) -> dict[str, Any]:
    """Adversarial well-posedness audit (lookahead, completeness, bounds)."""
    return {
        "audit_flags": audit_variants(state),
        "iteration_count": state.iteration_count + 1,
        "status": "auditing",
    }


# ---------------------------------------------------------------------------
# HITL gate
# ---------------------------------------------------------------------------


def hitl_payload(state: StrategyState) -> dict[str, Any]:
    """Batch the pending questions into typed clarification cards."""
    return {"hitl_cards": build_cards(state), "status": "hitl"}


def hitl_gate(state: StrategyState) -> dict[str, Any]:
    """LangGraph ``interrupt()`` — park the graph until the human answers.

    First run pauses with the card batch; a resume via
    ``Command(resume={"answers": ...})`` flows the answers into
    ``user_responses`` and the graph continues to ``apply_answers``.
    """
    answers = interrupt(
        {"type": "hitl_request", "round": state.round, "cards": state.hitl_cards}
    )
    if answers:
        return {"user_responses": answers.get("answers", {})}
    return {}


def apply_answers(state: StrategyState) -> dict[str, Any]:
    """Merge human answers (keyed mutations) and count the HITL round.

    The round only advances when at least one new answer was actually
    applied — a duplicate/no-op resume must not burn a HITL round.
    """
    update = run_apply(state)
    if update.pop("applied_any", False):
        update["round"] = state.round + 1
    update.setdefault("status", "auditing")
    return update


def mark_complete(state: StrategyState) -> dict[str, Any]:
    """Terminal success — formulation validated, ready for Block C."""
    return {"status": "complete"}


def mark_rejected(state: StrategyState) -> dict[str, Any]:
    """Terminal rejection — unanswered flaws after the round cap."""
    return {"status": "rejected"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_initial_state(session_id: str | None = None) -> StrategyState:
    """Fresh session state for a new submission."""
    return StrategyState(session_id=session_id or uuid.uuid4().hex, status="idle")


def state_from_checkpoint(saved: Any) -> StrategyState:
    """Convert a checkpointed state (dict or StateTuple) to StrategyState."""
    values = getattr(saved, "values", None)
    if isinstance(values, dict):
        if not values or "session_id" not in values:
            return StrategyState(session_id="")  # empty/unknown session
        return StrategyState.model_validate(values)
    if isinstance(saved, dict) and saved:
        return StrategyState.model_validate(saved)
    return values


__all__ = [
    "apply_answers",
    "cynical_auditor",
    "get_initial_state",
    "hitl_gate",
    "hitl_payload",
    "hydrate_spec",
    "mark_complete",
    "mark_rejected",
    "state_from_checkpoint",
    "structural_gate",
]
