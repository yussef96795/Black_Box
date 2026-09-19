"""Block B DAG nodes (LangGraph).

Four processing nodes plus the HITL payload generator:
  1. quant_extractor   — Extract numerical quantities from chunks
  2. schema_builder    — Build Pydantic schema from quantities
  3. cynical_auditor   — Adversarial validation for lookahead/bounds
  4. hitl_payload      — Batch missing rules + audit flags for HITL

Each node receives the current StrategyState and returns a partial
state update dict (LangGraph convention). All values use only
msgpack-serializable primitives so LangGraph can checkpoint
the state without custom serializers.
"""

from __future__ import annotations

import uuid
from typing import Any

from black_box.state.schema import StrategyState


# ---------------------------------------------------------------------------
# Node 1: Quant Extractor
# ---------------------------------------------------------------------------


def quant_extractor(state: StrategyState) -> dict[str, Any]:
    """Extract numerical quantities from document chunks.

    Returns plain dicts for all quantities to ensure
    msgpack-serializability for LangGraph checkpointing.

    Stub: derives quantities from chunk text patterns.
    Real implementation (Block B Step 1 Node 1) uses a PydanticAI
    agent with the `NULL_AMBIGUOUS` tool flag to extract numerical
    values, constraints, and rules from chunks.
    """
    quantities = [
        {
            "name": "stake",
            "value": 1.0,
            "unit": "fraction",
            "ambiguity_level": "clear",
            "source_chunk_id": "stub:0",
        }
    ]

    return {
        "extracted_quantities": quantities,
        "status": "building",
        "iteration_count": state.iteration_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 2: Schema Builder
# ---------------------------------------------------------------------------


def schema_builder(state: StrategyState) -> dict[str, Any]:
    """Build a schema definition dict from extracted quantities.

    Returns a plain dict for `schema_definition` to ensure
    msgpack-serializability for LangGraph checkpointing.

    Stub: generates a minimal schema dict from the quantities.
    Real implementation (Block B Step 1 Node 2) uses a Pydantic V2
    AST transpiler + SymPy validation of mathematical relationships.
    """
    quantities = state.extracted_quantities
    fields: dict[str, str] = {}
    sympy_exprs: list[str] = []

    for q in quantities:
        qname = q.get("name", "param") if isinstance(q, dict) else q.name
        qsource = q.get("source_chunk_id", "unknown") if isinstance(q, dict) else q.source_chunk_id
        fields[qname] = f"float (source: {qsource})"
        sympy_exprs.append(f"{qname} > 0")

    schema: dict[str, Any] = {
        "model_name": "StrategyParams",
        "fields": fields,
        "sympy_expressions": sympy_exprs,
    }

    return {
        "schema_definition": schema,
        "status": "auditing",
    }


# ---------------------------------------------------------------------------
# Node 3: Cynical Auditor
# ---------------------------------------------------------------------------


def cynical_auditor(state: StrategyState) -> dict[str, Any]:
    """Adversarial validator: detect lookahead bias, missing exits, unbounded params.

    Returns plain dicts for all audit flags to ensure
    msgpack-serializability for LangGraph checkpointing.

    Stub: generates a conservative flag set based on schema structure.
    Real implementation (Block B Step 1 Node 3) runs an adversarial
    agent that tries to find lookahead bias, missing exit rules,
    and unbounded parameters.
    """
    flags: list[dict[str, Any]] = []

    schema = state.schema_definition
    if schema is None or not isinstance(schema, dict) or not schema.get("fields"):
        flags.append(
            {
                "type": "missing_exit",
                "severity": "high",
                "description": "No exit rule detected in strategy parameters",
                "suggestion": "Add explicit exit conditions (e.g., stop_loss, take_profit)",
            }
        )
        return {"audit_flags": flags, "status": "hitl"}

    field_names = set(schema.get("fields", {}).keys())
    has_exit = any("exit" in name or "stop" in name for name in field_names)
    if not has_exit:
        flags.append(
            {
                "type": "missing_exit",
                "severity": "high",
                "description": "No exit rule detected in strategy parameters",
                "suggestion": "Add explicit exit conditions (e.g., stop_loss, take_profit)",
            }
        )

    for q in state.extracted_quantities:
        qname = q.get("name", "param") if isinstance(q, dict) else q.name
        ambiguity = q.get("ambiguity_level", "clear") if isinstance(q, dict) else q.ambiguity_level
        if ambiguity == "clear":
            flags.append(
                {
                    "type": "unbounded_param",
                    "severity": "medium",
                    "description": f"Parameter '{qname}' has no explicit bounds",
                    "suggestion": f"Add bounds constraint for '{qname}'",
                }
            )

    return {
        "audit_flags": flags,
        "status": "hitl" if flags else "complete",
    }


# ---------------------------------------------------------------------------
# Node 4: HITL Payload Generator
# ---------------------------------------------------------------------------


def hitl_payload(state: StrategyState) -> dict[str, Any]:
    """Batch missing rules and audit flags into an HITL payload.

    This node runs when the auditor returns flags requiring human
    review. It collects actionable items into a structured payload
    for the HITL interface.

    Stub: packages existing flags and missing quantities.
    Real implementation (Block B Step 2 Node 4) batches items
    ensuring ≤ 20 items per page.
    """
    audit_flags = state.audit_flags if isinstance(state.audit_flags, list) else []
    # Normalize: AuditFlag objects -> dicts for msgpack-serializability
    flag_dicts: list[dict[str, Any]] = []
    for flag in audit_flags:
        if isinstance(flag, dict):
            flag_dicts.append(flag)
        else:
            flag_dicts.append(flag.model_dump())

    missing_rules = [
        flag["description"] for flag in flag_dicts
        if flag.get("severity") in ("high", "critical")
    ]

    return {
        "hitl_payload": {
            "missing_rules": missing_rules,
            "audit_flags": flag_dicts,
            "current_schema": state.schema_definition,
            "session_id": state.session_id,
            "round": state.round,
        },
        "status": "hitl" if missing_rules else state.status,
    }


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def get_initial_state(session_id: str | None = None) -> StrategyState:
    """Create a fresh StrategyState for a new session."""
    return StrategyState(
        session_id=session_id or uuid.uuid4().hex,
        status="idle",
    )


def state_from_checkpoint(saved: Any) -> StrategyState:
    """Convert a checkpointed state (dict or StateTuple) to StrategyState.

    Returns an empty StrategyState when the checkpointed data
    is empty or missing required fields, so callers can check
    `session_id` to determine whether the session exists.
    """
    if hasattr(saved, "values"):
        values = saved.values
    elif isinstance(saved, dict):
        values = saved
    else:
        values = saved
    if isinstance(values, dict):
        if not values or "session_id" not in values:
            return StrategyState(session_id="")  # empty/unknown session
        return StrategyState.model_validate(values)
    return values
