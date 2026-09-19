"""Block B strategy API routes (Formulation Engine).

Provides endpoints for the LangGraph-driven strategy formulation:

    POST   /api/v1/strategy/submit   — Create a new strategy session and kick off the DAG
    GET    /api/v1/strategy/{id}/state — Retrieve current session state
    POST   /api/v1/strategy/{id}/resume — Inject HITL responses and resume the DAG

All endpoints use the strategy router mounted under `/strategy`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from black_box.state.graph import create_strategy_graph
from black_box.state.nodes import get_initial_state, state_from_checkpoint
from black_box.state.schema import StrategyState

logger = logging.getLogger(__name__)

strategy_router = APIRouter(prefix="/strategy", tags=["strategy"])

_graph = create_strategy_graph()


@strategy_router.post(
    "/submit",
    status_code=status.HTTP_201_CREATED,
)
async def submit_strategy(
    session_id: str | None = None,
) -> dict:
    """Create a new strategy session and kick off the LangGraph DAG.

    Creates a fresh `StrategyState`, invokes the graph to start
    the extraction pipeline, and returns the session ID plus the
    initial state.

    On completion the status will be `complete`, `rejected`, or
    `hitl` (requiring human review).
    """
    state = get_initial_state(session_id)

    try:
        _graph.invoke(
            state,
            config={"configurable": {"thread_id": state.session_id}},
        )
    except Exception as exc:
        logger.exception("Strategy DAG invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy formulation failed: {exc}",
        ) from exc

    # Get the persisted state from the checkpoint
    config = {"configurable": {"thread_id": state.session_id}}
    saved = _graph.get_state(config)
    persisted = state_from_checkpoint(saved)

    return {
        "session_id": persisted.session_id,
        "status": persisted.status,
        "iteration_count": persisted.iteration_count,
        "round": persisted.round,
        "extracted_quantities_count": len(persisted.extracted_quantities),
        "audit_flags_count": len(persisted.audit_flags),
        "schema_definition": persisted.schema_definition,
    }


def _session_exists(config: dict) -> bool:
    """Check if a session has meaningful data in the checkpoint."""
    try:
        saved = _graph.get_state(config)
        persisted = state_from_checkpoint(saved)
        # A real session has a non-empty session_id and either
        # extracted quantities, audit flags, or has progressed past idle
        return (
            persisted.session_id
            and (
                persisted.extracted_quantities
                or persisted.audit_flags
                or persisted.status != "idle"
            )
        )
    except Exception:
        return False


@strategy_router.get(
    "/{session_id}/state",
)
async def get_strategy_state(
    session_id: str,
) -> dict:
    """Retrieve the current state of a strategy session.

    Returns 404 if the session does not exist (no meaningful checkpoint).
    """
    config = {"configurable": {"thread_id": session_id}}

    persisted = state_from_checkpoint(_graph.get_state(config))
    if not persisted.session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy session found with id '{session_id}'",
        )

    return {
        "session_id": session_id,
        "state": persisted.model_dump(),
    }


@strategy_router.post(
    "/{session_id}/resume",
)
async def resume_strategy(
    session_id: str,
    user_responses: dict[str, str],
) -> dict:
    """Inject HITL responses and resume the strategy DAG.

    Args:
        session_id: The session to resume.
        user_responses: Keyed answers from the HITL interface,
            e.g. `{"exit_rule": "trailing_stop"}`.

    Returns:
        Updated strategy state after the DAG continues from
        the interrupt point.
    """
    config = {"configurable": {"thread_id": session_id}}

    persisted = state_from_checkpoint(_graph.get_state(config))
    if not persisted.session_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy session found with id '{session_id}'",
        )

    try:
        current_state = persisted
        current_state.user_responses.update(user_responses)
        current_state.round += 1
        current_state.status = "extracting"

        _graph.invoke(current_state, config=config)

        # Get the updated state from checkpoint
        saved = _graph.get_state(config)
        updated = state_from_checkpoint(saved)

        return {
            "session_id": updated.session_id,
            "status": updated.status,
            "iteration_count": updated.iteration_count,
            "round": updated.round,
            "extracted_quantities_count": len(updated.extracted_quantities),
            "audit_flags_count": len(updated.audit_flags),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Strategy resume failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume strategy: {exc}",
        ) from exc
