"""Block B strategy API routes (Formulation Engine).

    POST  /api/v1/strategy/submit          — start a formulation session
    GET   /api/v1/strategy/{id}/state      — current session state
    GET   /api/v1/strategy/{id}/stream     — SSE snapshot stream
    POST  /api/v1/strategy/{id}/resume     — apply answers + resume the DAG

The DAG runs synchronously to its first HITL interrupt; `resume` injects
answers via `Command(resume=...)` and the graph re-audits. Sessions are
checkpointed in-process (MemorySaver) AND snapshotted to disk
(`block_b_out_dir`) — after a process restart the disk snapshot is
replayed deterministically back onto the interrupt (`replay_park`) so a
parked HITL session stays resumable.

`POST /resume` is idempotent: resubmitting the identical answers batch
returns the current state without re-invoking the graph.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from black_box.block_a.models import ExecutableStrategySpec
from black_box.block_b.hitl import load_state, save_state
from black_box.core.config import get_settings
from black_box.schemas import StrategyResumeRequest, StrategySubmitRequest
from black_box.state.graph import create_strategy_graph, replay_park
from black_box.state.nodes import get_initial_state, state_from_checkpoint
from black_box.state.schema import StrategyState

logger = logging.getLogger(__name__)

strategy_router = APIRouter(prefix="/strategy", tags=["strategy"])

_graph = create_strategy_graph()


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _out_dir() -> object:
    return get_settings().block_b_out_dir


def _payload(state: StrategyState) -> dict:
    """Stable response shape for the strategy API contract."""
    return {
        "session_id": state.session_id,
        "status": state.status,
        "round": state.round,
        "variants": state.variants,
        "hitl_cards": state.hitl_cards,
        "structural_errors": state.structural_errors,
        "audit_flags": state.audit_flags,
        "answers_applied": state.answers_applied,
    }


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


def _resolve_spec(request: StrategySubmitRequest) -> dict:
    """Validate an inline spec or load a curated A6 spec by id."""
    if request.spec is not None:
        return ExecutableStrategySpec.model_validate(request.spec).model_dump(
            mode="json"
        )
    if request.spec_id:
        specs_file = get_settings().block_a_out_dir / "block_a_specs.json"
        if not specs_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="out/block_a_specs.json not found — run a Block A pass first",
            )
        specs = json.loads(specs_file.read_text(encoding="utf-8"))
        found = next((s for s in specs if s.get("spec_id") == request.spec_id), None)
        if found is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"spec {request.spec_id} not found in block_a_specs.json",
            )
        return found
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="provide either spec_id or an inline spec payload",
    )


# ---------------------------------------------------------------------------
# Session state access
# ---------------------------------------------------------------------------


def _ensure(session_id: str) -> StrategyState:
    """State from the live checkpointer, or the disk snapshot (re-parked)."""
    snap = _graph.get_state(_config(session_id))
    live = state_from_checkpoint(snap)
    if live.session_id == session_id and live.spec is not None:
        return live

    persisted = load_state(session_id, _out_dir())
    if persisted is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"strategy session {session_id} not found",
        )
    state = StrategyState.model_validate(persisted)
    if state.status == "hitl":
        # Process restarted: replay the deterministic pre-audit chain back
        # onto the interrupt so the thread becomes resumable again.
        try:
            replay_park(_graph, state.model_dump(), session_id)
        except GraphInterrupt:
            pass
        snap = _graph.get_state(_config(session_id))
        reparked = state_from_checkpoint(snap)
        if reparked.session_id == session_id:
            return reparked
    return state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@strategy_router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_strategy(
    body: StrategySubmitRequest,
    session_id: str | None = None,
) -> dict:
    """Start a formulation session from a curated A6 spec."""
    spec = _resolve_spec(body)
    sid = session_id or uuid.uuid4().hex

    existing = _graph.get_state(_config(sid))
    if state_from_checkpoint(existing).session_id == sid:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"session {sid} already exists — resume it or use a new id",
        )

    initial = get_initial_state(sid)
    initial.spec = spec
    try:
        _graph.invoke(initial.model_dump(), _config(sid))
    except GraphInterrupt:
        pass  # parked at the HITL gate — expected

    state = _ensure(sid)
    save_state(sid, state.model_dump(), _out_dir())
    return _payload(state)


@strategy_router.get("/{session_id}/state")
async def get_strategy_state(session_id: str) -> dict:
    """Current strategy session state (live checkpoint or disk snapshot)."""
    return _payload(_ensure(session_id))


@strategy_router.get("/{session_id}/stream")
async def stream_strategy(session_id: str) -> StreamingResponse:
    """Server-sent-events replay of the run so far.

    The DAG is synchronous, so this emits a snapshot of the completed
    supersteps: `formulation`, then either `hitl_request` (cards) or
    `done`. (Live per-node push lands with async graph runs.)
    """

    def _events() -> list[tuple[str, dict]]:
        state = _ensure(session_id)
        events: list[tuple[str, dict]] = [("formulation", _payload(state))]
        if state.status == "hitl":
            events.append(
                ("hitl_request", {"round": state.round, "cards": state.hitl_cards})
            )
        if state.status in ("complete", "rejected"):
            events.append(("done", {"status": state.status}))
        return events

    def _gen():
        for event_type, data in _events():
            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@strategy_router.post("/{session_id}/resume")
async def resume_strategy(session_id: str, body: StrategyResumeRequest) -> dict:
    """Apply typed answers and resume the DAG (idempotent per answers batch)."""
    state = _ensure(session_id)

    key = hashlib.sha1(
        json.dumps(body.answers, sort_keys=True, default=str).encode()
    ).hexdigest()
    if key == state.resume_key:
        return _payload(state)  # identical batch already applied — no-op

    if state.status != "hitl":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"session not awaiting input (status={state.status})",
        )

    try:
        _graph.invoke(Command(resume={"answers": body.answers}), _config(session_id))
    except GraphInterrupt:
        pass  # re-parked for a second HITL round

    updated = _ensure(session_id)
    save_state(session_id, updated.model_dump(), _out_dir())
    return _payload(updated)
