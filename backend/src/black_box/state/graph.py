"""Block B LangGraph DAG — Formulation Engine.

Topology:

    hydrate_spec → structural_gate → cynical_auditor ─ router ─► mark_complete → END
                                                     │          └► mark_rejected → END
                                                     └► hitl_payload → hitl_gate (interrupt)
                                                        resume (Command) → apply_answers → structural_gate (loop)

Caps are ConditionalEdges on state fields — the graph-level
``max_iterations`` dead-man is NOT the control (a parked HITL stream trips
it; see plan.md note). ``round`` counts completed HITL answer rounds;
``max_rounds = 2`` bounds them before rejection.

``replay_park`` restores a thread after a process restart from the disk
snapshot: the pre-audit chain is deterministic, so replaying to the
interrupt reproduces the same parked state with ``round`` preserved
(``hitl_payload`` never increments it — ``apply_answers`` does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from black_box.state.nodes import (
    apply_answers,
    cynical_auditor,
    hitl_gate,
    hitl_payload,
    hydrate_spec,
    mark_complete,
    mark_rejected,
    structural_gate,
)
from black_box.state.schema import StrategyState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def _route_after_audit(
    state: StrategyState,
) -> Literal["mark_complete", "mark_rejected", "hitl_payload"]:
    """Post-audit router: clean → complete; fixable → HITL; else rejected."""
    if not state.audit_flags and not state.structural_errors:
        return "mark_complete"
    if state.round < state.max_rounds:
        return "hitl_payload"
    return "mark_rejected"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> Any:
    """Compile the formulation DAG (MemorySaver by default)."""
    checkpointer = checkpointer or MemorySaver()

    workflow = StateGraph(StrategyState)
    workflow.add_node("hydrate_spec", hydrate_spec)
    workflow.add_node("structural_gate", structural_gate)
    workflow.add_node("cynical_auditor", cynical_auditor)
    workflow.add_node("hitl_payload", hitl_payload)
    workflow.add_node("hitl_gate", hitl_gate)
    workflow.add_node("apply_answers", apply_answers)
    workflow.add_node("mark_complete", mark_complete)
    workflow.add_node("mark_rejected", mark_rejected)

    workflow.add_edge(START, "hydrate_spec")
    workflow.add_edge("hydrate_spec", "structural_gate")
    workflow.add_edge("structural_gate", "cynical_auditor")
    workflow.add_conditional_edges("cynical_auditor", _route_after_audit)
    workflow.add_edge("hitl_payload", "hitl_gate")
    workflow.add_edge("hitl_gate", "apply_answers")
    workflow.add_edge("apply_answers", "structural_gate")  # re-audit loop
    workflow.add_edge("mark_complete", END)
    workflow.add_edge("mark_rejected", END)

    return workflow.compile(checkpointer=checkpointer)


def create_strategy_graph() -> Any:
    """Module-level compiled graph (in-memory checkpointer) for the API."""
    return build_graph()


def replay_park(graph: Any, persisted: dict[str, Any], thread_id: str) -> None:
    """Re-park a persisted HITL session onto the in-memory checkpointer.

    Only called when a process restart left the thread dangling: replaying
    the deterministic pre-audit chain stops at ``hitl_gate``'s interrupt,
    restoring a resumable thread with the card batch intact.
    """
    graph.invoke(
        persisted,
        config={"configurable": {"thread_id": thread_id}},
        interrupt_after="hitl_gate",
    )


__all__ = ["_route_after_audit", "build_graph", "create_strategy_graph", "replay_park"]
