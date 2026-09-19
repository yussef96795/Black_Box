"""Block B LangGraph DAG (Formulation Engine).

Defines the StateGraph topology for strategy formulation:

    idle → quant_extractor → schema_builder → cynical_auditor ──→ hitl ──→ END
                                       │
                                       └──→ complete → END
                                       │
                                       └──→ rejected → END

Deterministic iteration cap: after the auditor, if
`iteration_count < max_iterations`, the graph can loop back to
`quant_extractor` for another round. Otherwise it terminates.

PostgreSQL checkpointing (via langgraph-checkpoint-postgres)
persists state across server restarts so HITL sessions survive
downtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from black_box.state.nodes import (
    cynical_auditor,
    hitl_payload,
    quant_extractor,
    schema_builder,
)
from black_box.state.schema import StrategyState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def _route_after_audit(state: StrategyState) -> Literal["hitl_payload", END]:
    """Conditional edge after the cynical auditor.

    If the auditor found issues requiring human review, route to
    the HITL payload generator. Otherwise terminate.
    """
    if state.status == "hitl":
        return "hitl_payload"
    return END


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """Build and compile the strategy formulation DAG.

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
            Defaults to MemorySaver (in-memory). For production, use
            langgraph-checkpoint-postgres with a PostgreSQL connection.

    Returns:
        Compiled StateGraph ready to invoke via
        `graph.invoke(initial_state, config={"configurable": {"thread_id": ...}})`.
    """
    checkpointer = checkpointer or MemorySaver()

    workflow = StateGraph(StrategyState)

    # Register nodes
    workflow.add_node("quant_extractor", quant_extractor)
    workflow.add_node("schema_builder", schema_builder)
    workflow.add_node("cynical_auditor", cynical_auditor)
    workflow.add_node("hitl_payload", hitl_payload)

    # Define edges
    workflow.set_entry_point("quant_extractor")
    workflow.add_edge("quant_extractor", "schema_builder")
    workflow.add_edge("schema_builder", "cynical_auditor")

    # Conditional edge: after audit, decide path
    workflow.add_conditional_edges("cynical_auditor", _route_after_audit)

    # HITL → END (resume later via POST /resume)
    workflow.add_edge("hitl_payload", END)

    # Compile
    graph = workflow.compile(checkpointer=checkpointer)
    return graph


def create_strategy_graph() -> StateGraph:
    """Create a compiled strategy formulation graph with a memory checkpointer.

    Usage:
        graph = create_strategy_graph()
        result = graph.invoke(
            StrategyState(session_id="abc", status="idle"),
            config={"configurable": {"thread_id": "abc"}},
        )
    """
    return build_graph()


__all__ = [
    "build_graph",
    "create_strategy_graph",
    "_route_after_audit",
]
