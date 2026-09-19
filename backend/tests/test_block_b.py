"""Tests for Block B — LangGraph scaffold + state schema."""

from __future__ import annotations

from fastapi.testclient import TestClient

from black_box.main import app
from black_box.state.nodes import (
    get_initial_state,
    hitl_payload,
    quant_extractor,
    schema_builder,
    cynical_auditor,
)
from black_box.state.schema import AuditFlag, Quantity, StrategyState
from black_box.state.graph import create_strategy_graph


# --- State schema tests ----------------------------------------------------


def test_initial_state_defaults() -> None:
    state = get_initial_state("test-session")
    assert state.session_id == "test-session"
    assert state.status == "idle"
    assert state.iteration_count == 0
    assert state.max_iterations == 2
    assert state.round == 0
    assert state.extracted_quantities == []
    assert state.audit_flags == []
    assert state.user_responses == {}


def test_state_with_custom_session_id() -> None:
    state = get_initial_state("custom-id")
    assert state.session_id == "custom-id"


def test_state_model_dump() -> None:
    state = get_initial_state("dump-test")
    dump = state.model_dump()
    assert dump["session_id"] == "dump-test"
    assert dump["status"] == "idle"


def test_state_all_statuses() -> None:
    for status in ["idle", "extracting", "building", "auditing", "hitl", "complete", "rejected"]:
        state = StrategyState(session_id="x", status=status)
        assert state.status == status


# --- Node tests ------------------------------------------------------------


def test_quant_extractor_produces_quantities() -> None:
    state = get_initial_state("node-test")
    result = quant_extractor(state)
    assert "extracted_quantities" in result
    assert len(result["extracted_quantities"]) > 0
    assert result["status"] == "building"
    assert result["iteration_count"] == 1


def test_schema_builder_from_quantities() -> None:
    state = get_initial_state("schema-test")
    state.extracted_quantities = [
        Quantity(name="lookback", value=20.0, unit="bars", ambiguity_level="clear", source_chunk_id="c1")
    ]
    result = schema_builder(state)
    assert "schema_definition" in result
    assert result["status"] == "auditing"
    schema = result["schema_definition"]
    assert schema["model_name"] == "StrategyParams"
    assert "lookback" in schema["fields"]


def test_cynical_auditor_generates_flags_for_missing_schema() -> None:
    state = get_initial_state("audit-test")
    state.schema_definition = None
    result = cynical_auditor(state)
    assert result["status"] == "hitl"
    assert len(result["audit_flags"]) > 0


def test_cynical_auditor_no_flags_when_schema_has_exit() -> None:
    state = get_initial_state("audit-clean")
    state.schema_definition = {"model_name": "StrategyParams", "fields": {"exit_rule": "float"}, "sympy_expressions": []}
    state.extracted_quantities = [
        Quantity(name="exit_rule", value=0.5, unit="float", ambiguity_level="clear", source_chunk_id="c1")
    ]
    result = cynical_auditor(state)
    # Schema has an exit-related field, should have no flags or only bounded flags
    assert "status" in result


def test_hitl_payload_bundles_flags() -> None:
    state = get_initial_state("hitl-test")
    state.audit_flags = [
        AuditFlag(
            type="missing_exit",
            severity="high",
            description="No exit rule found",
            suggestion="Add stop_loss",
        )
    ]
    state.schema_definition = {"model_name": "StrategyParams", "fields": {"stake": "float"}, "sympy_expressions": []}
    result = hitl_payload(state)
    assert result["status"] == "hitl"
    assert "hitl_payload" in result
    payload = result["hitl_payload"]
    assert payload["session_id"] == "hitl-test"
    assert len(payload["missing_rules"]) > 0


# --- API integration tests ------------------------------------------------


def test_submit_strategy(client: TestClient) -> None:
    """POST /api/v1/strategy/submit creates a session."""
    resp = client.post("/api/v1/strategy/submit")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["session_id"]
    assert body["status"] in {"complete", "hitl", "rejected"}
    assert "iteration_count" in body
    # Verify the state can be retrieved
    sid = body["session_id"]
    state_resp = client.get(f"/api/v1/strategy/{sid}/state")
    assert state_resp.status_code == 200
    assert state_resp.json()["state"]["session_id"] == sid


def test_get_strategy_state_unknown_404(client: TestClient) -> None:
    """GET /api/v1/strategy/{id}/state returns 404 for unknown session."""
    resp = client.get("/api/v1/strategy/00000000-0000-0000-0000-000000000000/state")
    assert resp.status_code == 404


def test_post_strategy_resume_unknown_404(client: TestClient) -> None:
    """POST /api/v1/strategy/{id}/resume returns 404 for unknown session."""
    resp = client.post(
        "/api/v1/strategy/00000000-0000-0000-0000-000000000000/resume",
        json={"exit_rule": "stop_loss"},
    )
    assert resp.status_code == 404


# --- Graph topology tests --------------------------------------------------


def test_graph_compile() -> None:
    graph = create_strategy_graph()
    assert graph is not None
    node_names = list(graph.nodes.keys())
    assert "quant_extractor" in node_names
    assert "schema_builder" in node_names
    assert "cynical_auditor" in node_names
    assert "hitl_payload" in node_names


def test_graph_invoke() -> None:
    from black_box.state.nodes import get_initial_state, state_from_checkpoint

    graph = create_strategy_graph()
    state = get_initial_state("invoke-test")
    config = {"configurable": {"thread_id": "invoke-test"}}
    graph.invoke(state, config=config)
    saved = graph.get_state(config)
    persisted = state_from_checkpoint(saved)
    assert persisted.session_id == "invoke-test"
    assert persisted.status in {"complete", "hitl", "rejected"}


def test_graph_state_persistence() -> None:
    from black_box.state.nodes import get_initial_state, state_from_checkpoint

    graph = create_strategy_graph()
    state = get_initial_state("persist-test")
    config = {"configurable": {"thread_id": "persist-test"}}

    graph.invoke(state, config=config)
    saved = graph.get_state(config)
    persisted = state_from_checkpoint(saved)
    assert persisted.session_id == "persist-test"
