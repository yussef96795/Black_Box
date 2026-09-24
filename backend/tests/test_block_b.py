"""Block B Formulation Engine tests — engine, XLangGraph DAG, and API.

Engine tests run without the graph (pure functions over StrategyState);
graph tests exercise the interrupt/resume lifecycle through a real
MemorySaver checkpointer; API tests go through TestClient.

A mechanical spec (no sizing) is the canonical HITL case: hydration emits
the entry/exit triples, the missing sizing pillar becomes a typed
missingPillar card, and an `accept` resume re-derives the full pillar
payload (component + params + sweep grid) from the registry.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from langgraph.types import Command

from black_box.block_b.auditor import audit_variants
from black_box.block_b.formulation import (
    MAX_VARIANTS,
    hydrate_spec,
    referenced_primitives,
    structural_gate,
    unbacked_primitives,
)
from black_box.block_b.hitl import apply_answers, build_cards, load_state, save_state
from black_box.state.graph import build_graph, replay_park
from black_box.state.nodes import get_initial_state, state_from_checkpoint
from black_box.state.schema import StrategyState
from black_box.strategylib import by_pillar, resolve


def make_spec(**overrides: object) -> dict:
    """A valid A6 spec dict with sensible defaults; override per-test."""
    spec: dict = {
        "spec_id": "STRAT_TEST_T0",
        "tier": "TIER_0_LITERAL",
        "target_asset": "BTC-USD",
        "timeframe": "15m",
        "entry_trigger_primitive": "TRIGGER_CROSS_ABOVE",
        "exit_trigger_primitive": "TRIGGER_CROSS_BELOW",
        "filter_primitives": [],
        "parameters": {
            "mode": "static",
            "indicators": ["IND_EMA"],
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        },
        "risk_annotations": [],
        "causal_anchor_notes": "test §1",
        "is_testable": True,
    }
    spec.update(overrides)
    return spec


def with_sizing(
    spec: dict, component_id: str = "sizing.fractional", fraction: float = 0.02
) -> dict:
    """Inject a sizing component via A6 `parameters` (no-HITL route)."""
    spec["parameters"] = {
        **spec.get("parameters", {}),
        "sizing": {"component_id": component_id, "params": {"fraction": fraction}},
    }
    return spec


LOOKAHEAD_AST = {
    "kind": "binary_op",
    "op": "lag",
    "left": {"kind": "transform", "id": "close"},
    "right": {"kind": "literal", "id": "-1"},
}


def _formulated(spec: dict, *, audit: bool = True) -> StrategyState:
    """Drive the pure engine chain (mirrors hydrate→gate→auditor nodes)."""
    out = hydrate_spec(spec)
    state = StrategyState(session_id="t", spec=spec, variants=out["variants"])
    state.structural_errors = structural_gate(state)
    if audit:
        state.audit_flags = audit_variants(state)
    return state


def _answers(state: StrategyState) -> dict[str, dict]:
    return {c["card_id"]: {"action": "accept"} for c in state.hitl_cards}


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def test_initial_state_defaults() -> None:
    state = get_initial_state("sid-1")
    assert state.session_id == "sid-1"
    assert state.status == "idle"
    assert state.round == 0
    assert state.max_rounds == 2
    assert state.iteration_count == 0
    assert state.resume_key == ""
    assert state.variants == []
    assert state.hitl_cards == []


def test_state_all_statuses() -> None:
    for status in ("idle", "auditing", "hitl", "complete", "rejected"):
        assert StrategyState(session_id="x", status=status).status == status


# ---------------------------------------------------------------------------
# Hydration + structural gate
# ---------------------------------------------------------------------------


def test_referenced_primitives_collects_vocabulary() -> None:
    spec = make_spec(
        filter_primitives=["TRIGGER_CROSS_BELOW"],
        parameters={
            "mode": "static",
            "indicators": ["IND_EMA", "IND_ATR"],
            "secondary_signal": "IND_VOLUME_DELTA",
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        },
        signal_ast=LOOKAHEAD_AST,
    )
    prims = referenced_primitives(spec)
    assert {
        "TRIGGER_CROSS_ABOVE",
        "TRIGGER_CROSS_BELOW",
        "IND_EMA",
        "IND_ATR",
        "IND_VOLUME_DELTA",
        "close",
    } <= prims


def test_data_transforms_are_not_unbacked() -> None:
    spec = make_spec(signal_ast=LOOKAHEAD_AST)
    assert unbacked_primitives(spec) == []


def test_unbacked_primitive_detected() -> None:
    spec = make_spec(
        parameters={
            "mode": "static",
            "indicators": ["IND_EMA", "IND_NOVEL_SIGNAL"],
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        }
    )
    assert unbacked_primitives(spec) == ["IND_NOVEL_SIGNAL"]


def test_hydrate_mechanical_spec_produces_triples() -> None:
    out = hydrate_spec(make_spec())
    assert 1 <= len(out["variants"]) <= MAX_VARIANTS
    for variant in out["variants"]:
        assert resolve(variant["entry"]["component_id"]) is not None
        assert resolve(variant["exit"]["component_id"]) is not None
        assert variant["sizing"] is None
        assert variant["data"]["symbol"] == "BTC-USD"
    assert [e["type"] for e in out["structural_errors"]] == ["missing_pillar"] * len(
        out["variants"]
    )


def test_hydrate_with_sizing_param_route_is_clean() -> None:
    out = hydrate_spec(with_sizing(make_spec()))
    assert out["structural_errors"] == []
    assert out["variants"][0]["sizing"]["component_id"] == "sizing.fractional"
    assert out["variants"][0]["sizing"]["params"]["fraction"] == 0.02


def test_hydrate_sweep_grid_has_bounds() -> None:
    out = hydrate_spec(with_sizing(make_spec()))
    for variant in out["variants"]:
        for pillar in ("entry", "exit", "sizing"):
            for name, bound in variant[pillar]["grid"].items():
                assert bound["lo"] < bound["hi"], f"{pillar}.{name} lo<hi"
                assert bound["step"] > 0
                assert bound["lo"] <= bound["default"] <= bound["hi"]


def test_hydrate_variant_count_is_deterministic_and_capped() -> None:
    # TRIGGER_CROSS_ABOVE matches entry.ema_cross + the entry fallback ->
    # exactly 2 options, capped by MAX_VARIANTS.
    out = hydrate_spec(make_spec())
    assert len(out["variants"]) == 2 <= MAX_VARIANTS
    ids = [v["entry"]["component_id"] for v in out["variants"]]
    assert ids == ["entry.ema_cross", "entry.vwap_band"]


def test_structural_gate_reports_bad_data() -> None:
    state = _formulated(make_spec(target_asset="", timeframe=""), audit=False)
    assert {"type": "bad_data", "pillar": "universe"} in [
        {k: e[k] for k in ("type", "pillar")} for e in state.structural_errors
    ]


def test_structural_gate_reports_unregistered_component() -> None:
    spec = with_sizing(make_spec())
    out = hydrate_spec(spec)
    out["variants"][0]["exit"] = {"component_id": "exit.nope", "params": {}, "grid": {}}
    state = StrategyState(session_id="t", spec=spec, variants=out["variants"])
    errors = structural_gate(state)
    assert any(
        e["type"] == "missing_pillar" and "exit.nope" in e["message"] for e in errors
    )


# ---------------------------------------------------------------------------
# Cynical Auditor
# ---------------------------------------------------------------------------


def test_auditor_no_sizing_flag() -> None:
    state = _formulated(make_spec())
    assert {f["type"] for f in state.audit_flags} == {"no_sizing"}


def test_auditor_degenerate_window_flag() -> None:
    spec = with_sizing(make_spec())
    out = hydrate_spec(spec)
    out["variants"][0]["entry"]["params"] = {"fast": 30, "slow": 10}
    state = StrategyState(session_id="t", spec=spec, variants=out["variants"])
    flags = audit_variants(state)
    assert {f["type"] for f in flags} == {"degenerate_window"}


def test_auditor_unbounded_param_flag() -> None:
    out = hydrate_spec(with_sizing(make_spec()))
    out["variants"][0]["entry"]["grid"] = {"k": {"default": 2.0}}  # no lo/hi
    state = StrategyState(
        session_id="t", spec=with_sizing(make_spec()), variants=out["variants"]
    )
    flags = audit_variants(state)
    assert any(f["type"] == "unbounded_param" and f["param"] == "k" for f in flags)


def test_auditor_lookahead_critical_flag() -> None:
    state = _formulated(with_sizing(make_spec(signal_ast=LOOKAHEAD_AST)))
    assert [f["type"] for f in state.audit_flags] == ["lookahead_bias"] * len(
        state.variants
    )
    assert all(f["severity"] == "critical" for f in state.audit_flags)


def test_auditor_clean_lag_is_not_lookahead() -> None:
    ast = {
        "kind": "binary_op",
        "op": "lag",
        "left": {"kind": "transform", "id": "close"},
        "right": {"kind": "literal", "id": "1"},
    }
    state = _formulated(with_sizing(make_spec(signal_ast=ast)))
    assert state.audit_flags == []


# ---------------------------------------------------------------------------
# HITL cards + apply_answers
# ---------------------------------------------------------------------------


def test_build_cards_missing_pillar() -> None:
    state = _formulated(make_spec())
    state.hitl_cards = build_cards(state)
    sizing_ids = [c.id for c in by_pillar("sizing")]
    assert len(state.hitl_cards) == len(state.variants)
    for card in state.hitl_cards:
        assert card["type"] == "missingPillar"
        assert card["pillar"] == "sizing"
        assert card["mutation"].endswith(".sizing")
        assert card["proposal"]["component_id"] in sizing_ids
        assert card["options"] == sizing_ids
    ids = [c["card_id"] for c in state.hitl_cards]
    assert len(set(ids)) == len(ids), "card ids must be batch-unique"
    assert ids[0].startswith("C-R0-")


def test_build_cards_universe() -> None:
    state = _formulated(make_spec(target_asset="", timeframe=""), audit=False)
    state.hitl_cards = build_cards(state)
    assert any(
        c["type"] == "universe" and c["mutation"] == "spec" for c in state.hitl_cards
    )


def test_build_cards_missing_primitive_preview(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("black_box.strategylib.synthesize.GENERATED_DIR", tmp_path)
    # A signal_ast the interpreter can actually evaluate (div of two data
    # transforms) -> the preview compiles and becomes approvable.
    ast = {
        "kind": "binary_op",
        "op": "div",
        "left": {"kind": "transform", "id": "close"},
        "right": {"kind": "transform", "id": "open"},
    }
    spec = make_spec(
        signal_ast=ast,
        parameters={
            "mode": "static",
            "indicators": ["IND_EMA", "IND_NOVELX"],
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        },
    )
    state = _formulated(spec, audit=False)
    state.hitl_cards = build_cards(state)
    card = next(c for c in state.hitl_cards if c["type"] == "missingPrimitive")
    assert card["pillar"] == "codebase"
    assert card["proposal"]["entry"]["id"] == "ind.novelx"
    assert card["options"] == ["synthesize", "abandon"]
    assert (tmp_path / "ind_novelx.py").exists(), (
        "preview module must be written for review"
    )


def test_build_cards_missing_primitive_without_ast_is_error_preview(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("black_box.strategylib.synthesize.GENERATED_DIR", tmp_path)
    # No signal_ast -> the fallback operand references an unknown feed and
    # the hermetic self-check fails: the card honestly shows the error and
    # is NOT approvable (nothing to register).
    spec = make_spec(
        parameters={
            "mode": "static",
            "indicators": ["IND_NOVELX"],
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        }
    )
    state = _formulated(spec, audit=False)
    state.hitl_cards = build_cards(state)
    card = next(c for c in state.hitl_cards if c["type"] == "missingPrimitive")
    assert "error" in card["proposal"]["entry"]


def test_apply_answers_accept_missing_pillar_fills_full_payload() -> None:
    state = _formulated(make_spec())
    state.hitl_cards = build_cards(state)
    state.user_responses = _answers(state)
    update = apply_answers(state)

    assert update["applied_any"] is True
    assert update["answers_applied"] == [c["card_id"] for c in state.hitl_cards]
    sizing = update["variants"][0]["sizing"]
    assert sizing["component_id"] in [c.id for c in by_pillar("sizing")]
    assert set(sizing) == {"component_id", "params", "grid"}  # full pillar payload
    assert update["resume_key"]  # idempotency key set
    assert (
        update["resume_key"]
        == hashlib.sha1(
            json.dumps(state.user_responses, sort_keys=True, default=str).encode()
        ).hexdigest()
    )


def test_apply_answers_override_universe() -> None:
    state = _formulated(make_spec(target_asset="", timeframe=""), audit=False)
    state.hitl_cards = build_cards(state)
    state.user_responses = {
        c["card_id"]: {
            "action": "override",
            "value": {"symbol": "ETH-USD", "timeframe": "1h"},
        }
        for c in state.hitl_cards
        if c["type"] == "universe"
    }
    update = apply_answers(state)
    assert update["spec"]["target_asset"] == "ETH-USD"
    assert update["spec"]["timeframe"] == "1h"


def test_apply_answers_is_idempotent_noop() -> None:
    state = _formulated(make_spec())
    state.hitl_cards = build_cards(state)
    state.user_responses = _answers(state)
    first = apply_answers(state)

    applied = StrategyState(
        session_id=state.session_id,
        spec=first.get("spec", state.spec),
        variants=first["variants"],
        structural_errors=state.structural_errors,
        audit_flags=state.audit_flags,
        hitl_cards=state.hitl_cards,
        answers_applied=first["answers_applied"],
        user_responses=state.user_responses,
    )
    second = apply_answers(applied)
    assert second.get("applied_any") is False
    assert "variants" not in second, "no churn on a duplicate batch"


def test_apply_answers_approve_registers_component(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("black_box.strategylib.synthesize.GENERATED_DIR", tmp_path)
    registered: list[dict] = []
    monkeypatch.setattr(
        "black_box.block_b.hitl.append_manifest", lambda entry: registered.append(entry)
    )

    ast = {
        "kind": "binary_op",
        "op": "div",
        "left": {"kind": "transform", "id": "close"},
        "right": {"kind": "transform", "id": "open"},
    }
    spec = make_spec(
        signal_ast=ast,
        parameters={
            "mode": "static",
            "indicators": ["IND_EMA", "IND_NOVELX"],
            "origin": "paper_literal",
            "non_negotiable_invariants": [],
        },
    )
    state = _formulated(spec, audit=False)
    state.hitl_cards = build_cards(state)
    card = next(c for c in state.hitl_cards if c["type"] == "missingPrimitive")
    state.user_responses = {card["card_id"]: {"action": "approve"}}
    apply_answers(state)
    assert [e["id"] for e in registered] == ["ind.novelx"]

    # abandon must NOT register
    registered.clear()
    card2 = next(c for c in state.hitl_cards if c["type"] == "missingPrimitive")
    state.user_responses = {card2["card_id"]: {"action": "abandon"}}
    apply_answers(state)
    assert registered == []


# ---------------------------------------------------------------------------
# Graph lifecycle (real interrupt/resume via MemorySaver)
# ---------------------------------------------------------------------------


def test_graph_complete_path_without_hitl() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "g-complete"}}
    initial = get_initial_state("g-complete")
    initial.spec = with_sizing(make_spec())
    graph.invoke(initial.model_dump(), cfg)
    state = state_from_checkpoint(graph.get_state(cfg))
    assert state.status == "complete"
    assert state.structural_errors == []
    assert state.audit_flags == []
    assert state.variants


def test_graph_hitl_then_resume_to_complete() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "g-hitl"}}
    initial = get_initial_state("g-hitl")
    initial.spec = make_spec()
    graph.invoke(initial.model_dump(), cfg)
    parked = state_from_checkpoint(graph.get_state(cfg))
    assert parked.status == "hitl"
    assert {c["type"] for c in parked.hitl_cards} == {"missingPillar"}
    assert parked.hitl_cards, "interrupt must expose cards"

    graph.invoke(Command(resume={"answers": _answers(parked)}), cfg)
    done = state_from_checkpoint(graph.get_state(cfg))
    assert done.status == "complete"
    assert done.round == 1
    assert set(done.variants[0]["sizing"]) == {"component_id", "params", "grid"}
    assert (
        done.resume_key
        == hashlib.sha1(
            json.dumps(_answers(parked), sort_keys=True, default=str).encode()
        ).hexdigest()
    )


def test_graph_lookahead_rejected_after_round_cap() -> None:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "g-lookahead"}}
    initial = get_initial_state("g-lookahead")
    initial.spec = with_sizing(make_spec(signal_ast=LOOKAHEAD_AST))
    graph.invoke(initial.model_dump(), cfg)

    for expected_round in (1, 2):
        parked = state_from_checkpoint(graph.get_state(cfg))
        assert parked.status == "hitl"
        assert parked.round == expected_round - 1
        graph.invoke(Command(resume={"answers": _answers(parked)}), cfg)

    done = state_from_checkpoint(graph.get_state(cfg))
    assert done.status == "rejected"
    assert done.round == 2


def test_replay_park_restores_thread_after_restart(tmp_path) -> None:
    # First process: run to the interrupt and snapshot to disk.
    g1 = build_graph()
    cfg1 = {"configurable": {"thread_id": "rp"}}
    initial = get_initial_state("rp")
    initial.spec = make_spec()
    g1.invoke(initial.model_dump(), cfg1)
    parked = state_from_checkpoint(g1.get_state(cfg1))
    assert parked.status == "hitl"
    save_state("rp", parked.model_dump(), tmp_path)

    # "Restart": fresh graph (empty MemorySaver) + disk snapshot replay.
    g2 = build_graph()
    replay_park(g2, parked.model_dump(), "rp")
    restored = state_from_checkpoint(
        g2.get_state({"configurable": {"thread_id": "rp"}})
    )
    assert restored.status == "hitl"
    assert restored.hitl_cards, "replayed thread must be parked with cards"
    assert restored.round == 0  # hitl_payload never increments round

    g2.invoke(Command(resume={"answers": _answers(restored)}), cfg1)
    done = state_from_checkpoint(g2.get_state(cfg1))
    assert done.status == "complete"


def test_save_load_state_roundtrip(tmp_path) -> None:
    state = _formulated(make_spec())
    path = save_state("roundtrip", state.model_dump(), tmp_path)
    assert path.exists()
    loaded = load_state("roundtrip", tmp_path)
    assert loaded is not None
    assert loaded["session_id"] == "t"
    assert load_state("missing", tmp_path) is None


# ---------------------------------------------------------------------------
# API contract (TestClient)
# ---------------------------------------------------------------------------


@pytest.fixture()
def strategy_out(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("black_box.api.strategy_routes._out_dir", lambda: tmp_path)


def _post_submit(client, body: dict, session_id: str | None = None) -> object:
    url = "/api/v1/strategy/submit"
    if session_id:
        url += f"?session_id={session_id}"
    return client.post(url, json=body)


def test_api_submit_hitl_then_resume(client, strategy_out) -> None:
    resp = _post_submit(client, {"spec": make_spec()})
    assert resp.status_code == 201
    payload = resp.json()
    assert payload["status"] == "hitl"
    assert payload["session_id"]
    assert len(payload["hitl_cards"]) >= 1

    sid = payload["session_id"]
    answers = {c["card_id"]: {"action": "accept"} for c in payload["hitl_cards"]}
    resumed = client.post(f"/api/v1/strategy/{sid}/resume", json={"answers": answers})
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "complete"

    state = client.get(f"/api/v1/strategy/{sid}/state")
    assert state.status_code == 200
    assert state.json()["status"] == "complete"


def test_api_submit_complete_path_without_hitl(client, strategy_out) -> None:
    resp = _post_submit(client, {"spec": with_sizing(make_spec())})
    assert resp.status_code == 201
    payload = resp.json()
    assert payload["status"] == "complete"
    assert payload["variants"][0]["sizing"]["params"]["fraction"] == 0.02


def test_api_resume_is_idempotent_per_batch(client, strategy_out) -> None:
    resp = _post_submit(client, {"spec": make_spec()})
    sid = resp.json()["session_id"]
    answers = {c["card_id"]: {"action": "accept"} for c in resp.json()["hitl_cards"]}

    first = client.post(f"/api/v1/strategy/{sid}/resume", json={"answers": answers})
    assert first.json()["status"] == "complete"

    # Identical batch after completion -> graceful no-op, not 409.
    second = client.post(f"/api/v1/strategy/{sid}/resume", json={"answers": answers})
    assert second.status_code == 200
    assert second.json()["status"] == "complete"


def test_api_submit_via_spec_id(tmp_path, client, monkeypatch, strategy_out) -> None:
    from black_box.core.config import get_settings

    monkeypatch.setattr(get_settings(), "block_a_out_dir", tmp_path)
    (tmp_path / "block_a_specs.json").write_text(
        json.dumps([with_sizing(make_spec(spec_id="STRAT_CURATED_T0"))]),
        encoding="utf-8",
    )
    resp = _post_submit(client, {"spec_id": "STRAT_CURATED_T0"})
    assert resp.status_code == 201
    assert resp.json()["status"] == "complete"

    missing = _post_submit(client, {"spec_id": "STRAT_GHOST"})
    assert missing.status_code == 404


def test_api_submit_requires_spec_or_id(client) -> None:
    resp = _post_submit(client, {})
    assert resp.status_code == 422


def test_api_submit_duplicate_session_conflicts(client, strategy_out) -> None:
    first = _post_submit(client, {"spec": make_spec()}, session_id="dup-session")
    assert first.status_code == 201
    again = _post_submit(client, {"spec": make_spec()}, session_id="dup-session")
    assert again.status_code == 409


def test_api_state_unknown_session_404(client, strategy_out) -> None:
    resp = client.get("/api/v1/strategy/ghost-session/state")
    assert resp.status_code == 404


def test_api_stream_emits_events(client, strategy_out) -> None:
    resp = _post_submit(client, {"spec": make_spec()})
    sid = resp.json()["session_id"]
    stream = client.get(f"/api/v1/strategy/{sid}/stream")
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    body = stream.text
    assert "event: formulation" in body
    assert "event: hitl_request" in body
