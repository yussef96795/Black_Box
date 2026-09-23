"""P4 — granular per-spec gatekeeper (D4: GatekeeperDecision {status, specs}).

Covers the interactive loop contracts: subset approval curates the emitted
array, risk-tag toggling re-validates and lands in the exported payload, and
the back-compat paths (`n` reject-all, `approve=False` veto, `--yes`) are
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from black_box.block_a.catalog import build_catalog
from black_box.block_a.gatekeeper import confirm_gatekeeper
from black_box.block_a.llm_evaluator import fake_evaluator
from black_box.block_a.models import (
    BlockAState,
    ExecutableStrategySpec,
    RiskTag,
    StrategyTier,
)
from black_box.block_a.workflow import BlockAEngine
from tests.block_a_defs import vwap_abstraction, vwap_annotations, vwap_extraction

CATALOG_ROWS = [
    {
        "asset_class": "Crypto",
        "symbol": "BTCUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ETHUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
]


def _spec(spec_id: str, tags: list[RiskTag] | None = None) -> ExecutableStrategySpec:
    return ExecutableStrategySpec(
        spec_id=spec_id,
        tier=StrategyTier.TIER_0_LITERAL,
        target_asset="BTCUSDT",
        timeframe="1m",
        entry_trigger_primitive="TRIGGER_CROSS_ABOVE",
        exit_trigger_primitive="TRIGGER_CROSS_BELOW",
        risk_annotations=tags or [],
    )


def _state(*specs: ExecutableStrategySpec) -> BlockAState:
    return BlockAState(
        paper_id="p",
        source_path="paper.md",
        status="gatekeeper",
        specs=[s.model_dump(mode="json") for s in specs],
    )


def _input_sequence(*answers: str):
    queue = list(answers)

    def fake_input(_prompt: str = "") -> str:
        return queue.pop(0) if queue else "r"

    return fake_input


def test_per_spec_subset_approval_curates_specs(monkeypatch) -> None:
    """a/r/a over three specs → decision carries only the approved two."""
    import builtins

    monkeypatch.setattr(builtins, "input", _input_sequence("a", "r", "a"))
    decision = confirm_gatekeeper(_state(_spec("S1"), _spec("S2"), _spec("S3")))
    assert decision.status == "complete"
    assert [s.spec_id for s in decision.specs] == ["S1", "S3"]


def test_first_prompt_no_rejects_all(monkeypatch) -> None:
    """Back-compat: a bare `n` on the first spec rejects the whole run."""
    import builtins

    monkeypatch.setattr(builtins, "input", _input_sequence("n"))
    decision = confirm_gatekeeper(_state(_spec("S1"), _spec("S2")))
    assert decision.status == "rejected"
    assert decision.specs == []


def test_yes_approves_all_remaining(monkeypatch) -> None:
    """Back-compat: `y` approves the current spec and every remaining one."""
    import builtins

    monkeypatch.setattr(builtins, "input", _input_sequence("y"))
    decision = confirm_gatekeeper(_state(_spec("S1"), _spec("S2"), _spec("S3")))
    assert decision.status == "complete"
    assert [s.spec_id for s in decision.specs] == ["S1", "S2", "S3"]


def test_toggle_risk_tags_revalidates(monkeypatch) -> None:
    """`t` then an index toggles the tag; the spec is re-validated."""
    import builtins

    # menu order is the RiskTag enum definition: 4 == EXECUTION_SLIPPAGE_HEAVY
    spec = _spec("S1", tags=[RiskTag.EXECUTION_SLIPPAGE_HEAVY])
    monkeypatch.setattr(builtins, "input", _input_sequence("t", "4", "a"))
    decision = confirm_gatekeeper(_state(spec))
    assert decision.status == "complete"
    edited = decision.specs[0]
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY not in edited.risk_annotations


def test_toggle_off_adds_tag_back(monkeypatch) -> None:
    """Toggling an off tag turns it on; enum cap is structural (≤4)."""
    import builtins

    spec = _spec("S1")  # no tags: 1 → HIGH_SESSION_SENSITIVITY turns on
    monkeypatch.setattr(builtins, "input", _input_sequence("t", "1", "a"))
    decision = confirm_gatekeeper(_state(spec))
    assert decision.specs[0].risk_annotations == [RiskTag.HIGH_SESSION_SENSITIVITY]


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    return build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)


def _complete_engine(catalog: Path) -> BlockAEngine:
    evaluator, _ = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(slippage=True),
    )
    return BlockAEngine(
        evaluator=evaluator,
        parser=lambda _p: [
            {
                "id": "c0",
                "text": "synthetic",
                "heading": None,
                "page": None,
                "tokens": 1,
            }
        ],
        catalog_path=catalog,
    )


def test_run_curated_specs_only_in_export(
    tmp_path: Path, catalog: Path, monkeypatch
) -> None:
    """Approve 2, reject the rest → file contains exactly the approved specs."""
    import builtins

    # enough confirmations for the compiled matrix (≤5 specs); fallback "r"
    monkeypatch.setattr(builtins, "input", _input_sequence("a", "a"))
    out = tmp_path / "out"
    result = _complete_engine(catalog).run("paper.md", approve=None, out_dir=out)

    assert result.status == "complete"
    assert len(result.specs) == 2
    approved_ids = {s.spec_id for s in result.specs}
    assert len(approved_ids) == 2

    payload = json.loads((out / "block_a_specs.json").read_text(encoding="utf-8"))
    assert {s["spec_id"] for s in payload} == approved_ids
    # risk_tags recomputed from the curated array only
    assert result.risk_tags == sorted(
        {t for s in result.specs for t in s.risk_annotations}
    )


def test_toggle_reflected_in_emitted_payload(
    tmp_path: Path, catalog: Path, monkeypatch
) -> None:
    """Toggle a tag off on the first spec; the edit lands in the export."""
    import builtins

    # first spec: t → toggles index 4 (EXECUTION_SLIPPAGE_HEAVY) → approve;
    # second spec: approve untouched; rest rejected
    monkeypatch.setattr(builtins, "input", _input_sequence("t", "4", "a", "a"))
    out = tmp_path / "out"
    result = _complete_engine(catalog).run("paper.md", approve=None, out_dir=out)

    assert result.status == "complete"
    assert len(result.specs) == 2
    payload = json.loads((out / "block_a_specs.json").read_text(encoding="utf-8"))
    assert {s["spec_id"] for s in payload} == {s.spec_id for s in result.specs}
    first, second = payload
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY.value not in first["risk_annotations"]
    # the untouched approved spec keeps the original union of risk tags
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY.value in second["risk_annotations"]
