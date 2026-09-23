"""Tests for the Block A LangGraph DAG (workflow.py) + stateless isolation."""

from __future__ import annotations

import json
from pathlib import Path

from black_box.block_a.catalog import build_catalog
from black_box.block_a.llm_evaluator import fake_evaluator
from black_box.block_a.models import (
    RESOURCE_INSUFFICIENT_ERROR,
    BlockAState,
    CausalAbstractionSchema,
    ExecutableStrategySpec,
    OperatorAnnotations,
    PaperExtractionSchema,
    RiskTag,
    StrategyTier,
)
from black_box.block_a.workflow import BlockAEngine, build_block_a_graph
from tests.block_a_defs import (
    qqq_extraction,
    vwap_abstraction,
    vwap_annotations,
    vwap_extraction,
)

FIXTURES = Path(__file__).parent / "fixtures" / "block_a"
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


def _synthetic_chunks(text: str) -> list[dict]:
    return [{"id": "c0", "text": text, "heading": None, "page": None, "tokens": 5}]


def _graph(evaluator, catalog: Path, approve: bool = True):
    return build_block_a_graph(
        evaluator=evaluator,
        parser=lambda _p: _synthetic_chunks("synthetic paper body"),
        catalog_path=catalog,
        approve=approve,
    )


def test_dag_complete_path(tmp_path: Path) -> None:
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    evaluator, client = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(slippage=True),
    )
    graph = _graph(evaluator, catalog)
    final = BlockAState.model_validate(
        graph.invoke(
            BlockAState(paper_id="p1", source_path="paper.md"),
            config={"configurable": {"thread_id": "p1"}},
        )
    )
    assert final.status == "complete"
    assert final.resource_check is not None
    assert final.resource_check["all_available"] is True
    assert final.abstraction is not None
    assert len(final.annotations) == 8
    assert 3 <= len(final.specs) <= 5
    # zero raw text: every spec revalidates through the export contract
    specs = [ExecutableStrategySpec.model_validate(s) for s in final.specs]
    assert all(s.is_testable for s in specs)
    # exactly three structured LLM stages — no hidden calls
    assert client.calls == [
        PaperExtractionSchema,
        CausalAbstractionSchema,
        OperatorAnnotations,
    ]


def test_dag_hard_stop_resource_insufficient(tmp_path: Path) -> None:
    """QQQ (equity, not in crypto catalog) → hard stop, zero specs, error code."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    evaluator, _ = fake_evaluator(
        extraction=qqq_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(),
    )
    graph = _graph(evaluator, catalog)
    final = BlockAState.model_validate(
        graph.invoke(
            BlockAState(paper_id="p2", source_path="qqq.md"),
            config={"configurable": {"thread_id": "p2"}},
        )
    )
    assert final.status == "resource_insufficient"
    assert final.error_code == RESOURCE_INSUFFICIENT_ERROR
    assert final.specs == []
    assert final.abstraction is None  # causal stages never ran


def test_dag_vacuous_empty_datasets_passes_without_specs(tmp_path: Path) -> None:
    """No datasets_used → no verdicts → compiler emits nothing, no hard stop."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    extraction = vwap_extraction()
    extraction.datasets_used = []
    evaluator, _ = fake_evaluator(
        extraction=extraction,
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(),
    )
    final = BlockAState.model_validate(
        _graph(evaluator, catalog).invoke(
            BlockAState(paper_id="p3", source_path="x.md"),
            config={"configurable": {"thread_id": "p3"}},
        )
    )
    assert final.status == "complete"
    assert final.specs == []


def test_run_engine_stateless_no_cross_paper_leak(tmp_path: Path) -> None:
    """Two sequential runs: outputs are independent, no shared state."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)

    evaluator_a, client_a = fake_evaluator(
        extraction=vwap_extraction("BTCUSDT"),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(),
    )
    engine_a = BlockAEngine(
        evaluator=evaluator_a,
        parser=lambda _p: _synthetic_chunks("MTAG-A7G3k BTCUSDT vwap paper"),
        catalog_path=catalog,
    )
    result_a = engine_a.run("paper_a.md", approve=True, out_dir=tmp_path / "out_a")

    evaluator_b, client_b = fake_evaluator(
        extraction=vwap_extraction("ETHUSDT"),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(),
    )
    engine_b = BlockAEngine(
        evaluator=evaluator_b,
        parser=lambda _p: _synthetic_chunks("MTAG-B9Q2m ETHUSDT vwap paper"),
        catalog_path=catalog,
    )
    result_b = engine_b.run("paper_b.md", approve=True, out_dir=tmp_path / "out_b")

    assert result_a.status == "complete" and result_b.status == "complete"
    # Tier 0 (literal) always mirrors its own paper's anchor — and nothing
    # from paper A leaks into paper B's outputs.
    t0_a = next(s for s in result_a.specs if s.tier == StrategyTier.TIER_0_LITERAL)
    t0_b = next(s for s in result_b.specs if s.tier == StrategyTier.TIER_0_LITERAL)
    assert t0_a.target_asset == "BTCUSDT"
    assert t0_b.target_asset == "ETHUSDT"
    # (Tier 2 of B legitimately generalizes to BTCUSDT from the catalog —
    # that is deterministic diversification, not state leakage.)
    # each run performed exactly 3 fresh LLM calls — no shared conversation
    assert client_a.calls == [
        PaperExtractionSchema,
        CausalAbstractionSchema,
        OperatorAnnotations,
    ]
    assert client_b.calls == [
        PaperExtractionSchema,
        CausalAbstractionSchema,
        OperatorAnnotations,
    ]
    # paper B's first prompt never carries paper A's chunk marker
    assert "MTAG-B9Q2m" in client_b.recorded_prompts[0]
    assert "MTAG-A7G3k" not in client_b.recorded_prompts[0]
    assert "MTAG-A7G3k" in client_a.recorded_prompts[0]


def test_engine_rejected_when_gatekeeper_vetoes(tmp_path: Path) -> None:
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    evaluator, _ = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(),
    )
    engine = BlockAEngine(
        evaluator=evaluator,
        parser=lambda _p: _synthetic_chunks("x"),
        catalog_path=catalog,
    )
    result = engine.run("paper.md", approve=False, out_dir=tmp_path / "out_r")
    assert result.status == "rejected"
    assert result.out_path is None  # nothing emitted


def test_engine_integration_real_docling(tmp_path: Path) -> None:
    """End-to-end against the real Docling parser + export file."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    evaluator, _ = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(slippage=True),
    )
    engine = BlockAEngine(evaluator=evaluator, catalog_path=catalog)
    out = tmp_path / "out_int"
    result = engine.run(FIXTURES / "vwap_trend.md", approve=True, out_dir=out)

    assert result.status == "complete"
    assert result.out_path is not None
    assert (out / "block_a_specs.json").exists()
    payload = json.loads((out / "block_a_specs.json").read_text(encoding="utf-8"))
    specs = [ExecutableStrategySpec.model_validate(o) for o in payload]
    assert 3 <= len(specs) <= 5
    assert any(s.tier == StrategyTier.TIER_2_GENERALIZED for s in specs)
    assert all(s.is_testable for s in specs)
    tags = {t for s in specs for t in s.risk_annotations}
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY in tags  # annotations flow through
