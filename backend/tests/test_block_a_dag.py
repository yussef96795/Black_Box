"""Tests for the Block A LangGraph DAG (workflow.py) + stateless isolation."""

from __future__ import annotations

import json
from pathlib import Path

from instructor.core import InstructorRetryException
from pydantic import ValidationError

from black_box.block_a.catalog import build_catalog
from black_box.block_a.llm_evaluator import (
    FakeInstructorClient,
    LLMEvaluator,
    fake_evaluator,
)
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


class _FailingStageClient(FakeInstructorClient):
    """Fake client that exhausts retries on exactly one stage (P2 error path)."""

    def __init__(self, responses: dict, fail_on: type) -> None:
        super().__init__(responses)
        self.fail_on = fail_on

    def create(self, response_model, messages, max_retries=3, **kwargs):
        if response_model is self.fail_on:
            if self.fail_on is OperatorAnnotations:
                raise InstructorRetryException(
                    "garbage output after 3 retries",
                    n_attempts=3,
                    total_usage={"total_tokens": 128},
                )
            raise ValidationError.from_exception_data(
                "PaperExtractionSchema",
                [
                    {
                        "type": "value_error",
                        "loc": ("x",),
                        "msg": "boom",
                        "input": None,
                        "ctx": {"error": ValueError("boom")},
                    }
                ],
            )
        return super().create(response_model, messages, max_retries, **kwargs)


def _failing_evaluator(fail_on: type):
    client = _FailingStageClient(
        {
            PaperExtractionSchema: vwap_extraction(),
            CausalAbstractionSchema: vwap_abstraction(),
        },
        fail_on=fail_on,
    )
    return LLMEvaluator(client)


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


def test_dag_check_resources_schema_drift_is_structured(tmp_path, monkeypatch) -> None:
    """Extraction that fails re-validation inside check_resources becomes a
    structured terminal failure (stage_error recorded), not an unhandled
    crash — and does NOT get overwritten by a resource verdict (P1.2)."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    evaluator, _ = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(slippage=True),
    )
    real_validate = PaperExtractionSchema.model_validate

    def drift_on_dump(obj):
        # check_resources re-validates the JSON-dumped extraction (a dict);
        # the fake client hands over already-validated model instances.
        if isinstance(obj, dict):
            raise ValidationError.from_exception_data(
                "PaperExtractionSchema",
                [
                    {
                        "type": "value_error",
                        "loc": ("core_mechanism",),
                        "msg": "schema drift",
                        "input": None,
                        "ctx": {"error": ValueError("schema drift")},
                    }
                ],
            )
        return real_validate(obj)

    monkeypatch.setattr(
        PaperExtractionSchema, "model_validate", staticmethod(drift_on_dump)
    )
    graph = _graph(evaluator, catalog)
    final = BlockAState.model_validate(
        graph.invoke(
            BlockAState(paper_id="p3", source_path="paper.md"),
            config={"configurable": {"thread_id": "p3"}},
        )
    )
    assert final.status == "llm_extraction_failed"
    assert final.stage_error["stage"] == "check_resources"
    assert final.stage_error["error_type"] == "ValidationError"
    assert final.specs == []


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


# --- P2: graceful LLM failure / DAG error states ----------------------------


def test_dag_llm_failure_on_a5_ends_terminal_and_preserves_partials(
    tmp_path: Path,
) -> None:
    """InstructorRetryException on Stage A5 → llm_extraction_failed, partials kept."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    graph = _graph(_failing_evaluator(OperatorAnnotations), catalog)
    final = BlockAState.model_validate(
        graph.invoke(
            BlockAState(paper_id="p5", source_path="paper.md"),
            config={"configurable": {"thread_id": "p5"}},
        )
    )
    assert final.status == "llm_extraction_failed"
    # Successfully completed stages are preserved — nothing dropped.
    assert final.extraction is not None
    assert final.abstraction is not None
    assert final.annotations == []
    assert final.specs == []
    assert final.stage_error is not None
    assert final.stage_error["stage"] == "apply_operators"
    assert final.stage_error["error_type"] == "InstructorRetryException"
    assert "3 retries" in final.stage_error["detail"]
    assert len(final.stage_error["trace_hash"]) == 16


def test_dag_llm_failure_on_a1_ends_terminal_early() -> None:
    """Failure at the FIRST LLM stage still terminates with structured error."""
    # catalog_path omitted: the A1 failure ends the run before Stage A3.
    graph = _graph(_failing_evaluator(PaperExtractionSchema), None)
    final = BlockAState.model_validate(
        graph.invoke(
            BlockAState(paper_id="p6", source_path="paper.md"),
            config={"configurable": {"thread_id": "p6"}},
        )
    )
    assert final.status == "llm_extraction_failed"
    assert final.stage_error["stage"] == "extract_requirements"
    assert final.stage_error["error_type"] == "ValidationError"
    assert final.extraction is None
    assert final.specs == []


def test_engine_llm_failure_result_structured_error_and_no_export(
    tmp_path: Path,
) -> None:
    """BlockAResult carries {stage, error_type, detail, trace_hash}; no export."""
    catalog = build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)
    engine = BlockAEngine(
        evaluator=_failing_evaluator(OperatorAnnotations),
        parser=lambda _p: _synthetic_chunks("x"),
        catalog_path=catalog,
    )
    out = tmp_path / "out_llm"
    result = engine.run("paper.md", approve=True, out_dir=out)
    assert result.status == "llm_extraction_failed"
    assert result.error is not None
    assert result.error["stage"] == "apply_operators"
    assert result.error["error_type"] == "InstructorRetryException"
    assert set(result.error) == {"stage", "error_type", "detail", "trace_hash"}
    # extraction succeeded, so the title is preserved in the result too
    assert result.paper_title != ""
    assert result.out_path is None
    assert not (out / "block_a_specs.json").exists()


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
