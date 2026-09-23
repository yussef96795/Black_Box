"""P3 — live prompt/response observability (JSONL traces, D3).

Hermetic by construction: tracing defaults to OFF, so the disabled test
asserts nothing appears on disk; the enabled tests write into `tmp_path`.
Settings are lru-cached process-wide, so each test clears the cache around
its env mutation and re-clears in a `finally` (monkeypatch restores the env
after the body, where the cache is already invalidated) — no settings state
leaks between tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from instructor.core import InstructorRetryException

from black_box.block_a.catalog import build_catalog
from black_box.block_a.llm_evaluator import (
    FakeInstructorClient,
    LLMEvaluator,
    fake_evaluator,
)
from black_box.block_a.models import (
    CausalAbstractionSchema,
    OperatorAnnotations,
    PaperExtractionSchema,
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
    }
]


@pytest.fixture()
def catalog(tmp_path: Path) -> Path:
    return build_catalog(tmp_path / "catalog.parquet", rows=CATALOG_ROWS)


def _synthetic_chunks(text: str) -> list[dict]:
    return [{"id": "c0", "text": text, "heading": None, "page": None, "tokens": 5}]


def _settings():
    from black_box.core.config import get_settings

    return get_settings


@pytest.fixture()
def trace_enabled(monkeypatch) -> None:
    """Turn tracing on for one test, isolated from the cached settings."""
    monkeypatch.setenv("BLOCK_A_TRACE_ENABLED", "1")
    _settings().cache_clear()
    yield
    _settings().cache_clear()


@pytest.fixture()
def trace_disabled(monkeypatch) -> None:
    """Force tracing off (default), isolated from any prior cache state."""
    monkeypatch.delenv("BLOCK_A_TRACE_ENABLED", raising=False)
    _settings().cache_clear()
    yield
    _settings().cache_clear()


def _complete_engine(catalog: Path, text: str = "VWAP paper body") -> BlockAEngine:
    evaluator, _ = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=vwap_annotations(slippage=True),
    )
    return BlockAEngine(
        evaluator=evaluator,
        parser=lambda _p: _synthetic_chunks(text),
        catalog_path=catalog,
    )


class _FailingA5Client(FakeInstructorClient):
    """Canned A1/A4, A5 always exhausts its validation retries."""

    def create(self, response_model, messages, max_retries=3, **kwargs):
        if response_model is OperatorAnnotations:
            raise InstructorRetryException(
                "garbage output after 3 retries",
                n_attempts=3,
                total_usage={"total_tokens": 128},
            )
        return super().create(response_model, messages, max_retries, **kwargs)


def _read_frames(trace_dir: Path) -> list[dict]:
    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1
    return [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_trace_enabled_writes_jsonl_keyed_by_paper_id(
    tmp_path: Path, catalog: Path, trace_enabled: None
) -> None:
    out = tmp_path / "out"
    result = _complete_engine(catalog).run("paper.md", approve=True, out_dir=out)

    trace_dir = out / "traces"
    assert trace_dir.exists()
    files = list(trace_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert files[0].name == f"{result.paper_id}.jsonl"

    frames = _read_frames(trace_dir)
    requests = [f for f in frames if f["event"] == "request"]
    responses = [f for f in frames if f["event"] == "response"]
    # each of the three LLM stages emits a request + response frame
    assert {f["stage"] for f in requests} == {
        "extract_requirements",
        "abstract_mechanism",
        "apply_operators",
    }
    assert len(requests) == 3
    assert len(responses) == 3
    assert not [f for f in frames if f["event"] == "error"]
    for frame in frames:
        assert frame["paper_id"] == result.paper_id

    req = next(f for f in requests if f["stage"] == "extract_requirements")
    assert req["prompt_chars"] > 0
    assert len(req["prompt_hash"]) == 16
    resp = next(f for f in responses if f["stage"] == "apply_operators")
    # fake client has no event hooks → zero raw completions observed
    assert resp["attempts"] == 0
    assert resp["final_model"]["annotations"]


def test_trace_disabled_creates_no_directory(
    tmp_path: Path, catalog: Path, trace_disabled: None
) -> None:
    out = tmp_path / "out"
    _complete_engine(catalog).run("paper.md", approve=True, out_dir=out)
    assert not (out / "traces").exists()


def test_trace_error_frame_on_llm_failure(
    tmp_path: Path, catalog: Path, trace_enabled: None
) -> None:
    """P2×P3: an exhausted-retry stage writes an `error` frame with the trace."""
    evaluator = LLMEvaluator(
        _FailingA5Client(
            {
                PaperExtractionSchema: vwap_extraction(),
                CausalAbstractionSchema: vwap_abstraction(),
            }
        )
    )
    out = tmp_path / "out"
    result = BlockAEngine(
        evaluator=evaluator,
        parser=lambda _p: _synthetic_chunks("x"),
        catalog_path=catalog,
    ).run("paper.md", approve=True, out_dir=out)

    assert result.status == "llm_extraction_failed"
    frames = _read_frames(out / "traces")
    error_frames = [f for f in frames if f["event"] == "error"]
    assert len(error_frames) == 1
    err = error_frames[0]
    assert err["stage"] == "apply_operators"
    assert err["error_type"] == "InstructorRetryException"
    assert "3 retries" in err["detail"]
    assert "Traceback" in err["trace"]
