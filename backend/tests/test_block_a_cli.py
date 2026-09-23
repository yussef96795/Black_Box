"""CLI tests — `python -m black_box.block_a.cli` (Module 5 entry point)."""

from __future__ import annotations

import json
from pathlib import Path

from instructor.core import InstructorRetryException

from black_box.block_a.cli import (
    EXIT_FAILURE,
    EXIT_LLM_FAILED,
    EXIT_OK,
    EXIT_UNPARSEABLE,
    main,
)
from black_box.block_a.models import ExecutableStrategySpec

FIXTURES = Path(__file__).parent / "fixtures" / "block_a"


def test_cli_complete_path(tmp_path: Path) -> None:
    out = tmp_path / "out"
    paper = FIXTURES / "vwap_trend.md"
    code = main([str(paper), "--llm", "fake", "--yes", "--out", str(out)])
    assert code == EXIT_OK
    target = out / "block_a_specs.json"
    assert target.exists()
    specs = [
        ExecutableStrategySpec.model_validate(o)
        for o in json.loads(target.read_text(encoding="utf-8"))
    ]
    assert 3 <= len(specs) <= 5
    assert all(s.is_testable for s in specs)


def test_cli_rejects_without_yes_emits_nothing(tmp_path: Path, monkeypatch) -> None:
    """Gatekeeper veto (interactive answer 'n') writes no spec file."""
    import builtins

    out = tmp_path / "out"
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "n")
    code = main([str(FIXTURES / "vwap_trend.md"), "--llm", "fake", "--out", str(out)])
    assert code == EXIT_OK  # rejection is a decision, not a failure
    assert not (out / "block_a_specs.json").exists()


def test_cli_unparseable_paper(tmp_path: Path) -> None:
    garbage = tmp_path / "fake.pdf"
    garbage.write_bytes(b"%PDF-1.4 definitely not a real pdf")
    code = main([str(garbage), "--llm", "fake", "--yes", "--out", str(tmp_path / "o")])
    assert code == EXIT_UNPARSEABLE


def test_cli_missing_paper() -> None:
    code = main(["/nonexistent/paper.pdf", "--llm", "fake", "--yes"])
    assert code == EXIT_FAILURE


def test_cli_llm_failure_exits_4_and_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    """Stage A5 exhausts its validation retries → exit 4, no spec file."""
    from black_box.block_a import cli
    from black_box.block_a.llm_evaluator import FakeInstructorClient, LLMEvaluator
    from black_box.block_a.models import (
        CausalAbstractionSchema,
        OperatorAnnotations,
        PaperExtractionSchema,
    )
    from tests.block_a_defs import vwap_abstraction, vwap_extraction

    class _FailingA5(FakeInstructorClient):
        def create(self, response_model, messages, max_retries=3, **kwargs):
            if response_model is OperatorAnnotations:
                raise InstructorRetryException(
                    "garbage output after 3 retries",
                    n_attempts=3,
                    total_usage={"total_tokens": 128},
                )
            return super().create(response_model, messages, max_retries, **kwargs)

    evaluator = LLMEvaluator(
        _FailingA5(
            {
                PaperExtractionSchema: vwap_extraction(),
                CausalAbstractionSchema: vwap_abstraction(),
            }
        )
    )
    monkeypatch.setattr(cli, "_fake_evaluator", lambda _stem: evaluator)

    out = tmp_path / "out"
    code = main([str(FIXTURES / "vwap_trend.md"), "--llm", "fake", "--out", str(out)])
    assert code == EXIT_LLM_FAILED
    assert not (out / "block_a_specs.json").exists()
