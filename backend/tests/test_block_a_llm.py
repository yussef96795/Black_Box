"""Tests for Block A Stages A1–A5 — Instructor evaluator + retry contract."""

from __future__ import annotations

import json

import httpx
import pytest
from instructor.v2.core.errors import InstructorRetryException

from black_box.block_a.llm_evaluator import (
    fake_evaluator,
    openai_compat_client,
)
from black_box.block_a.models import (
    OperatorAnnotations,
    PaperExtractionSchema,
    RiskTag,
)
from tests.block_a_defs import vwap_abstraction, vwap_annotations, vwap_extraction


def _chunks(texts: list[str]) -> list[dict]:
    return [
        {"id": f"c{i}", "text": t, "heading": f"[H{i}]", "page": None, "tokens": 10}
        for i, t in enumerate(texts)
    ]


def test_fake_evaluator_extract_paper() -> None:
    canned = vwap_extraction()
    evaluator, client = fake_evaluator(extraction=canned)
    result = evaluator.extract_paper(_chunks(["some paper text"]))
    assert result == canned
    assert client.calls == [PaperExtractionSchema]


def test_fake_evaluator_apply_operators_full_sweep() -> None:
    annotations = vwap_annotations()
    evaluator, client = fake_evaluator(
        extraction=vwap_extraction(),
        abstraction=vwap_abstraction(),
        annotations=annotations,
    )
    result = evaluator.apply_operators(
        vwap_extraction(), vwap_abstraction(), _chunks(["x"])
    )
    assert len(result) == 8
    assert {a.operator_name for a in result} == {
        "Restrict",
        "Invert",
        "Relax",
        "Substitute",
        "Decompose",
        "Combine",
        "Adversarial",
        "Generalize",
    }
    assert client.calls == [OperatorAnnotations]


def test_fake_client_missing_response_raises() -> None:
    evaluator, _ = fake_evaluator()  # no canned responses at all
    with pytest.raises(KeyError):
        evaluator.extract_paper(_chunks(["x"]))


def test_operator_prompt_lists_all_eight() -> None:
    # The prompt must enumerate all 8 operators (schema-first enforcement
    # of the full sweep is via min_length=8 + Literal operator names).
    from black_box.block_a.operators import operator_prompt_block

    prompt_text = operator_prompt_block()
    for name in (
        "Restrict",
        "Invert",
        "Relax",
        "Substitute",
        "Decompose",
        "Combine",
        "Adversarial",
        "Generalize",
    ):
        assert name in prompt_text


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_instructor_retry_loop_recovers_after_validation_errors() -> None:
    """Garbage text twice, valid JSON on the 3rd attempt → success (max_retries)."""
    from pydantic import BaseModel

    class Ping(BaseModel):
        ok: bool
        name: str

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        content = (
            "not json at all"
            if calls["n"] < 3
            else json.dumps({"ok": True, "name": "pong"})
        )
        return httpx.Response(200, json=_completion(content), request=request)

    evaluator = openai_compat_client(
        base_url="http://mock/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = evaluator._create(Ping, "ping")
    assert result.ok is True and result.name == "pong"
    assert calls["n"] == 3  # 2 failures + 1 success within the retry budget


def test_instructor_retry_exhaustion_raises() -> None:
    """Persistent garbage → InstructorRetryException after attempts <= retries."""
    from pydantic import BaseModel

    class Ping(BaseModel):
        ok: bool
        name: str

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_completion("garbage forever"), request=request)

    evaluator = openai_compat_client(
        base_url="http://mock/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
    )
    with pytest.raises(InstructorRetryException):
        evaluator._create(Ping, "ping")
    assert calls["n"] == 3  # attempt + 2 retries


def test_statelessness_no_cross_paper_memory() -> None:
    """Fresh message per call; paper B never sees paper A's chunks."""
    canned_a = vwap_extraction("BTCUSDT")
    evaluator, client = fake_evaluator(extraction=canned_a)

    chunks_a = _chunks(["unique marker A-7G3k VWAP anchor paper"])

    chunks_b = _chunks(["completely different ETH paper marker B-9Q2m"])
    evaluator.extract_paper(chunks_a)
    evaluator.extract_paper(chunks_b)

    # FakeInstructorClient records prompts so we can prove isolation:
    assert len(client.recorded_prompts) == 2
    prompt_a, prompt_b = client.recorded_prompts
    assert "A-7G3k" in prompt_a
    assert "A-7G3k" not in prompt_b  # paper A never leaks into paper B's prompt
    assert "B-9Q2m" in prompt_b


def test_openai_compat_client_factory() -> None:
    evaluator = openai_compat_client(
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_completion("{}"), request=r)
            )
        ),
    )
    assert evaluator.model == "mock"
    assert evaluator.max_retries == 3


def test_annotations_risk_tags_typed() -> None:
    annotations = vwap_annotations(slippage=True)
    adversarial = next(a for a in annotations if a.operator_name == "Adversarial")
    assert RiskTag.EXECUTION_SLIPPAGE_HEAVY in adversarial.risk_tags
    assert RiskTag.LOW_LIQUIDITY_FRAGILITY in adversarial.risk_tags
