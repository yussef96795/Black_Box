"""Stage A1–A5 — stateless Instructor-backed LLM evaluator.

Strictly schema-bound: every call returns a validated Pydantic model
(`PaperExtractionSchema`, `CausalAbstractionSchema`, `OperatorAnnotations`).
Unstructured text outputs are disallowed (spec §1 invariant 2).

Statelessness (spec §1 invariant 1): each `create` builds a fresh message
list from the current paper's chunks only — zero chat memory, no cross-paper
state. The client is rebuilt per run by the DAG factory.

Validation retries (spec §6 item 3): `max_retries` (default 3) is delegated
to Instructor's built-in retry loop, which re-prompts with the Pydantic
validation error until the response validates.

Backends:
    * ollama          — local Ollama via `instructor.from_provider`
                        (`Mode.JSON`, model from `Settings.ollama_model`).
    * openai_compat   — any OpenAI-compatible endpoint
                        (`instructor.from_openai`), used by tests against a
                        mocked HTTP transport.
    * fake            — `FakeInstructorClient` with canned models (CI / fast
                        tests / `--llm fake`). No network.
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from datetime import UTC, datetime
from typing import Any, Protocol

from black_box.block_a.models import (
    CausalAbstractionSchema,
    OperatorAnnotations,
    PaperExtractionSchema,
    StrategyAnnotation,
)
from black_box.block_a.operators import operator_prompt_block
from black_box.block_a.traces import TraceWriter, json_safe
from black_box.core.config import get_settings

logger = logging.getLogger(__name__)

#: Chunk text cap per prompt so local models stay within context cheaply.
MAX_CHUNK_CHARS = 12_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StructuredClient(Protocol):
    """Duck-typed Instructor client surface the evaluator depends on."""

    def create(
        self,
        response_model: type[Any],
        messages: list[dict[str, str]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def ollama_client(
    model: str | None = None, base_url: str | None = None
) -> StructuredClient:
    """Instructor client against local Ollama (Mode.JSON_SCHEMA, model embedded).

    JSON_SCHEMA hands Ollama the actual Pydantic schema as structured output,
    which measurably improves compliance on small local models (llama3.2 3B
    double-encodes JSON arrays under plain Mode.JSON). Retry contract lives on
    `create` (`max_retries`), never on the factory.
    """
    import instructor  # heavy import — keep at call site

    settings = get_settings()
    model = model or settings.block_a_ollama_model
    base_url = base_url or settings.block_a_ollama_base_url
    kwargs: dict[str, Any] = {"mode": instructor.Mode.JSON_SCHEMA}
    if base_url:
        kwargs["base_url"] = base_url
    return instructor.from_provider(f"ollama/{model}", **kwargs)


def openai_compat_client(
    *,
    base_url: str,
    http_client: Any,
    api_key: str = "test",
    model: str = "mock",
    max_retries: int = 3,
) -> LLMEvaluator:
    """Evaluator bound to an OpenAI-compatible endpoint (tests / enterprise).

    `http_client` is an `httpx.Client` (e.g. `MockTransport`) so no real
    network is required — this is how the Instructor retry contract is tested
    hermetically.
    """
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(
        OpenAI(base_url=base_url, api_key=api_key, http_client=http_client),
        mode=instructor.Mode.JSON,
    )
    return LLMEvaluator(client, model=model, max_retries=max_retries)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LLMEvaluator:
    """One stateless entry point for the three Block A LLM stages."""

    def __init__(
        self,
        client: StructuredClient,
        *,
        model: str | None = None,
        max_retries: int = 3,
        trace: TraceWriter | None = None,
        paper_id: str | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.max_retries = max_retries
        #: P3 observability — None keeps runs hermetic (D3).
        self.trace: TraceWriter | None = trace
        self.paper_id: str | None = paper_id
        #: Raw completions observed during the current `_create` (retries).
        self._attempts = 0
        # Real Instructor clients expose event hooks; the fake client does
        # not, so `getattr` keeps the StructuredClient protocol untouched and
        # trace hooks no-op until `set_trace` supplies a writer.
        on = getattr(client, "on", None)
        if on is not None:
            on("completion:response", self._on_raw_response)
            on("completion:usage", self._on_usage)
            on("completion:error", self._on_raw_error)
            on("completion:last_attempt", self._on_raw_last_attempt)

    def set_trace(self, trace: TraceWriter | None, paper_id: str | None) -> None:
        """Bind a per-run trace writer + paper id (called by the engine).

        Passing ``(None, None)`` detaches tracing (statelessness: a fresh
        writer per paper, nothing leaks between runs).
        """
        self.trace = trace
        self.paper_id = paper_id

    # -- trace hooks (instructor event API; no-op when tracing is off) -------

    def _trace_event(self, name: str, **payload: Any) -> None:
        if self.trace is None:
            return
        self.trace.write(
            {
                "event": name,
                "ts": _now(),
                "paper_id": self.paper_id,
                **payload,
            }
        )

    def _on_raw_response(self, response: Any) -> None:
        self._attempts += 1
        self._trace_event(
            "raw_response",
            attempt=self._attempts,
            response=json_safe(response),
        )

    def _on_usage(self, usage: Any, *, attempt_number: int = 1) -> None:
        self._trace_event("usage", attempt=attempt_number, usage=json_safe(usage))

    def _on_raw_error(
        self,
        error: Exception,
        *,
        attempt_number: int = 1,
        max_attempts: int | None = None,
        is_last_attempt: bool = False,
    ) -> None:
        self._trace_event(
            "raw_error",
            attempt=attempt_number,
            max_attempts=max_attempts,
            is_last_attempt=is_last_attempt,
            error_type=type(error).__name__,
            detail=str(error),
        )

    def _on_raw_last_attempt(
        self,
        error: Exception,
        *,
        attempt_number: int = 1,
        max_attempts: int | None = None,
        is_last_attempt: bool = False,
    ) -> None:
        self._trace_event(
            "last_attempt",
            attempt=attempt_number,
            max_attempts=max_attempts,
            error_type=type(error).__name__,
            detail=str(error),
        )

    # -- primitives ---------------------------------------------------------

    def _create(
        self,
        response_model: type[Any],
        prompt: str,
        *,
        stage: str | None = None,
    ) -> Any:
        if self.trace is not None:
            self.trace.write(
                {
                    "event": "request",
                    "ts": _now(),
                    "paper_id": self.paper_id,
                    "stage": stage,
                    "model": self.model,
                    "prompt_chars": len(prompt),
                    "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[
                        :16
                    ],
                }
            )
        self._attempts = 0
        kwargs: dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        try:
            result = self._client.create(
                response_model=response_model,
                messages=[{"role": "user", "content": prompt}],
                max_retries=self.max_retries,
                **kwargs,
            )
        except Exception as exc:
            if self.trace is not None:
                self.trace.write(
                    {
                        "event": "error",
                        "ts": _now(),
                        "paper_id": self.paper_id,
                        "stage": stage,
                        "model": self.model,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                        "trace": traceback.format_exc(),
                    }
                )
            raise
        if self.trace is not None:
            self.trace.write(
                {
                    "event": "response",
                    "ts": _now(),
                    "paper_id": self.paper_id,
                    "stage": stage,
                    "model": self.model,
                    "final_model": json_safe(result),
                    "attempts": self._attempts,
                }
            )
        return result

    @staticmethod
    def _join_chunks(chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        budget = MAX_CHUNK_CHARS
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            heading = chunk.get("heading") or ""
            block = f"[{heading}]\n{text}" if heading else text
            if len(block) <= budget:
                parts.append(block)
                budget -= len(block)
            else:
                parts.append(block[:budget])
                break  # best-effort context cap
        return "\n\n".join(parts)

    # -- stages -------------------------------------------------------------

    def extract_paper(self, chunks: list[dict[str, Any]]) -> PaperExtractionSchema:
        """Stage A1–A2: explicit datasets + core mechanism."""
        prompt = (
            "You extract quantitative research specifications. From the paper "
            "chunks below, produce:\n"
            "- paper_title: the paper's title or a short identifier\n"
            "- primary_hypothesis: the core market claim in one sentence\n"
            "- datasets_used: every EXPLICIT data requirement — asset_class "
            "strictly one of {Equities, Crypto, Futures, Forex}. INFER IT FROM "
            "THE SYMBOL: symbols ending in USDT/USDC, or BTC/ETH/SOL/ADA/DOGE "
            "perpetuals, are 'Crypto'; QQQ/SPY/NQ index futures are "
            "'Equities'/'Futures'. symbol in UPPERCASE (e.g. BTCUSDT, QQQ); "
            "required_granularity one of {tick, 1s, 1m, 5m, 15m, 1h, 1d}; "
            "requires_l2_book TRUE only when the paper demands order book / "
            "L2/L3 depth; requires_order_flow TRUE only when the paper "
            "explicitly demands order flow / flow imbalance data (funding "
            "rates alone do NOT count); start_year an integer when a start "
            "date is given.\n"
            "- core_mechanism: a technical summary of the entry rule, exit "
            "rule, and sizing rule (at least 50 characters, naming any "
            "specific indicators/triggers the paper uses).\n\n"
            f"Paper chunks:\n{self._join_chunks(chunks)}"
        )
        return self._create(PaperExtractionSchema, prompt, stage="extract_requirements")

    def abstract_mechanism(
        self,
        extraction: PaperExtractionSchema,
        chunks: list[dict[str, Any]],
    ) -> CausalAbstractionSchema:
        """Stage A4: domain-agnostic causal physics + non-negotiable bounds."""
        prompt = (
            "You abstract a trading mechanism into domain-agnostic causal "
            "physics. Given the paper's extraction and chunks, produce:\n"
            "- abstract_hypothesis: the mechanism stated without asset-class "
            "specifics (e.g. 'Price displacement relative to a volume "
            "benchmark creates short-term directional persistence')\n"
            "- causal_anchor: the underlying market-structure reason (e.g. "
            "'Institutional execution algorithms benchmark against intraday "
            "VWAP')\n"
            "- non_negotiable_invariants: structural bounds that MUST be "
            "preserved (e.g. session-open anchors, benchmark identity)\n\n"
            f"Core claim: {extraction.primary_hypothesis}\n"
            f"Core mechanism: {extraction.core_mechanism}\n\n"
            f"Paper chunks:\n{self._join_chunks(chunks)}"
        )
        return self._create(CausalAbstractionSchema, prompt, stage="abstract_mechanism")

    def apply_operators(
        self,
        extraction: PaperExtractionSchema,
        abstraction: CausalAbstractionSchema,
        chunks: list[dict[str, Any]],
    ) -> list[StrategyAnnotation]:
        """Stage A5: run the 8 quant reasoning operators (one structured call).

        Returns exactly 8 annotations (schema-enforced via min_length=8).
        Annotation only — no deletion authority (No-Drop rule).
        """
        prompt = (
            "You are a quant reasoning engine. Process the mechanism through "
            "EXACTLY the 8 operators below, one `annotations` entry per "
            "operator (no more, no less). For each: observation (what the "
            "operator reveals), proposed_modification (concrete, implementable "
            "change), risk_tags (0..n from "
            "HIGH_SESSION_SENSITIVITY, LOW_LIQUIDITY_FRAGILITY, "
            "PARAMETRIC_OVERFIT_RISK, EXECUTION_SLIPPAGE_HEAVY).\n\n"
            "IMPORTANT: annotations are advisory only — never a recommendation "
            "to discard the strategy.\n\n"
            f"{operator_prompt_block()}\n\n"
            f"Core claim: {extraction.primary_hypothesis}\n"
            f"Core mechanism: {extraction.core_mechanism}\n"
            f"Abstract hypothesis: {abstraction.abstract_hypothesis}\n"
            f"Causal anchor: {abstraction.causal_anchor}\n"
            f"Invariants: {', '.join(abstraction.non_negotiable_invariants)}\n\n"
            f"Paper chunks:\n{self._join_chunks(chunks)}"
        )
        wrapped: OperatorAnnotations = self._create(
            OperatorAnnotations, prompt, stage="apply_operators"
        )
        return wrapped.annotations


# ---------------------------------------------------------------------------
# Fake client (CI / fast tests / `--llm fake`)
# ---------------------------------------------------------------------------


class FakeInstructorClient(StructuredClient):
    """Deterministic Instructor-shaped client with canned responses.

    Intended for tests and local dry-runs. `responses` maps a response_model
    type to the instance Instructor would have returned. Every `create` call
    is recorded on `.calls` so tests can assert statelessness.
    """

    def __init__(self, responses: dict[type[Any], Any] | None = None) -> None:
        self.responses: dict[type[Any], Any] = responses or {}
        self.calls: list[type[Any]] = []
        #: Prompt content of every `create` — lets tests assert statelessness.
        self.recorded_prompts: list[str] = []

    def create(
        self,
        response_model: type[Any],
        messages: list[dict[str, str]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(response_model)
        self.recorded_prompts.append(" ".join(m.get("content", "") for m in messages))
        if response_model not in self.responses:
            raise KeyError(
                f"FakeInstructorClient has no canned response for {response_model.__name__}"
            )
        return self.responses[response_model]


def fake_evaluator(
    *,
    extraction: PaperExtractionSchema | None = None,
    abstraction: CausalAbstractionSchema | None = None,
    annotations: list[StrategyAnnotation] | None = None,
) -> tuple[LLMEvaluator, FakeInstructorClient]:
    """Build an evaluator + its underlying fake client for tests.

    Returns `(evaluator, client)` so tests can both drive the DAG and assert
    on call history.
    """
    responses: dict[type[Any], Any] = {}
    if extraction is not None:
        responses[PaperExtractionSchema] = extraction
    if abstraction is not None:
        responses[CausalAbstractionSchema] = abstraction
    if annotations is not None:
        responses[OperatorAnnotations] = OperatorAnnotations(annotations=annotations)
    client = FakeInstructorClient(responses)
    return LLMEvaluator(client), client


__all__ = [
    "MAX_CHUNK_CHARS",
    "FakeInstructorClient",
    "LLMEvaluator",
    "StructuredClient",
    "fake_evaluator",
    "ollama_client",
    "openai_compat_client",
]
