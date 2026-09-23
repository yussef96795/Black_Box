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

import logging
from typing import Any, Protocol

from black_box.block_a.models import (
    CausalAbstractionSchema,
    OperatorAnnotations,
    PaperExtractionSchema,
    StrategyAnnotation,
)
from black_box.block_a.operators import operator_prompt_block
from black_box.core.config import get_settings

logger = logging.getLogger(__name__)

#: Chunk text cap per prompt so local models stay within context cheaply.
MAX_CHUNK_CHARS = 12_000


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
    ) -> None:
        self._client = client
        self.model = model
        self.max_retries = max_retries

    # -- primitives ---------------------------------------------------------

    def _create(self, response_model: type[Any], prompt: str) -> Any:
        kwargs: dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        return self._client.create(
            response_model=response_model,
            messages=[{"role": "user", "content": prompt}],
            max_retries=self.max_retries,
            **kwargs,
        )

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
            "required_granularity one of {tick, 1s, 1m, 1h, 1d}; "
            "requires_l2_book TRUE only when the paper demands order book / "
            "L2/L3 depth; requires_order_flow TRUE only when the paper "
            "explicitly demands order flow / flow imbalance data (funding "
            "rates alone do NOT count); start_year an integer when a start "
            "date is given.\n"
            "- core_mechanism: a technical summary of the entry rule, exit "
            "rule, and sizing rule.\n\n"
            f"Paper chunks:\n{self._join_chunks(chunks)}"
        )
        return self._create(PaperExtractionSchema, prompt)

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
        return self._create(CausalAbstractionSchema, prompt)

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
        wrapped: OperatorAnnotations = self._create(OperatorAnnotations, prompt)
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
