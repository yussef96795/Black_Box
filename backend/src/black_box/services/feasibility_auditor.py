"""Feasibility Auditor Gate (Block A Step 2).

Runs four rule-based checkers over an extracted strategy specification and
returns an aggregated verdict:

  1. hard_dependencies — data providers/granularity the strategy references,
     checked against the platform capability contract (core/capabilities.py).
  2. algorithmic_compute — black-box ML / compute-budget demands vs supported
     primitives (vectorized backtest only; in-loop inference unsupported).
  3. domain_mapping    — TradFi concepts (rates, trading hours, settlement…)
     mapped to crypto analogs with confidence.
  4. tables            — table-validation reports from 1d gate the audit:
     an unreadable/parametrically-invalid table forces REQUIRES_HITL.

Verdict rules (deterministic, testable — Rules.md §1 KISS):
  * any REJECTED check            → REJECTED
  * else any REQUIRES_HITL check  → REQUIRES_HITL
  * else                          → PASSED

Checkers are keyword-taxonomy based by design: a later version may swap in
LLM-assisted extraction, but the decision rules and contracts stay identical.
"""

from __future__ import annotations

from typing import Any, Literal

from black_box.schemas import (
    DataDependency,
    DomainMapping,
    FeasibilityCheck,
    FeasibilityResult,
    ModelCheck,
    TableValidationReport,
)

# ---------------------------------------------------------------------------
# Keyword taxonomies (single source of truth for the rule-based checkers)
# ---------------------------------------------------------------------------

#: name -> ("tradfi" | "crypto")  — known data providers.
DATA_PROVIDERS: dict[str, str] = {
    "bloomberg": "tradfi",
    "reuters": "tradfi",
    "refinitiv": "tradfi",
    "fred": "tradfi",
    "yahoo finance": "tradfi",
    "eod historical": "tradfi",
    "stooq": "tradfi",
    "binance": "crypto",
    "coinbase": "crypto",
    "okx": "crypto",
    "kraken": "crypto",
    "bybit": "crypto",
    "ccxt": "crypto",
}

#: granularity demands -> supported by the platform capability table?
GRANULARITY_HINTS: dict[str, str] = {
    "tick data": "tick (L3)",
    "order book data": "orderbook",
    "orderbook data": "orderbook",
    "l3 data": "L3",
    "minute data": "1m",
    "daily data": "daily",
    "intraday": "1m",
    "real-time data": "1m",
}
SUPPORTED_GRANULARITIES = {"tick (L3)", "orderbook", "L3", "1m", "daily"}

#: black-box models we will NOT transpile into a vectorized backtest.
BLACK_BOX_MODELS: list[str] = [
    "neural network",
    "deep learning",
    "dnn",
    "lstm",
    "transformer model",
    "gradient boosting",
    "xgboost",
    "lightgbm",
    "random forest",
    "llm",
    "large language model",
]

#: compute hints that indicate tighter-than-supported budgets.
COMPUTE_HINTS: list[str] = [
    "gpu",
    "gpus",
    "real-time inference",
    "low-latency inference",
    "high-frequency",
    "microsecond",
]

#: TradFi concept -> (crypto analog, confidence, note).
DOMAIN_MAP: dict[str, tuple[str, float, str]] = {
    "interest rate": ("funding rate", 0.9, "perpetual funding replaces TradFi rates"),
    "treasury yield": ("staking/protocol yield", 0.85, "yield analog, not risk-free"),
    "trading hours": (
        "24/7 market",
        0.95,
        "no session close; schedule logic must adapt",
    ),
    "settlement": ("instant settlement", 0.85, "block confirmation replaces T+2"),
    "dividend": ("staking reward / airdrop", 0.7, "non-deterministic schedule"),
    "limit order": ("limit order", 1.0, "identical primitive"),
    "tick size": ("tick size", 1.0, "identical primitive"),
    "margin rate": ("borrow / funding cost", 0.8, "borrow pool rates"),
    "dark pool": ("RFQ / OTC desk", 0.7, "less transparent liquidity"),
    "fomc": (
        "macro events / on-chain flows",
        0.6,
        "weak analog — macro calendar differs",
    ),
    "market maker": ("liquidity provider", 0.8, "incentivized by maker rebates"),
    "earnings": ("protocol releases / listings", 0.6, "weak analog"),
}


# ---------------------------------------------------------------------------
# Checkers
# ---------------------------------------------------------------------------


def _mentions(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [t for t in terms if t in lowered]


def audit_dependencies(text: str) -> list[DataDependency]:
    """Detect referenced data sources and granularity vs the contract."""
    found: dict[str, str] = {}
    for provider, kind in DATA_PROVIDERS.items():
        if provider.lower() in text.lower():
            found[provider] = kind

    dependencies: list[DataDependency] = []
    lowered = text.lower()
    mentioned_domain = [c for c in DOMAIN_MAP if c in lowered]
    for provider, kind in found.items():
        if kind == "crypto":
            dependencies.append(
                DataDependency(
                    name=provider,
                    status="satisfied",
                    latency_ms=0,
                    evidence=f"crypto-native provider '{provider}' within platform data layer",
                )
            )
        else:
            # TradFi provider: map via a domain concept actually mentioned.
            if mentioned_domain:
                concept = mentioned_domain[0]
                analog, confidence, note = DOMAIN_MAP[concept]
                dependencies.append(
                    DataDependency(
                        name=provider,
                        status="partial",
                        latency_ms=None,
                        evidence=(
                            f"TradFi provider '{provider}' — analog '{analog}' via '{concept}' "
                            f"(confidence {confidence:.2f}); {note}"
                        ),
                    )
                )
            else:
                dependencies.append(
                    DataDependency(
                        name=provider,
                        status="missing",
                        latency_ms=None,
                        evidence=f"provider '{provider}' not available on platform",
                    )
                )

    for hint, granularity in GRANULARITY_HINTS.items():
        if hint in text.lower():
            dependencies.append(
                DataDependency(
                    name=f"granularity:{granularity}",
                    status="satisfied"
                    if granularity in SUPPORTED_GRANULARITIES
                    else "missing",
                    latency_ms=None,
                    evidence=f"document demands '{hint}' ({granularity})",
                )
            )
    return dependencies


def audit_compute(text: str) -> tuple[list[ModelCheck], dict[str, Any]]:
    """Flag black-box models and compute-budget demands."""
    models = [
        ModelCheck(
            name=model,
            supported=False,
            reason="black-box model unsupported in vectorized backtest",
        )
        for model in _mentions(text, BLACK_BOX_MODELS)
    ]
    compute_budget: dict[str, Any] = {
        "gpu_memory_mb": None,
        "inference_ms": None,
        "hints": _mentions(text, COMPUTE_HINTS),
    }
    return models, compute_budget


def audit_domain(text: str) -> list[DomainMapping]:
    """Map TradFi concepts found in the document to crypto analogs."""
    lowered = text.lower()
    mappings: list[DomainMapping] = []
    for concept, (analog, confidence, note) in DOMAIN_MAP.items():
        if concept in lowered:
            mappings.append(
                DomainMapping(
                    tradfi=concept, crypto=analog, confidence=confidence, note=note
                )
            )
    return mappings


def _check_tables(
    tables: list[TableValidationReport],
) -> tuple[Literal["PASSED", "REQUIRES_HITL"], list[str]]:
    """Unreadable/orphan tables must be confirmed by a human."""
    problems = [t.table_id for t in tables if t.status != "fit"]
    if problems:
        return "REQUIRES_HITL", [
            f"table validation: {t.table_id} is {t.status}: {'; '.join(t.errors) or 'no details'}"
            for t in tables
            if t.status != "fit"
        ]
    return "PASSED", []


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def audit(
    text: str, tables: list[TableValidationReport] | None = None
) -> FeasibilityResult:
    """Run all checkers in sequence and aggregate the verdict."""
    text = (text or "").strip()
    tables = tables or []

    checks: list[FeasibilityCheck] = []

    # 1. hard dependencies
    dependencies = audit_dependencies(text)
    dep_statuses = {d.status for d in dependencies}
    if "missing" in dep_statuses:
        dep_verdict: Literal["PASSED", "REQUIRES_HITL", "REJECTED"] = "REJECTED"
    elif "partial" in dep_statuses:
        dep_verdict = "REQUIRES_HITL"
    else:
        dep_verdict = "PASSED"
    checks.append(
        FeasibilityCheck(
            name="hard_dependencies",
            status=dep_verdict,
            details={
                "dependencies": [d.model_dump() for d in dependencies],
                "note": "latency_ms: null until data layer brings up real feeds (MVP assumption)",
            },
        )
    )

    # 2. algorithmic & compute
    models, compute_budget = audit_compute(text)
    if models:
        comp_verdict: Literal["PASSED", "REQUIRES_HITL", "REJECTED"] = "REJECTED"
    elif compute_budget["hints"]:
        comp_verdict = "REQUIRES_HITL"
    else:
        comp_verdict = "PASSED"
    checks.append(
        FeasibilityCheck(
            name="algorithmic_compute",
            status=comp_verdict,
            details={
                "models": [m.model_dump() for m in models],
                "compute_budget": compute_budget,
            },
        )
    )

    # 3. domain mapping
    mappings = audit_domain(text)
    weak = [m for m in mappings if m.confidence < 0.7]
    if weak:
        domain_verdict: Literal["PASSED", "REQUIRES_HITL", "REJECTED"] = "REQUIRES_HITL"
    else:
        domain_verdict = "PASSED"
    checks.append(
        FeasibilityCheck(
            name="domain_mapping",
            status=domain_verdict,
            details={
                "mapping": [m.model_dump() for m in mappings],
                "note": "low-confidence analogs (<0.7) require HITL",
            },
        )
    )

    # 4. tables (from 1d ingestion reports)
    table_verdict, table_notes = _check_tables(tables)
    checks.append(
        FeasibilityCheck(
            name="tables",
            status=table_verdict,
            details={
                "notes": table_notes,
                "reported": [t.model_dump() for t in tables],
            },
        )
    )

    # Aggregate: REJECTED > REQUIRES_HITL > PASSED
    if any(c.status == "REJECTED" for c in checks):
        status: Literal["PASSED", "REQUIRES_HITL", "REJECTED"] = "REJECTED"
    elif any(c.status == "REQUIRES_HITL" for c in checks):
        status = "REQUIRES_HITL"
    else:
        status = "PASSED"

    summary = _summarize(status, checks)
    return FeasibilityResult(status=status, checks=checks, summary=summary)


def _summarize(status: str, checks: list[FeasibilityCheck]) -> str:
    flagged = [c for c in checks if c.status != "PASSED"]
    if status == "PASSED":
        return "Strategy is fully feasible: all hard dependencies satisfied, no unsupported models or domain gaps."
    parts = [f"Audit result: {status}."]
    for c in flagged:
        detail_keys = list(c.details.keys()) if isinstance(c.details, dict) else []
        parts.append(f"{c.name}={c.status} ({', '.join(detail_keys) or 'see details'})")
    return " ".join(parts)


__all__ = [
    "DATA_PROVIDERS",
    "DOMAIN_MAP",
    "audit",
    "audit_compute",
    "audit_dependencies",
    "audit_domain",
]
