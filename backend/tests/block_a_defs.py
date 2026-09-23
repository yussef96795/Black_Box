"""Shared canned Block A models for tests (deterministic, no LLM/Ollama)."""

from __future__ import annotations

from black_box.block_a.models import (
    CausalAbstractionSchema,
    DataGranularity,
    DatasetRequirement,
    PaperExtractionSchema,
    RiskTag,
    StrategyAnnotation,
)

VWAP_MECHANISM = (
    "Long when price crosses above the VWAP anchor and cumulative volume "
    "delta is positive; exit when price crosses below VWAP."
)


def vwap_extraction(
    symbol: str = "BTCUSDT", *, l2: bool = True
) -> PaperExtractionSchema:
    return PaperExtractionSchema(
        paper_title=f"VWAP Displacement ({symbol})",
        primary_hypothesis="Price displacement vs VWAP predicts short-term persistence",
        datasets_used=[
            DatasetRequirement(
                asset_class="Crypto",
                symbol=symbol,
                required_granularity=DataGranularity.MINUTE_1,
                requires_l2_book=l2,
            )
        ],
        core_mechanism=VWAP_MECHANISM,
    )


def vwap_abstraction() -> CausalAbstractionSchema:
    return CausalAbstractionSchema(
        abstract_hypothesis="Anchor-relative displacement predicts short-term persistence",
        causal_anchor="Institutional execution algorithms benchmark against intraday VWAP",
        non_negotiable_invariants=["anchor must reset at primary session open"],
    )


def vwap_annotations(*, slippage: bool = True) -> list[StrategyAnnotation]:
    """8 operator annotations (exactly the operator sweep) with soft risks."""
    base = [
        ("Restrict", "Fails in flat low-volume regimes", []),
        ("Invert", "Short when displacement is negative", []),
        ("Relax", "Use distance-to-anchor instead of cross", []),
        ("Substitute", "Volume delta proxies tick imbalance", []),
        ("Decompose", "Entry anchored at open, exit at VWAP", []),
        ("Combine", "Add volatility regime filter", [RiskTag.PARAMETRIC_OVERFIT_RISK]),
        ("Adversarial", "Spoofing can fake delta", [RiskTag.LOW_LIQUIDITY_FRAGILITY]),
        ("Generalize", "Same microstructure across liquid perps", []),
    ]
    if slippage:
        base[6] = (
            "Adversarial",
            "Spoofing can fake delta",
            [RiskTag.LOW_LIQUIDITY_FRAGILITY, RiskTag.EXECUTION_SLIPPAGE_HEAVY],
        )
    return [
        StrategyAnnotation(
            operator_name=name,
            observation=obs,
            proposed_modification=f"mod: {obs}",
            risk_tags=tags,
        )
        for name, obs, tags in base
    ]


def qqq_extraction() -> PaperExtractionSchema:
    """Equity paper — QQQ is NOT in the seed crypto catalog → hard stop."""
    return PaperExtractionSchema(
        paper_title="US Open Auction Breakout",
        primary_hypothesis="Open auction range breakouts predict continuation",
        datasets_used=[
            DatasetRequirement(
                asset_class="Equities",
                symbol="QQQ",
                required_granularity=DataGranularity.MINUTE_1,
                requires_l2_book=True,
                start_year=2015,
            )
        ],
        core_mechanism=(
            "Long when price breaks above the opening range high during the "
            "first 30 minutes; exit at VWAP cross or session close."
        ),
    )


__all__ = [
    "VWAP_MECHANISM",
    "qqq_extraction",
    "vwap_abstraction",
    "vwap_annotations",
    "vwap_extraction",
]
