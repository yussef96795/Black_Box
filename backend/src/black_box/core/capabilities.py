"""System Capabilities contract (Block A Step 2).

Defines what the Black_Box platform can do — data granularity, compute
primitives, and domain coverage. The Feasibility Auditor checks an extracted
strategy specification against this contract and emits
PASSED / REQUIRES_HITL / REJECTED. Exposed read-only via
`GET /api/v1/feasibility/capabilities`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CAPABILITIES_VERSION = "1.0.0"


class Capability(BaseModel):
    """One platform capability the auditor can test a strategy against."""

    name: str
    type: str = Field(description="data | compute | domain")
    granularity: str = Field(
        description="Coarsest supported granularity, e.g. L3 / 1m / daily"
    )
    primitives: list[str] = Field(default_factory=list)


class Capabilities(BaseModel):
    """The full platform capability table (audit input contract)."""

    version: str
    capabilities: list[Capability] = Field(default_factory=list)


DEFAULT_CAPABILITIES = Capabilities(
    version=CAPABILITIES_VERSION,
    capabilities=[
        # --- data -----------------------------------------------------------
        Capability(
            name="crypto_market_data",
            type="data",
            granularity="L3",
            primitives=[
                "crypto_ohlcv",  # open/high/low/close/volume
                "orderbook",  # L2 book snapshots + L3 tick-by-tick
                "trades",  # executed trades (tick)
                "funding_rates",  # perpetual funding
                "open_interest",
            ],
        ),
        Capability(
            name="equity_market_data",
            type="data",
            granularity="unsupported",
            primitives=["equity_ohlcv", "dividends", "corporate_actions"],
        ),
        Capability(
            name="macro_rates_data",
            type="data",
            granularity="unsupported",
            primitives=["treasury_yields", "central_bank_rates", "term_structure"],
        ),
        # --- compute ---------------------------------------------------------
        Capability(
            name="vectorized_backtest",
            type="compute",
            granularity="in-process",
            primitives=[
                "pandas/numpy/polars vectorization",
                "combinatorial purged cross-validation (CPCV)",
                "monte carlo parameter sweep",
                "risk guardrail injection (drawdown/leverage/loss caps)",
            ],
        ),
        # Explicitly NOT supported: in-loop black-box inference.
        Capability(
            name="ml_inference",
            type="compute",
            granularity="unsupported",
            primitives=["neural_network", "gradient_boosting", "llm_in_loop"],
        ),
        # --- domain -----------------------------------------------------------
        Capability(
            name="crypto_domain",
            type="domain",
            granularity="24/7",
            primitives=[
                "perpetual_futures",  # no expiry
                "funding_rate_analog",  # replaces TradFi interest rates
                "instant_settlement",  # replaces T+2
                "staking_yield",  # replaces dividends/Treasury yield
                "rfq_otc_desk",  # replaces dark pools
            ],
        ),
    ],
)
