"""Block A (Alpha Feasibility & Strategy Ingestion Engine) Pydantic schemas.

Schema-first enforcement (BLOCK_A_SPECIFICATION.md §1 invariant 2): every
stage boundary transfers strictly-validated Pydantic models — unstructured
strings are never passed between stages. All models use `extra="forbid"`
(Rules.md §3) so schema drift fails loudly.

Stage map:
    A1–A2  PaperExtractionSchema, DatasetRequirement, DataGranularity
    A3     ResourceVerdict / ResourceCheckResult (DuckDB catalog check)
    A4     CausalAbstractionSchema
    A5     StrategyAnnotation, OperatorAnnotations, RiskTag
    A6     ExecutableStrategySpec, StrategyTier
    DAG    BlockAState (msgpack-safe primitives for LangGraph checkpoints)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Hard-stop error code emitted when Stage A3 finds missing data + no proxy.
RESOURCE_INSUFFICIENT_ERROR = "RESOURCE_INSUFFICIENT_ERROR"


class DataGranularity(str, Enum):
    """Supported data granularities (spec §4 Module 2 schema)."""

    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAILY = "1d"


class DatasetRequirement(BaseModel):
    """One explicit data requirement extracted from the paper (Stage A1/A2)."""

    model_config = ConfigDict(extra="forbid")

    asset_class: str = Field(..., description="Equities, Crypto, Futures, Forex")
    symbol: str = Field(..., description="Target symbol e.g., QQQ, NQ, BTCUSDT")
    required_granularity: DataGranularity
    requires_l2_book: bool = Field(default=False)
    requires_order_flow: bool = Field(default=False)
    start_year: int | None = Field(default=None)


class PaperExtractionSchema(BaseModel):
    """Stage A1–A2 output: explicit dataset requirements + core mechanism."""

    model_config = ConfigDict(extra="forbid")

    paper_title: str
    primary_hypothesis: str
    datasets_used: list[DatasetRequirement] = Field(default_factory=list)
    core_mechanism: str = Field(
        ...,
        min_length=50,
        max_length=2000,
        description=(
            "Technical summary of entry, exit, and sizing rules (must include "
            "specific indicators/triggers mentioned)"
        ),
    )


class CausalAbstractionSchema(BaseModel):
    """Stage A4 output: domain-agnostic causal physics of the mechanism."""

    model_config = ConfigDict(extra="forbid")

    abstract_hypothesis: str = Field(
        ...,
        description=(
            "Domain-agnostic theory, e.g. 'Price displacement relative to "
            "volume benchmark creates short-term directional persistence'"
        ),
    )
    causal_anchor: str = Field(
        ...,
        description=(
            "Underlying market structure reason, e.g. 'Institutional execution "
            "algorithms benchmarking against intraday VWAP'"
        ),
    )
    non_negotiable_invariants: list[str] = Field(
        default_factory=list,
        description="Structural bounds that MUST be preserved",
    )


class RiskTag(str, Enum):
    """Annotation-only risk flags (Stage A5 output schema)."""

    HIGH_SESSION_SENSITIVITY = "HIGH_SESSION_SENSITIVITY"
    LOW_LIQUIDITY_FRAGILITY = "LOW_LIQUIDITY_FRAGILITY"
    PARAMETRIC_OVERFIT_RISK = "PARAMETRIC_OVERFIT_RISK"
    EXECUTION_SLIPPAGE_HEAVY = "EXECUTION_SLIPPAGE_HEAVY"


#: Canonical risk-tag priority order — single source of truth shared by the
#: gatekeeper toggle menu, Tier 1 guard selection, and annotation unions.
#: Explicit (not derived from enum/dict declaration order) so menu indices
#: and guard precedence never drift silently when the enum is reordered.
RISK_PRIORITY: tuple[RiskTag, ...] = (
    RiskTag.EXECUTION_SLIPPAGE_HEAVY,
    RiskTag.LOW_LIQUIDITY_FRAGILITY,
    RiskTag.HIGH_SESSION_SENSITIVITY,
    RiskTag.PARAMETRIC_OVERFIT_RISK,
)


class StrategyAnnotation(BaseModel):
    """One Stage A5 quant-operator observation.

    No `is_testable` / deletion authority exists here — the No-Drop rule
    (spec §1 invariant 3) is structural: this schema simply cannot carry a
    delete signal. Only the Stage A3 Data Check may halt / set
    `is_testable = False` on the final spec.
    """

    model_config = ConfigDict(extra="forbid")

    operator_name: Literal[
        "Restrict",
        "Invert",
        "Relax",
        "Substitute",
        "Decompose",
        "Combine",
        "Adversarial",
        "Generalize",
    ]
    observation: str
    proposed_modification: str
    risk_tags: list[RiskTag] = Field(default_factory=list)


class OperatorAnnotations(BaseModel):
    """Wrapper around a full 8-operator pass (Stage A5).

    `min_length` enforces the complete operator sweep in one structured call
    — a partial sweep is a schema violation and triggers Instructor's retry.
    """

    model_config = ConfigDict(extra="forbid")

    annotations: list[StrategyAnnotation] = Field(min_length=8, max_length=8)


class StrategyTier(str, Enum):
    """Tier assignment rules (spec §4 Module 4)."""

    TIER_0_LITERAL = "TIER_0_LITERAL"
    TIER_1_PARAMETRIC = "TIER_1_PARAMETRIC"
    TIER_2_GENERALIZED = "TIER_2_GENERALIZED"
    TIER_3_AUGMENTED = "TIER_3_AUGMENTED"


# ---------------------------------------------------------------------------
# GenericPrimitiveNode — recursive DSL Abstract Syntax Tree (Stage A6)
# ---------------------------------------------------------------------------
#: Max nesting depth for `OperatorNode` trees (spec §4 Module 4). Deep trees
#: exceed what a vectorized backtest engine can express as flat primitives —
#: fail loudly instead of silently producing an un-expressible spec.
MAX_AST_DEPTH = 8

#: `op` values an OperatorNode may carry (deterministic, not LLM-extensible).
AST_OPERATORS = (
    "zscore",
    "ema",
    "sma",
    "lag",
    "returns",
    "abs",
    "log",
    "ratio",
    "add",
    "sub",
    "mul",
    "div",
)


class DataStreamNode(BaseModel):
    """Leaf: a raw symbol/granularity data stream (the AST input source)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["data_stream"] = "data_stream"
    symbol: str
    granularity: str


class OperandNode(BaseModel):
    """Leaf: a registry primitive, an OHLCV data series, or a numeric literal.

    `id` resolution is enforced at compile time against the primitive
    registry (see `spec_compiler.validate_ast_registry`) plus the allowed
    data-series/literal escape hatches — unknown ids fail loudly.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["literal", "indicator", "transform"]
    id: str


class OperatorNode(BaseModel):
    """Interior node: composes sub-trees (recursion point of the AST).

    Strict by construction: `extra="forbid"`, `kind`/`op` are Literals, and a
    `model_validator` enforces arity (binary requires `right`, unary forbids
    it) and the `MAX_AST_DEPTH` ceiling so runaway nesting cannot pass.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["binary_op", "unary_op"]
    op: Literal[*AST_OPERATORS]
    left: GenericPrimitiveNode
    right: GenericPrimitiveNode | None = None

    @model_validator(mode="after")
    def _enforce_arity_and_depth(self) -> OperatorNode:
        if self.kind == "binary_op" and self.right is None:
            raise ValueError("binary_op requires `right` operand")
        if self.kind == "unary_op" and self.right is not None:
            raise ValueError("unary_op forbids `right` operand")
        if _node_depth(self) > MAX_AST_DEPTH:
            raise ValueError(f"AST depth exceeds MAX_AST_DEPTH={MAX_AST_DEPTH}")
        return self


GenericPrimitiveNode = Annotated[
    DataStreamNode | OperandNode | OperatorNode,
    Field(discriminator="kind"),
]


def _node_depth(node: GenericPrimitiveNode) -> int:
    if isinstance(node, (DataStreamNode, OperandNode)):
        return 1
    right = _node_depth(node.right) if node.right else 0
    return 1 + max(_node_depth(node.left), right)


OperatorNode.model_rebuild()


class ExecutableStrategySpec(BaseModel):
    """Final Block B/C/D export contract (Stage A6 + Module 5 output)."""

    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(..., description="e.g. STRAT_VWAP_BTCUSDT_T0")
    tier: StrategyTier
    target_asset: str
    timeframe: str

    entry_trigger_primitive: str
    exit_trigger_primitive: str
    filter_primitives: list[str] = Field(default_factory=list)

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Static/Dynamic parameter bounds for Block B/C",
    )

    # Optional recursive expression tree backing the headline signal (e.g.
    # ZScore(EMA(close, 20))). Flat primitive fields above remain the
    # vectorized-execution boundary contract; `signal_ast` is the expressive
    # layer Block C/D may consume. None unless the compiler emits one.
    signal_ast: GenericPrimitiveNode | None = None

    # Annotations passed to Block C validation suite
    risk_annotations: list[RiskTag] = Field(default_factory=list)
    causal_anchor_notes: str = ""
    is_testable: bool = Field(
        default=True,
        description=(
            "Only the Stage A3 hard data check may set False; Stage A5 "
            "quant operators never touch this field."
        ),
    )


class ResourceVerdict(BaseModel):
    """One Stage A3 catalog verdict for a DatasetRequirement."""

    model_config = ConfigDict(extra="forbid")

    asset_class: str
    symbol: str
    granularity: DataGranularity
    status: Literal["available", "proxy", "missing"]
    proxy_source: str | None = Field(
        default=None,
        description="Approved substitute when status == 'proxy'",
    )
    note: str = Field(default="")


class ResourceCheckResult(BaseModel):
    """Aggregated Stage A3 output; drives the hard-stop router edge."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[ResourceVerdict] = Field(default_factory=list)
    all_available: bool = Field(
        default=False,
        description="True iff no verdict is 'missing' (data verified)",
    )


class BlockAState(BaseModel):
    """LangGraph state for the Block A DAG.

    Uses msgpack-serializable primitives only (list/dict/str) so the graph
    can be checkpointed by MemorySaver without custom serializers — mirrors
    the house convention in `black_box/state/schema.py`.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_path: str | None = None
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    extraction: dict[str, Any] | None = None
    resource_check: dict[str, Any] | None = None
    abstraction: dict[str, Any] | None = None
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    specs: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal[
        "running",
        "resource_insufficient",
        "gatekeeper",
        "complete",
        "rejected",
        "llm_extraction_failed",
    ] = Field(default="running")
    error_code: str | None = None
    #: Structured failure from a Stage A1/A4/A5 LLM call that exhausted its
    #: validation retries — {stage, error_type, detail, trace_hash}. The full
    #: traceback lives in the trace file (P3) / logs, never in the payload.
    stage_error: dict[str, Any] | None = None


class BlockAResult(BaseModel):
    """Outcome of one paper run (CLI / API return contract)."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    paper_title: str = ""
    status: Literal[
        "complete",
        "resource_insufficient",
        "rejected",
        "llm_extraction_failed",
    ]
    specs: list[ExecutableStrategySpec] = Field(default_factory=list)
    risk_tags: list[RiskTag] = Field(default_factory=list)
    out_path: str | None = None
    error_code: str | None = None
    #: {stage, error_type, detail, trace_hash} when an LLM stage exhausted its
    #: retries (D2). Full stack trace is written to the trace file, not here.
    error: dict[str, Any] | None = None


class GatekeeperDecision(BaseModel):
    """Module 5 outcome — status plus the human-curated spec array (D4).

    `specs` is the filtered/edited list the DAG writes back into
    `BlockAState.specs`, so only curated specs reach `validate_and_export`.
    Full rejection carries an empty array; `approve=True` an idealized
    decision carrying every compiled spec.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "rejected"]
    specs: list[ExecutableStrategySpec] = Field(default_factory=list)


__all__ = [
    "AST_OPERATORS",
    "MAX_AST_DEPTH",
    "RESOURCE_INSUFFICIENT_ERROR",
    "BlockAResult",
    "BlockAState",
    "CausalAbstractionSchema",
    "DataGranularity",
    "DataStreamNode",
    "DatasetRequirement",
    "ExecutableStrategySpec",
    "GatekeeperDecision",
    "GenericPrimitiveNode",
    "OperandNode",
    "OperatorAnnotations",
    "OperatorNode",
    "PaperExtractionSchema",
    "ResourceCheckResult",
    "ResourceVerdict",
    "RiskTag",
    "StrategyAnnotation",
    "StrategyTier",
]
