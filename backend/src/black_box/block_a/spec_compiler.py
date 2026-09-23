"""Stage A6 — Resource Engine Mapping & DSL Compiler (deterministic, no LLM).

Maps extracted mechanisms onto the primitive vocabulary
(`config/primitives_registry.json`), assigns Tier 0–3 labels, and prunes the
matrix to at most 5 high-plausibility specs (spec §1 invariant 4) for Block B.

Tier assignment (spec §4 Module 4):
    T0  literal replication on the paper's asset/timeframe
    T1  paper mechanism + 1 primary Stage A5 risk-guard filter
    T2  core mechanism generalized to another liquid asset (dynamic params)
    T3  mechanism augmented with a secondary predictive signal

Pruning rules: deterministic plausibility order T0 < T1 < T2 < T3, dedupe on
(target_asset, timeframe, tier), cap at 5. No LLM decides pruning — FDR
control stays deterministic (Rules.md §1 KISS).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl

from black_box.block_a.models import (
    CausalAbstractionSchema,
    ExecutableStrategySpec,
    GenericPrimitiveNode,
    OperandNode,
    OperatorNode,
    PaperExtractionSchema,
    ResourceVerdict,
    RiskTag,
    StrategyAnnotation,
    StrategyTier,
)

logger = logging.getLogger(__name__)

MAX_SPECS = 5

#: OperandNode ids that need not be registry primitives: OHLCV data-series
#: fields and bare numeric constants (e.g. an EMA window of 20).
AST_DATA_SERIES = frozenset({"open", "high", "low", "close", "volume", "vwap"})

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "config" / "primitives_registry.json"
)

#: Risk-tag → primary risk-guard filter (Tier 1), first-listed wins.
RISK_GUARD_FILTERS: dict[RiskTag, str] = {
    RiskTag.EXECUTION_SLIPPAGE_HEAVY: "FLT_VOLUME_RATIO",
    RiskTag.LOW_LIQUIDITY_FRAGILITY: "FLT_VOLUME_RATIO",
    RiskTag.HIGH_SESSION_SENSITIVITY: "FLT_TIME_OF_DAY",
    RiskTag.PARAMETRIC_OVERFIT_RISK: "FLT_VOLATILITY_THRESHOLD",
}

#: Annotation union ordering — highest guard priority first.
RISK_PRIORITY: list[RiskTag] = list(RISK_GUARD_FILTERS)

#: Ordered (phrase → primitive) — first match wins (deterministic).
_ENTRY_PHRASES: tuple[tuple[str, str], ...] = (
    ("band breakout", "TRIGGER_BAND_BREAKOUT"),
    ("breakout of the band", "TRIGGER_BAND_BREAKOUT"),
    ("break out of the band", "TRIGGER_BAND_BREAKOUT"),
    ("cross above", "TRIGGER_CROSS_ABOVE"),
    ("crosses above", "TRIGGER_CROSS_ABOVE"),
    ("crossing above", "TRIGGER_CROSS_ABOVE"),
    ("break above", "TRIGGER_CROSS_ABOVE"),
    ("breaks above", "TRIGGER_CROSS_ABOVE"),
    ("above the upper", "TRIGGER_CROSS_ABOVE"),
    ("rises above", "TRIGGER_CROSS_ABOVE"),
    ("reclaim", "TRIGGER_CROSS_ABOVE"),
    ("cross below", "TRIGGER_CROSS_BELOW"),
    ("crosses below", "TRIGGER_CROSS_BELOW"),
    ("crossing below", "TRIGGER_CROSS_BELOW"),
    ("break below", "TRIGGER_CROSS_BELOW"),
    ("breaks below", "TRIGGER_CROSS_BELOW"),
    ("below the lower", "TRIGGER_CROSS_BELOW"),
    ("falls below", "TRIGGER_CROSS_BELOW"),
    ("drops below", "TRIGGER_CROSS_BELOW"),
    ("touch the lower", "TRIGGER_CROSS_BELOW"),
    ("touches the lower", "TRIGGER_CROSS_BELOW"),
    ("dips below", "TRIGGER_CROSS_BELOW"),
    ("below the vwap", "TRIGGER_CROSS_BELOW"),
)

_EXIT_PHRASES: tuple[tuple[str, str], ...] = (
    ("returns to the middle", "TRIGGER_CROSS_ABOVE"),
    ("crosses back above", "TRIGGER_CROSS_ABOVE"),
    ("cross back above", "TRIGGER_CROSS_ABOVE"),
    ("reclaim", "TRIGGER_CROSS_ABOVE"),
    ("crosses below", "TRIGGER_CROSS_BELOW"),
    ("cross below", "TRIGGER_CROSS_BELOW"),
    ("breaks below", "TRIGGER_CROSS_BELOW"),
    ("break below", "TRIGGER_CROSS_BELOW"),
    ("falls below", "TRIGGER_CROSS_BELOW"),
    ("drops below", "TRIGGER_CROSS_BELOW"),
    ("below the vwap", "TRIGGER_CROSS_BELOW"),
    ("stop-loss", "TRIGGER_CROSS_BELOW"),
    ("stop loss", "TRIGGER_CROSS_BELOW"),
    ("trailing stop", "TRIGGER_CROSS_BELOW"),
    ("band breakout", "TRIGGER_BAND_BREAKOUT"),
    ("crosses above", "TRIGGER_CROSS_ABOVE"),
    ("cross above", "TRIGGER_CROSS_ABOVE"),
)

_INDICATOR_PHRASES: tuple[tuple[str, str], ...] = (
    ("vwap", "IND_VWAP"),
    ("volume weighted average", "IND_VWAP"),
    ("volume-weighted average", "IND_VWAP"),
    ("ema", "IND_EMA"),
    ("exponential moving average", "IND_EMA"),
    ("moving average", "IND_EMA"),
    ("bollinger", "IND_EMA"),
    ("middle band", "IND_EMA"),
    ("atr", "IND_ATR"),
    ("average true range", "IND_ATR"),
    ("volume delta", "IND_VOLUME_DELTA"),
    ("cumulative delta", "IND_VOLUME_DELTA"),
    ("delta volume", "IND_VOLUME_DELTA"),
    ("order flow delta", "IND_VOLUME_DELTA"),
)

_TIMEFRAME_RE = re.compile(
    r"(?i)\b(\d+)\s*(?:-|\s)?(minutes?|mins?|m|hours?|hrs?|h|days?|d)\b"
)

_TIER_ORDER: tuple[StrategyTier, ...] = (
    StrategyTier.TIER_0_LITERAL,
    StrategyTier.TIER_1_PARAMETRIC,
    StrategyTier.TIER_2_GENERALIZED,
    StrategyTier.TIER_3_AUGMENTED,
)

_UNIT_NORMALIZE: dict[str, str] = {
    "m": "m",
    "min": "m",
    "mins": "m",
    "minute": "m",
    "minutes": "m",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_registry(path: Path | None = None) -> dict[str, list[str]]:
    """Load the primitive vocabulary (triggers, indicators, filters)."""
    target = path or DEFAULT_REGISTRY_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    return {
        "triggers": list(data.get("triggers", [])),
        "indicators": list(data.get("indicators", [])),
        "filters": list(data.get("filters", [])),
    }


# ---------------------------------------------------------------------------
# Deterministic text → primitive mapping
# ---------------------------------------------------------------------------


def map_entry_primitive(text: str, registry: dict[str, list[str]]) -> str:
    lowered = text.lower()
    for phrase, primitive in _ENTRY_PHRASES:
        if phrase in lowered and primitive in registry["triggers"]:
            return primitive
    return "TRIGGER_CROSS_ABOVE"  # documented fallback


def map_exit_primitive(text: str, registry: dict[str, list[str]]) -> str:
    lowered = text.lower()
    for phrase, primitive in _EXIT_PHRASES:
        if phrase in lowered and primitive in registry["triggers"]:
            return primitive
    return "TRIGGER_CROSS_BELOW"  # documented fallback (protective bias)


def detect_indicators(text: str, registry: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for phrase, primitive in _INDICATOR_PHRASES:
        if (
            phrase in lowered
            and primitive in registry["indicators"]
            and primitive not in found
        ):
            found.append(primitive)
    return found


def detect_timeframe(text: str, fallback: str) -> str:
    """Extract the chart timeframe ('15-minute' → '15m'), else the fallback."""
    match = _TIMEFRAME_RE.search(text or "")
    if not match:
        return fallback
    unit = _UNIT_NORMALIZE.get(match.group(2).lower())
    if not unit:
        return fallback
    return f"{int(match.group(1))}{unit}"


# ---------------------------------------------------------------------------
# Stage A5 → guards
# ---------------------------------------------------------------------------


def risk_union(annotations: list[StrategyAnnotation]) -> list[RiskTag]:
    """Union of risk tags across annotations, ordered by guard priority."""
    seen: set[RiskTag] = set()
    for ann in annotations:
        seen.update(ann.risk_tags)
    return [tag for tag in RISK_PRIORITY if tag in seen]


def primary_guard_filter(annotations: list[StrategyAnnotation]) -> str | None:
    """Highest-priority risk-guard filter (Tier 1), None when no tags."""
    tags = risk_union(annotations)
    return RISK_GUARD_FILTERS[tags[0]] if tags else None


# ---------------------------------------------------------------------------
# GenericPrimitiveNode helpers (Stage A6 AST)
# ---------------------------------------------------------------------------


def validate_ast_registry(
    node: GenericPrimitiveNode, registry: dict[str, list[str]]
) -> list[str]:
    """Return primitive ids referenced by `node` that do NOT resolve.

    Compile-time registry guard for the recursive AST: indicator operands
    must name a registry indicator; transform/literal operands may also name
    an OHLCV data series (``AST_DATA_SERIES``) or a bare numeric constant.
    An empty list means the tree is fully resolvable. Deterministic, no LLM.
    """
    if isinstance(node, OperandNode):
        allowed = set(registry["indicators"]) | AST_DATA_SERIES
        if node.kind == "indicator":
            return [] if node.id in set(registry["indicators"]) else [node.id]
        if node.id in allowed or str(node.id).replace(".", "", 1).isdigit():
            return []
        return [node.id]
    if isinstance(node, OperatorNode):
        unresolved: list[str] = []
        unresolved += validate_ast_registry(node.left, registry)
        if node.right is not None:
            unresolved += validate_ast_registry(node.right, registry)
        return unresolved
    return []  # DataStreamNode — no primitive ids


def secondary_signal_ast(secondary: str) -> GenericPrimitiveNode:
    """Tier 3 predictive-signal tree: `ZScore(<secondary indicator>)`.

    The standardized secondary signal (e.g. IND_VOLUME_DELTA) — flat
    `parameters.secondary_signal` remains the execution projection.
    """
    return OperatorNode(
        kind="unary_op",
        op="zscore",
        left=OperandNode(kind="indicator", id=secondary),
    )


# ---------------------------------------------------------------------------
# Catalog helpers (Tier 2 generalization targets)
# ---------------------------------------------------------------------------


def _alternate_symbol(catalog_path: Path, anchor: str) -> str | None:
    """Deterministic Tier 2 target: prefer another asset class, then symbol."""
    df = pl.read_parquet(catalog_path)
    candidates = df.filter(pl.col("symbol") != anchor)
    if candidates.is_empty():
        return None
    anchor_class = df.filter(pl.col("symbol") == anchor)["asset_class"].to_list()
    anchor_class = anchor_class[0] if anchor_class else ""
    candidates = candidates.with_columns(
        (pl.col("asset_class") == anchor_class).alias("_same_class")
    )
    ranked = (
        candidates.sort(["_same_class", "symbol"])
        .filter(pl.col("has_l2_book") == True)
        .head(1)
    )
    if ranked.is_empty():  # fall back to any alternate symbol
        ranked = candidates.sort(["_same_class", "symbol"]).head(1)
    if ranked.is_empty():
        return None
    return str(ranked["symbol"][0])


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


def compile_specs(
    extraction: PaperExtractionSchema,
    abstraction: CausalAbstractionSchema | None,
    annotations: list[StrategyAnnotation],
    verdicts: list[ResourceVerdict],
    *,
    catalog_path: Path,
    registry_path: Path | None = None,
    max_specs: int = MAX_SPECS,
) -> list[ExecutableStrategySpec]:
    """Compile the pruned, tiered strategy matrix (Stage A6 output).

    `verdicts` come from the Stage A3 hard check: if any dataset is
    `missing`, the DAG already hard-stopped before reaching this function —
    the compiler never emits an untestable spec from a failed resource check.
    """
    registry = load_registry(registry_path)

    if not extraction.datasets_used:
        logger.info("no dataset anchor in extraction — emitting no specs")
        return []

    anchor = extraction.datasets_used[0].symbol.upper()
    timeframe_fallback = extraction.datasets_used[0].required_granularity.value
    mechanism = extraction.core_mechanism or extraction.primary_hypothesis
    text = f"{mechanism} {abstraction.abstract_hypothesis if abstraction else ''}"

    entry = map_entry_primitive(text, registry)
    exit_ = map_exit_primitive(text, registry)
    indicators = detect_indicators(text, registry)
    timeframe = detect_timeframe(mechanism, fallback=timeframe_fallback)
    tags = risk_union(annotations)
    guard_filter = primary_guard_filter(annotations)
    anchor_notes = abstraction.causal_anchor if abstraction else mechanism
    invariants = list(abstraction.non_negotiable_invariants) if abstraction else []

    provenance = {v.symbol: v for v in verdicts}
    verified = anchor in provenance and provenance[anchor].status != "missing"

    # No-drop guarantee mirrored here: the compiler NEVER emits when the
    # anchor dataset is unverified — in the DAG the hard-stop already
    # prevented reaching this stage for any missing verdict.
    if not verified:
        logger.info("anchor %s not verified in catalog — emitting no specs", anchor)
        return []

    candidates: list[ExecutableStrategySpec] = []

    # --- Tier 0: literal replication on the paper's asset/timeframe --------
    candidates.append(
        ExecutableStrategySpec(
            spec_id=_slug(entry, anchor, StrategyTier.TIER_0_LITERAL),
            tier=StrategyTier.TIER_0_LITERAL,
            target_asset=anchor,
            timeframe=timeframe,
            entry_trigger_primitive=entry,
            exit_trigger_primitive=exit_,
            filter_primitives=[],
            parameters={
                "mode": "static",
                "indicators": indicators,
                "origin": "paper_literal",
                "non_negotiable_invariants": invariants,
            },
            risk_annotations=tags,
            causal_anchor_notes=anchor_notes,
        )
    )

    # --- Tier 1: paper mechanism + 1 primary risk-guard filter ---------
    if guard_filter is not None:
        candidates.append(
            ExecutableStrategySpec(
                spec_id=_slug(entry, anchor, StrategyTier.TIER_1_PARAMETRIC),
                tier=StrategyTier.TIER_1_PARAMETRIC,
                target_asset=anchor,
                timeframe=timeframe,
                entry_trigger_primitive=entry,
                exit_trigger_primitive=exit_,
                filter_primitives=[guard_filter],
                parameters={
                    "mode": "static",
                    "indicators": indicators,
                    "origin": "paper_literal_with_guard",
                    "risk_guard_filter": guard_filter,
                },
                risk_annotations=tags,
                causal_anchor_notes=anchor_notes,
            )
        )

    # --- Tier 3: augmented with a secondary predictive signal ----------
    secondary = (
        "IND_VOLUME_DELTA" if "IND_VOLUME_DELTA" not in indicators else "IND_EMA"
    )
    if secondary in registry["indicators"]:
        candidates.append(
            ExecutableStrategySpec(
                spec_id=_slug(entry, anchor, StrategyTier.TIER_3_AUGMENTED),
                tier=StrategyTier.TIER_3_AUGMENTED,
                target_asset=anchor,
                timeframe=timeframe,
                entry_trigger_primitive=entry,
                exit_trigger_primitive=exit_,
                filter_primitives=[guard_filter] if guard_filter else [],
                parameters={
                    "mode": "dynamic",
                    "indicators": indicators,
                    "origin": "augmented_with_secondary_signal",
                    "secondary_signal": secondary,
                },
                signal_ast=secondary_signal_ast(secondary),
                risk_annotations=tags,
                causal_anchor_notes=anchor_notes,
            )
        )

    # --- Tier 2: generalized to another liquid asset (dynamic params) ------
    alternate = _alternate_symbol(catalog_path, anchor)
    if alternate:
        candidates.append(
            ExecutableStrategySpec(
                spec_id=_slug(entry, alternate, StrategyTier.TIER_2_GENERALIZED),
                tier=StrategyTier.TIER_2_GENERALIZED,
                target_asset=alternate,
                timeframe=timeframe,
                entry_trigger_primitive=entry,
                exit_trigger_primitive=exit_,
                filter_primitives=[guard_filter] if guard_filter else [],
                parameters={
                    "mode": "dynamic",
                    "indicators": indicators,
                    "origin": f"generalized_from:{anchor}",
                    "non_negotiable_invariants": invariants,
                },
                risk_annotations=tags,
                causal_anchor_notes=(
                    f"{anchor_notes} | generalized from {anchor} to {alternate}"
                ),
            )
        )

    # --- Prune: tier order, dedupe, cap ------------------------------------
    candidates.sort(key=lambda s: _TIER_ORDER.index(s.tier))
    pruned: list[ExecutableStrategySpec] = []
    seen: set[tuple[str, str, StrategyTier]] = set()
    used_ids: set[str] = set()
    for spec in candidates:
        key = (spec.target_asset, spec.timeframe, spec.tier)
        if key in seen:
            continue
        seen.add(key)
        if spec.spec_id in used_ids:
            spec.spec_id = f"{spec.spec_id}_{len(pruned) + 1}"
        used_ids.add(spec.spec_id)
        pruned.append(spec)
        if len(pruned) >= max_specs:
            break

    logger.info(
        "compiled %d/%d candidate specs for %s (anchor=%s)",
        len(pruned),
        len(candidates),
        extraction.paper_title,
        anchor,
    )
    return pruned


def _slug(entry_primitive: str, symbol: str, tier: StrategyTier) -> str:
    """Deterministic spec id: STRAT_<ENTRY>_<SYMBOL>_T<n>."""
    entry_short = entry_primitive.replace("TRIGGER_", "")
    tier_n = _TIER_ORDER.index(tier)
    asset = re.sub(r"[^A-Za-z0-9]", "", symbol).upper()
    return f"STRAT_{entry_short}_{asset}_T{tier_n}"


def validate_and_export(specs: list[ExecutableStrategySpec], out_path: Path) -> Path:
    """Round-trip validate then write the JSON array to `out_path`.

    Raises `pydantic.ValidationError` if the serialized payload does not
    re-validate — the export contract (spec §6 item 4).
    """
    payload = [s.model_dump(mode="json") for s in specs]
    text = json.dumps(payload)
    revalidated: list[ExecutableStrategySpec] = [
        ExecutableStrategySpec.model_validate(item) for item in json.loads(text)
    ]
    assert len(revalidated) == len(specs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("exported %d specs -> %s", len(specs), out_path)
    return out_path


__all__ = [
    "AST_DATA_SERIES",
    "DEFAULT_REGISTRY_PATH",
    "MAX_SPECS",
    "RISK_GUARD_FILTERS",
    "compile_specs",
    "detect_indicators",
    "detect_timeframe",
    "load_registry",
    "map_entry_primitive",
    "map_exit_primitive",
    "primary_guard_filter",
    "risk_union",
    "secondary_signal_ast",
    "validate_and_export",
    "validate_ast_registry",
]
