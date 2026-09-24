"""Block B — Cynical Auditor (adversarial *well-posedness* check).

Deterministic for v1. Asks only: *can this formulation be evaluated
honestly?* — never "is it profitable" (that is Block C's job; letting two
stages judge quality produces contradictory gates). Findings:
lookahead/lead references in the signal AST, missing exit or sizing,
unbounded sweep params, degenerate parameter windows.

The AuditFlag contract (``state.schema.AuditFlag``) is unchanged from the
original plan so the LLM-backed adversarial pass can drop in later without
touching the graph topology (ponytail: rule → LLM upgrade path is a node
internal swap).
"""

from __future__ import annotations

from typing import Any


def audit_variants(state: Any) -> list[dict[str, Any]]:
    """Run the adversarial rule set over every variant."""
    flags: list[dict[str, Any]] = []
    for variant in state.variants or []:
        vid = variant["variant_id"]
        flags.extend(_audit_pillars(variant, vid))
        flags.extend(_audit_grid(variant, vid))
        flags.extend(_audit_lookahead(state.spec or {}, variant, vid))
    return flags


def _audit_pillars(variant: dict[str, Any], vid: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if variant.get("exit") is None:
        flags.append(
            {
                "type": "missing_exit",
                "severity": "high",
                "variant_id": vid,
                "description": f"{vid}: no exit rule — a position can never be closed.",
                "suggestion": "Assign an exit component (e.g. exit.atr_stop).",
            }
        )
    if variant.get("sizing") is None:
        flags.append(
            {
                "type": "no_sizing",
                "severity": "high",
                "variant_id": vid,
                "description": f"{vid}: no position sizing — capital allocation undefined.",
                "suggestion": "Assign a sizing component (e.g. sizing.fractional).",
            }
        )
    entry = variant.get("entry") or {}
    if entry.get("component_id") == "entry.ema_cross":
        fast, slow = (
            entry.get("params", {}).get("fast", 0),
            entry.get("params", {}).get("slow", 0),
        )
        if slow <= fast:
            flags.append(
                {
                    "type": "degenerate_window",
                    "severity": "high",
                    "variant_id": vid,
                    "description": f"{vid}: slow EMA ({slow}) <= fast EMA ({fast}) — the cross can never trigger.",
                    "suggestion": "Set slow > fast.",
                }
            )
    return flags


def _audit_grid(variant: dict[str, Any], vid: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for pillar in ("entry", "exit", "sizing"):
        cfg = variant.get(pillar)
        for name, bound in (cfg or {}).get("grid", {}).items():
            if "lo" not in bound or "hi" not in bound:
                flags.append(
                    {
                        "type": "unbounded_param",
                        "severity": "medium",
                        "variant_id": vid,
                        "pillar": pillar,
                        "param": name,
                        "description": f"{vid}/{pillar}.{name}: sweep param has no bounds — the surface is meaningless.",
                        "suggestion": f"Add lo/hi bounds for {name}.",
                    }
                )
    return flags


def _audit_lookahead(
    spec: dict[str, Any], _variant: dict[str, Any], vid: str
) -> list[dict[str, Any]]:
    """Flag lead references: lag/returns with a negative (future) shift."""
    ast = spec.get("signal_ast")
    if not ast or not isinstance(ast, dict):
        return []
    hits = _find_lead(ast)
    if not hits:
        return []
    return [
        {
            "type": "lookahead_bias",
            "severity": "critical",
            "variant_id": vid,
            "description": f"{vid}: signal uses future data {hits} — backtest results would be rigged.",
            "suggestion": "Replace the future-shifted term before backtesting.",
        }
    ]


def _find_lead(node: dict[str, Any] | None) -> list[str]:
    """Walk the AST dict for operators shifted into the future."""
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    if node.get("op") in ("lag", "returns"):
        right = node.get("right")
        if isinstance(right, dict) and right.get("kind") == "literal":
            value = float(right.get("id"))
            if value < 0:
                found.append(f"{node['op']}({value:g})")
    found.extend(_find_lead(node.get("left")))
    found.extend(_find_lead(node.get("right")))
    return found


__all__ = ["audit_variants"]
