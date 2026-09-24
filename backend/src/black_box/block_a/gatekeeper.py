"""Module 5 — Human Gatekeeper (spec §4 Module 5, granular per-spec review).

Renders the compiled spec matrix (tiers, primitives, risk tags, data
verdicts) for human sign-off before emission to Block B. The gatekeeper —
not Stage A5 — holds veto authority. `confirm_gatekeeper` returns a
`GatekeeperDecision {status, specs}` (D4): the curated spec array is what
the DAG writes back into `BlockAState.specs`, so subset approval and risk
edits actually affect what `validate_and_export` emits.

`rich` CLI (per spec) rather than streamlit: zero JS deps, scriptable,
testable with `auto_approve` / injected input.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from black_box.block_a.models import (
    RISK_PRIORITY,
    BlockAState,
    ExecutableStrategySpec,
    GatekeeperDecision,
)

console = Console()


def render_specs(specs: list[ExecutableStrategySpec]) -> str:
    """Render the spec matrix as a rich table (returned as a rich renderable)."""
    table = Table(title="Compiled Strategy Specifications (Tier 0–3)")
    table.add_column("Spec ID", style="bold cyan", no_wrap=True)
    table.add_column("Tier")
    table.add_column("Asset")
    table.add_column("Timeframe")
    table.add_column("Entry")
    table.add_column("Exit")
    table.add_column("Filters")
    table.add_column("Risk tags", style="yellow")

    for spec in specs:
        table.add_row(
            spec.spec_id,
            spec.tier.value.removeprefix("TIER_").replace("_", " "),
            spec.target_asset,
            spec.timeframe,
            spec.entry_trigger_primitive,
            spec.exit_trigger_primitive,
            ", ".join(spec.filter_primitives) or "—",
            ", ".join(t.value for t in spec.risk_annotations) or "—",
        )
    return Panel(table, border_style="green")


def _render_single(spec: ExecutableStrategySpec, index: int, total: int) -> str:
    table = Table(title=f"Spec {index}/{total} — {spec.spec_id}")
    table.add_column("Tier")
    table.add_column("Asset")
    table.add_column("Timeframe")
    table.add_column("Entry")
    table.add_column("Exit")
    table.add_column("Filters")
    table.add_column("Risk tags", style="yellow")
    table.add_row(
        spec.tier.value.removeprefix("TIER_").replace("_", " "),
        spec.target_asset,
        spec.timeframe,
        spec.entry_trigger_primitive,
        spec.exit_trigger_primitive,
        ", ".join(spec.filter_primitives) or "—",
        ", ".join(t.value for t in spec.risk_annotations) or "—",
    )
    return Panel(table, border_style="blue")


def _toggle_risk_tags(spec: ExecutableStrategySpec) -> ExecutableStrategySpec:
    """Interactively toggle the RiskTags in canonical priority order."""
    console.print(
        "Risk tags (enter comma/space-separated indices to toggle, blank to keep):"
    )
    for idx, tag in enumerate(RISK_PRIORITY, start=1):
        state = "on" if tag in spec.risk_annotations else "off"
        console.print(f"  {idx}. {tag.value} [{state}]")
    raw = input("toggle > ").strip()
    indices = {
        int(tok) for tok in raw.replace(",", " ").split() if tok.strip().isdigit()
    }
    if not indices:
        return spec
    tags = list(spec.risk_annotations)
    for idx in indices:
        if 1 <= idx <= len(RISK_PRIORITY):
            tag = RISK_PRIORITY[idx - 1]
            if tag in tags:
                tags.remove(tag)
            else:
                tags.append(tag)
    # Re-validate the edited spec before acceptance (schema-first invariant).
    payload = spec.model_dump(mode="json")
    payload["risk_annotations"] = [t.value for t in tags]
    return ExecutableStrategySpec.model_validate(payload)


def _interactive_review(specs: list[ExecutableStrategySpec]) -> GatekeeperDecision:
    """Per-spec [a]pprove / [r]eject / [t]oggle tags / [q]uit loop (D4).

    Back-compat: bare ``n``/``no`` on the FIRST spec rejects the whole run
    (keeps the old single-prompt behavior); ``y``/``yes`` approves the
    current spec and every remaining one. ``q`` aborts as rejected.
    """
    approved: list[ExecutableStrategySpec] = []
    total = len(specs)
    for i, spec in enumerate(specs, start=1):
        current = spec
        while True:
            console.print(_render_single(current, i, total))
            answer = (
                input(
                    f"Spec {i}/{total}: [a]pprove [r]eject [t]oggle risk tags [q]uit > "
                )
                .strip()
                .lower()
            )
            if answer in {"y", "yes"}:
                return GatekeeperDecision(
                    status="complete", specs=approved + [current] + specs[i:]
                )
            if answer in {"a", "approve", ""}:
                approved.append(current)
                break
            if answer in {"r", "reject", "n", "no"}:
                if i == 1 and answer in {"n", "no"}:
                    return GatekeeperDecision(status="rejected", specs=[])
                break  # reject this spec, continue reviewing the rest
            if answer in {"t", "toggle"}:
                current = _toggle_risk_tags(current)
                continue
            if answer in {"q", "quit"}:
                return GatekeeperDecision(status="rejected", specs=approved)
            console.print("[yellow]invalid choice — a/r/t/q[/yellow]")
    return GatekeeperDecision(status="complete", specs=approved)


def confirm_gatekeeper(state: BlockAState) -> GatekeeperDecision:
    """Interactive terminal review; auto-complete when no spec matrix to show.

    Returns the D4 decision: status PLUS the curated spec array, so subset
    approval / risk edits flow into the exported matrix.
    """
    specs = [ExecutableStrategySpec.model_validate(s) for s in state.specs]
    title = ""
    if state.extraction:
        title = str(state.extraction.get("paper_title", ""))

    console.print(Panel(f"📄 {title or state.paper_id}", border_style="blue"))
    console.print(render_specs(specs))

    if not specs:
        console.print("[yellow]No executable specs compiled for this paper.[/yellow]")
        return GatekeeperDecision(status="complete", specs=[])

    return _interactive_review(specs)


def log_result(status: str, out_path: str | None = None) -> None:
    """Human-readable terminal summary of a run."""
    if status == "complete" and out_path:
        console.print(f"[green]✓ Approved — specs emitted to {out_path}[/green]")
    elif status == "complete":
        console.print(
            "[yellow]✓ Complete — no specs emitted (human approved).[/yellow]"
        )
    elif status == "rejected":
        console.print("[red]✗ Rejected by gatekeeper — nothing emitted.[/red]")
    elif status == "llm_extraction_failed":
        console.print(
            "[red]✗ LLM_EXTRACTION_FAILED — no specs emitted; see trace for stack.[/red]"
        )
    else:
        console.print(
            "[red]✗ RESOURCE_INSUFFICIENT_ERROR — required data missing.[/red]"
        )


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 2:
        console.print(
            "[red]usage: python -m black_box.block_a.gatekeeper <specs.json>[/red]"
        )
        raise SystemExit(2)
    import json
    from pathlib import Path

    from black_box.block_a.spec_compiler import load_registry  # noqa: F401

    raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    specs = [ExecutableStrategySpec.model_validate(o) for o in raw]
    console.print(render_specs(specs))


__all__ = [
    "confirm_gatekeeper",
    "log_result",
    "render_specs",
]
