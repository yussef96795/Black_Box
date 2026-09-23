"""Module 5 — Human Gatekeeper (spec §4 Module 5).

Renders the compiled spec matrix (tiers, primitives, risk tags, data
verdicts) for human sign-off before emission to Block B. The gatekeeper —
not Stage A5 — holds veto authority: rejecting here writes nothing.

`rich` CLI (per spec) rather than streamlit: zero JS deps, scriptable,
testable with `auto_approve`.
"""

from __future__ import annotations

import sys
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from black_box.block_a.models import (
    BlockAState,
    ExecutableStrategySpec,
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


def confirm_gatekeeper(state: BlockAState) -> Literal["complete", "rejected"]:
    """Interactive terminal review; auto-complete when no spec matrix to show.

    `state.specs` are already validated on entry (compiled by Stage A6);
    this only decides approval.
    """
    specs = [ExecutableStrategySpec.model_validate(s) for s in state.specs]
    title = ""
    if state.extraction:
        title = str(state.extraction.get("paper_title", ""))

    console.print(Panel(f"📄 {title or state.paper_id}", border_style="blue"))
    console.print(render_specs(specs))

    if not specs:
        console.print("[yellow]No executable specs compiled for this paper.[/yellow]")
        return "complete"

    answer = input("Approve and emit to Block B? [y/N] ").strip().lower()
    return "complete" if answer in {"y", "yes"} else "rejected"


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


def _noop(state: BlockAState) -> Literal["complete", "rejected"]:
    return "complete"


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
