"""Block B — the HITL layer: typed cards, keyed mutations, persistence.

One mechanism: at the interrupt the graph emits a batch of typed
`ClarificationCard`s; a human answers them; `apply_answers` patches the
formulation via keyed mutations (``mutation`` paths like
``variants.0.sizing``), and the graph re-audits. Cards are typed
(missingPillar / missingPrimitive / remediation / universe; riskOverride
reserved) so a resume merge is deterministic, never free text.

A `missingPrimitive` card carries a *preview* of the AST-compiled component
(code already written to `components/generated/`, manifest NOT yet): the
human's `approve` action registers it — mandatory review before generated
code enters the trusted library.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from black_box.block_b.formulation import pillar_payload
from black_box.strategylib import append_manifest, by_pillar, resolve
from black_box.strategylib.synthesize import synthesize_component

# ---------------------------------------------------------------------------
# Card generation
# ---------------------------------------------------------------------------


def build_cards(state: Any) -> list[dict[str, Any]]:
    """Generate the typed clarification-card batch for one HITL round."""
    cards: list[dict[str, Any]] = []
    idx_by_vid = {v["variant_id"]: i for i, v in enumerate(state.variants or [])}

    for err in state.structural_errors or []:
        err_type = err.get("type")
        pillar = err.get("pillar", "")
        idx = idx_by_vid.get(err.get("variant_id", ""), 0)
        if err_type == "missing_pillar":
            cards.append(
                _pillar_card(
                    pillar=pillar,
                    variant_idx=idx,
                    evidence=err.get("message", ""),
                )
            )
        elif err_type == "unbacked_primitive":
            cards.append(_primitive_card(state, err))
        elif err_type == "bad_data":
            cards.append(_universe_card(state))

    for flag in state.audit_flags or []:
        card = _remediation_for_flag(flag, idx_by_vid, state.variants or [])
        if card is not None:
            cards.append(card)

    for i, card in enumerate(cards):
        card["card_id"] = f"C-R{state.round}-{i + 1:03d}"
    return _dedupe(cards)


def _dedupe(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One card per mutation path (keeps the first, most specific card)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for card in cards:
        mutation = card.get("mutation", "")
        key = mutation or f"{card.get('type')}:{str(card.get('title'))[:40]}"
        if key in seen:
            continue
        seen.add(key)
        kept.append(card)
    return kept


def _pillar_card(*, pillar: str, variant_idx: int, evidence: str) -> dict[str, Any]:
    options = [c.id for c in by_pillar(pillar)]
    return {
        "card_id": "pending",
        "type": "missingPillar",
        "pillar": pillar,
        "title": f"Missing {pillar} pillar",
        "evidence": evidence,
        "proposal": {"component_id": options[0] if options else ""},
        "options": options,
        "mutation": f"variants.{variant_idx}.{pillar}",
        "required": True,
        "state": "pending",
    }


def _primitive_card(state: Any, err: dict[str, Any]) -> dict[str, Any]:
    """Preview the AST-compiled component; registration happens on approve."""
    spec = state.spec or {}
    primitive = err.get("primitive", "")
    component_id = primitive.lower().replace("_", ".").replace(".ind.", "ind.")
    preview: dict[str, Any] = {}
    try:
        preview = synthesize_component(
            spec.get("signal_ast") or {"kind": "operand", "id": primitive},
            component_id=component_id,
            pillar="codebase",
            spec_id=spec.get("spec_id", ""),
        )
    except ValueError as exc:
        preview = {"error": str(exc)}
    return {
        "card_id": "pending",
        "type": "missingPrimitive",
        "pillar": "codebase",
        "title": f"Novel primitive: {primitive}",
        "evidence": (
            f"New signal {primitive} has vocabulary but no executable component. "
            "Review the synthesized module and approve registration."
        ),
        "proposal": {"entry": preview, "replaces": err.get("suggestion", "")},
        "options": ["synthesize", "abandon"],
        "mutation": "",
        "required": True,
        "state": "pending",
    }


def _universe_card(state: Any) -> dict[str, Any]:
    spec = state.spec or {}
    return {
        "card_id": "pending",
        "type": "universe",
        "pillar": "universe",
        "title": "Confirm trading universe / timeframe",
        "evidence": "Spec data binding is incomplete or invalid.",
        "proposal": {
            "symbol": spec.get("target_asset", ""),
            "timeframe": spec.get("timeframe", ""),
        },
        "options": [],
        "mutation": "spec",
        "required": True,
        "state": "pending",
    }


def _remediation_for_flag(
    flag: dict[str, Any], idx_by_vid: dict[str, int], variants: list[dict[str, Any]]
) -> dict[str, Any] | None:
    ftype = flag.get("type")
    idx = idx_by_vid.get(flag.get("variant_id", ""), 0)
    pillar = flag.get("pillar", "entry")
    if ftype == "missing_exit":
        mutation, proposal = f"variants.{idx}.exit", {"component_id": "exit.atr_stop"}
    elif ftype == "no_sizing":
        mutation, proposal = (
            f"variants.{idx}.sizing",
            {"component_id": "sizing.fractional"},
        )
    elif ftype == "unbounded_param":
        name = flag.get("param", "")
        if not name:
            return None
        mutation = f"variants.{idx}.{pillar}.grid.{name}"
        # Derive sensible bounds from the value the component defaulted to
        # (closing the hole generically rather than inventing a range).
        variant = variants[idx] if idx < len(variants) else {}
        cfg = variant.get(pillar) or {}
        default = (cfg.get("grid") or {}).get(name, {}).get("default", 0.5)
        lo = default * 0.5 if default > 0 else 0.0
        hi = default * 2.0 if default > 0 else 1.0
        proposal = {
            "default": default,
            "lo": lo,
            "hi": hi,
            "step": round((hi - lo) / 5.0, 6),
        }
    elif ftype == "degenerate_window":
        mutation, proposal = f"variants.{idx}.entry.params", {"fast": 10, "slow": 40}
    elif ftype == "lookahead_bias":
        # No auto-fix: a future-shifted signal cannot be healed generically.
        # Re-audit persists the flag until the round cap rejects (honest).
        return {
            "card_id": "pending",
            "type": "remediation",
            "pillar": "risk",
            "title": flag.get("description", "Lookahead bias in signal"),
            "evidence": flag.get("suggestion", ""),
            "proposal": None,
            "options": ["accept", "reject"],
            "mutation": "",
            "required": True,
            "state": "pending",
        }
    else:
        return None
    return {
        "card_id": "pending",
        "type": "remediation",
        "pillar": "risk"
        if mutation.endswith("params") or ".grid." in mutation
        else pillar,
        "title": flag.get("description", "Remediation required")[:120],
        "evidence": flag.get("suggestion", ""),
        "proposal": proposal,
        "options": ["accept", "override"],
        "mutation": mutation,
        "required": True,
        "state": "pending",
    }


# ---------------------------------------------------------------------------
# Resume: apply keyed mutations
# ---------------------------------------------------------------------------


def apply_answers(state: Any) -> dict[str, Any]:
    """Merge human answers into the formulation via keyed mutations.

    Idempotent: card ids already in ``answers_applied`` are skipped, so a
    double-submitted resume is a no-op (testable).
    """
    variants = _deepcopy(state.variants or [])
    spec = _deepcopy(state.spec or {})
    applied = list(state.answers_applied or [])
    cards = {c.get("card_id"): c for c in state.hitl_cards or []}

    for card_id, answer in (state.user_responses or {}).items():
        if card_id in applied or not isinstance(answer, dict):
            continue
        card = cards.get(card_id)
        if card is None:
            continue
        _apply_one(card, answer, variants, spec)
        applied.append(card_id)

    update: dict[str, Any] = {
        "answers_applied": applied,
        "applied_any": len(applied) > len(state.answers_applied or []),
    }
    if state.variants != variants:
        update["variants"] = variants
    if state.spec != spec:
        update["spec"] = spec
    # Resume idempotency key: the applied batch hash, carried in graph state.
    if state.user_responses:
        update["resume_key"] = hashlib.sha1(
            json.dumps(state.user_responses, sort_keys=True, default=str).encode()
        ).hexdigest()
    return update


def _apply_one(
    card: dict[str, Any],
    answer: dict[str, Any],
    variants: list[dict[str, Any]],
    spec: dict[str, Any],
) -> None:
    action = answer.get("action", "accept")
    value = answer.get("value")
    card_type = card.get("type")

    if card_type == "missingPrimitive":
        entry = (card.get("proposal") or {}).get("entry") or {}
        if action in ("approve", "accept") and entry.get("id") and "error" not in entry:
            append_manifest(entry)
        return

    if card_type == "universe":
        value = value or (card.get("proposal") or {})
        if value.get("symbol"):
            spec["target_asset"] = value["symbol"]
        if value.get("timeframe"):
            spec["timeframe"] = value["timeframe"]
        return

    mutation = card.get("mutation", "")
    if not mutation:
        return

    if action == "accept":
        proposal = card.get("proposal")
        value = proposal if isinstance(proposal, dict) else value
    if (
        isinstance(value, dict)
        and value.get("component_id")
        and mutation.split(".")[-1] in ("entry", "exit", "sizing")
    ):
        # Accept/override re-derives the FULL pillar from the registry
        # (component + params + sweep grid). Setting a raw component id
        # would leave the variant with no sweep surface for Block C.
        component = resolve(value["component_id"])
        value = pillar_payload(component) if component is not None else value
    path = mutation.removeprefix("variants.").removeprefix("spec.")
    _set_path(
        _target(variants, spec, mutation),
        path,
        value if isinstance(value, dict) else {},
    )


def _target(variants: list[dict[str, Any]], spec: dict[str, Any], mutation: str) -> Any:
    return variants if mutation.startswith("variants.") else spec


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        if isinstance(parent, list):
            parent = parent[int(part)]
        else:
            parent = parent.setdefault(part, {})
    if isinstance(parent, list):
        parent[int(parts[-1])] = value
    else:
        parent[parts[-1]] = value


def _deepcopy(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Session persistence (survives process restarts; in-process resume via the
# MemorySaver checkpointer stays the live path).
# ---------------------------------------------------------------------------


def save_state(session_id: str, state_obj: Any, out_dir: Path) -> Path:
    path = Path(out_dir) / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_obj, indent=2, default=str), encoding="utf-8")
    return path


def load_state(session_id: str, out_dir: Path) -> dict[str, Any] | None:
    path = Path(out_dir) / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["apply_answers", "build_cards", "load_state", "save_state"]
