"""Block B — Formulation Engine (deterministic core + HITL layer).

Consumes the curated Block A spec; produces backtest-ready formulations
(variant triples + sweep grids) via the LangGraph DAG in
``black_box.state``. Deterministic v1: the Cynical Auditor and structural
gate are rule-based; LLM-backed adversarial pass is the documented upgrade
(see ``auditor.py``).
"""

from black_box.block_b.auditor import audit_variants
from black_box.block_b.formulation import (
    MAX_VARIANTS,
    hydrate_spec,
    pillar_payload,
    structural_gate,
    unbacked_primitives,
)
from black_box.block_b.hitl import apply_answers, build_cards, load_state, save_state

__all__ = [
    "MAX_VARIANTS",
    "apply_answers",
    "audit_variants",
    "build_cards",
    "hydrate_spec",
    "load_state",
    "pillar_payload",
    "save_state",
    "structural_gate",
    "unbacked_primitives",
]
