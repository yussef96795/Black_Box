"""Stage A5 quant reasoning operators (spec §4 Module 3).

Single source of truth for the 8 operator definitions. The LLM evaluator
frames one structured call around these; the instruction strings double as
documentation and as the prompt body (schema-first, no raw text transfer).
"""

from __future__ import annotations

#: (operator_name, instruction) — exactly the 8 operators from the spec.
OPERATOR_INSTRUCTIONS: tuple[tuple[str, str], ...] = (
    (
        "Restrict",
        (
            "Identify regimes where the mechanism fails (volatility, session, "
            "liquidity, regime shifts)."
        ),
    ),
    (
        "Invert",
        ("Identify conditions under which the opposite trade holds."),
    ),
    (
        "Relax",
        (
            "Substitute point-in-time discrete triggers with continuous distance "
            "metrics (e.g. gap to band instead of touch)."
        ),
    ),
    (
        "Substitute",
        (
            "Identify alternative primitive features capturing the same intent "
            "(e.g. volume delta instead of tick imbalances)."
        ),
    ),
    (
        "Decompose",
        ("Isolate the entry signal driver vs. the risk/exit driver."),
    ),
    (
        "Combine",
        ("Identify complementary structural filters that strengthen the edge."),
    ),
    (
        "Adversarial",
        (
            "Evaluate market structures designed to break this edge (spoofing, "
            "arbitrage, liquidity traps)."
        ),
    ),
    (
        "Generalize",
        ("Identify candidate asset classes sharing the same market microstructure."),
    ),
)

OPERATOR_NAMES: tuple[str, ...] = tuple(name for name, _ in OPERATOR_INSTRUCTIONS)


def operator_prompt_block() -> str:
    """Render the operator instruction list for an LLM prompt."""
    return "\n".join(
        f"- {name}: {instruction}" for name, instruction in OPERATOR_INSTRUCTIONS
    )


__all__ = ["OPERATOR_INSTRUCTIONS", "OPERATOR_NAMES", "operator_prompt_block"]
