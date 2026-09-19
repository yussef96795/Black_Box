"""Variable Resolution Module (Block A Step 1c).

Maps math symbols / LaTeX fragments in chunk text to canonical text
definitions, keeping the original token stream plus a resolved symbol table
for Block C (statistical validation) to consume.

Symbols Docling emits as unicode (σ, ×) and inline LaTeX (`$...$`, `\\(...\\)`)
are both handled. Resolution is regex + curated table — no heavyweight math
engine (KISS, Rules.md §1).
"""

from __future__ import annotations

import re

from black_box.schemas import MathResolution

# Canonical definitions for symbols we expect in quant strategy documents.
# NOTE: LaTeX keys use a SINGLE backslash — Docling emits `\sigma`, not `\\sigma`.
SYMBOL_TABLE: dict[str, str] = {
    # Greek used in finance (unicode — what Docling emits for rich text)
    "σ": "sigma (annualized volatility)",
    "α": "alpha (excess return)",
    "β": "beta (market sensitivity)",
    "λ": "lambda (decay / risk aversion)",
    "μ": "mu (expected return)",
    "ρ": "rho (correlation)",
    "τ": "tau (time constant)",
    "Σ": "summation",
    "Δ": "delta (sensitivity / change)",
    "Γ": "gamma (convexity)",
    "Θ": "theta (time decay)",
    # LaTeX fragments that may appear after Docling math extraction
    r"\sigma": "sigma (annualized volatility)",
    r"\alpha": "alpha (excess return)",
    r"\beta": "beta (market sensitivity)",
    r"\lambda": "lambda (decay / risk aversion)",
    r"\mu": "mu (expected return)",
    r"\rho": "rho (correlation)",
    r"\sum": "summation",
    r"\Delta": "delta (change)",
    r"\sqrt": "square root",
    r"\infty": "infinity",
    # Math operators / relations
    "×": "multiplied by",
    "÷": "divided by",
    "≤": "less than or equal to",
    "≥": "greater than or equal to",
    "±": "plus or minus",
}

_INLINE_LATEX = re.compile(r"\$([^$]+)\$|\\\(([^)]+)\\\)")


def resolve_math_in_text(text: str, context: str = "") -> MathResolution | None:
    """Return a MathResolution if `text` contains resolvable math, else None."""
    used: dict[str, str] = {}
    resolved = text

    for match in _INLINE_LATEX.finditer(text):
        raw = match.group(0)
        body = match.group(1) or match.group(2) or ""
        for sym, definition in SYMBOL_TABLE.items():
            if sym in body:
                used[sym] = definition
                resolved = resolved.replace(raw, definition)
                break
        else:
            used[raw] = f"unresolved LaTeX: {body}"

    # Replace any remaining standalone symbols (unicode / operators)
    for sym, definition in SYMBOL_TABLE.items():
        if sym in resolved:
            used[sym] = definition
            resolved = resolved.replace(sym, definition)

    if not used:
        return None

    return MathResolution(
        original=text.strip(),
        resolved=resolved.strip(),
        symbol_table=used,
        context=context,
    )
