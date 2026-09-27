"""Regenerate the Block A indicator tables from the strategylib registry.

``block_a`` cannot import ``strategylib`` — ``strategylib.synthesize`` already
imports ``block_a.models``, so the reverse edge would be a cycle. The tables are
therefore *generated into* the checked-in JSON rather than read at runtime, and
this script is what keeps them in step.

Two things are written:

``indicators``
    the allowlist of feed ids, which is what ``detect_indicators`` and
    ``validate_ast_registry`` gate on. Before this existed the list held four ids
    and any other indicator in a paper silently resolved to nothing.
``phrases.indicators``
    the ``[phrase, feed_id]`` rows, which replaced a table duplicated by hand in
    both this file and ``spec_compiler.py``.

Run it after adding, renaming or re-synonymising any indicator::

    uv run python -m black_box.strategylib.indicators._generate

A ``--check`` flag reports drift without writing, so a test or CI step can fail
loudly when someone adds an indicator and forgets. The generic contract test
calls it.

ponytail: this is a script, not a build step — there is no generated-file header
and no "do not edit" banner, because the file it writes is a normal checked-in
config that a human may reasonably want to hand-tune. The tradeoff is that a
hand edit is silently reverted by the next run; the ``--check`` flag is the
mitigation, and it is exercised by the test suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from black_box.strategylib.indicators import FEEDS, INDICATORS, phrase_table
from black_box.strategylib.indicators._deferred import deferred_report

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "block_a"
    / "config"
    / "primitives_registry.json"
)


def build_tables() -> dict[str, Any]:
    """The generated fragments, ready to be merged into the registry dict."""
    return {
        "indicators": sorted(FEEDS),
        "phrases": {"indicators": [[phrase, feed] for phrase, feed in phrase_table()]},
    }


def merge(registry: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    """Return ``registry`` with the generated tables replaced, other keys intact."""
    merged = dict(registry)
    merged["indicators"] = tables["indicators"]
    phrases = dict(merged.get("phrases", {}))
    phrases["indicators"] = tables["phrases"]["indicators"]
    merged["phrases"] = phrases
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero instead of writing",
    )
    args = parser.parse_args(argv)

    tables = build_tables()
    current = json.loads(args.registry.read_text(encoding="utf-8"))
    merged = merge(current, tables)

    if args.check:
        drifted = [
            key
            for key in ("indicators", "phrases")
            if json.dumps(merged.get(key), sort_keys=True)
            != json.dumps(current.get(key), sort_keys=True)
        ]
        if drifted:
            print(
                f"primitives_registry.json is out of date ({', '.join(drifted)}); "
                f"run `uv run python -m black_box.strategylib.indicators._generate`",
                file=sys.stderr,
            )
            return 1
        print("primitives_registry.json matches the indicator registry")
        return 0

    args.registry.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(merged['indicators'])} indicator feed ids and "
        f"{len(merged['phrases']['indicators'])} phrases to {args.registry}"
    )
    for item in deferred_report().values():
        print(f"  deferred: {item['name']} — breaks {item['breaks']}")
    print(f"  registered: {len(INDICATORS)} indicators, {len(FEEDS)} feeds")
    return 0


if __name__ == "__main__":  # pragma: no cover — script entry
    raise SystemExit(main())
