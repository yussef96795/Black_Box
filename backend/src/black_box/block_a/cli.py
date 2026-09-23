"""Block A CLI entry point (spec §5 directory layout: `main.py`).

    python -m black_box.block_a.cli <paper.pdf> [--yes] [--out DIR] [--llm fake]

Exit codes:
    0  complete (specs approved + emitted, or no specs to emit)
    1  unexpected pipeline failure
    2  RESOURCE_INSUFFICIENT_ERROR (hard Stage A3 data stop)
    3  paper could not be parsed
    4  LLM_EXTRACTION_FAILED (a stage exhausted its validation retries)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from black_box.block_a.gatekeeper import log_result
from black_box.block_a.llm_evaluator import LLMEvaluator, fake_evaluator, ollama_client
from black_box.block_a.models import (
    CausalAbstractionSchema,
    DataGranularity,
    DatasetRequirement,
    PaperExtractionSchema,
    StrategyAnnotation,
)
from black_box.block_a.pdf_parser import UnsupportedPaperError, parse_paper
from black_box.block_a.workflow import BlockAEngine
from black_box.core.config import get_settings

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_RESOURCE_INSUFFICIENT = 2
EXIT_UNPARSEABLE = 3
EXIT_LLM_FAILED = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="block-a",
        description=(
            "Block A — Alpha Feasibility & Strategy Ingestion Engine. "
            "Ingests a research paper, verifies data availability, runs the "
            "8 quant operators, and emits ≤5 executable strategy specs."
        ),
    )
    parser.add_argument("paper", help="path to the research paper (PDF/MD/HTML)")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="auto-approve at the human gatekeeper (non-interactive / CI)",
    )
    parser.add_argument(
        "--out",
        default=str(get_settings().block_a_out_dir),
        help="output directory for block_a_specs.json (default: ./out)",
    )
    parser.add_argument(
        "--llm",
        choices=["ollama", "fake"],
        default=get_settings().block_a_llm_backend,
        help="LLM backend (default: ollama; 'fake' for tests/dry-runs)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: BLOCK_A_OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="path to data_catalog.parquet (default: packaged seed catalog)",
    )
    return parser


def _fake_evaluator(paper_stem: str):
    """Deterministic canned models for `--llm fake` (no network, tests/CI)."""
    extraction = PaperExtractionSchema(
        paper_title=paper_stem,
        primary_hypothesis="Price displacement vs VWAP creates persistence",
        datasets_used=[
            DatasetRequirement(
                asset_class="Crypto",
                symbol="BTCUSDT",
                required_granularity=DataGranularity.MINUTE_1,
                requires_l2_book=True,
            )
        ],
        core_mechanism=(
            "Long when price crosses above the VWAP anchor and volume delta "
            "is positive; exit when price crosses below VWAP."
        ),
    )
    abstraction = CausalAbstractionSchema(
        abstract_hypothesis="Anchor-relative displacement predicts short-term persistence",
        causal_anchor="Institutional execution algorithms benchmark against VWAP",
        non_negotiable_invariants=["anchor must reset at session open"],
    )
    annotation = StrategyAnnotation(
        operator_name="Generalize",
        observation="Same microstructure across liquid perps",
        proposed_modification="Generalize anchor to other liquid symbols",
        risk_tags=[],
    )
    evaluator, _ = fake_evaluator(
        extraction=extraction,
        abstraction=abstraction,
        annotations=[annotation] * 8,
    )
    return evaluator


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    paper = Path(args.paper)
    if not paper.exists():
        print(f"error: paper not found: {paper}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        if args.llm == "fake":
            evaluator = _fake_evaluator(paper.stem)
        else:
            evaluator = LLMEvaluator(ollama_client(model=args.model))

        engine = BlockAEngine(
            evaluator=evaluator,
            parser=parse_paper,
            catalog_path=Path(args.catalog) if args.catalog else None,
        )
        result = engine.run(paper, approve=args.yes, out_dir=args.out)
    except KeyboardInterrupt:
        print("aborted by user.")
        return EXIT_FAILURE
    except UnsupportedPaperError as exc:
        print(f"error: unparseable paper: {exc}", file=sys.stderr)
        return EXIT_UNPARSEABLE
    except Exception as exc:  # noqa: BLE001 — CLI boundary, report + nonzero
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    log_result(result.status, result.out_path)
    if result.status == "resource_insufficient":
        return EXIT_RESOURCE_INSUFFICIENT
    if result.status == "llm_extraction_failed":
        err = result.error or {}
        print(
            "error: LLM_EXTRACTION_FAILED at stage "
            f"{err.get('stage', '?')}: {err.get('detail', 'see trace file')}",
            file=sys.stderr,
        )
        return EXIT_LLM_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
