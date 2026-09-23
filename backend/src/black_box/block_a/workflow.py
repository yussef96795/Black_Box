"""Block A — LangGraph DAG + run engine (spec §3 workflow).

Topology:

    parse_document → extract_requirements → check_resources ──┬─(ok)→ abstract_mechanism
                                                              └─(missing)→ hard_stop → END
    abstract_mechanism → apply_operators → compile_specs → gatekeeper → END

Statelessness (spec §1 invariant 1): every `run()` compiles a fresh graph
with a fresh `MemorySaver` and a fresh evaluator — no checkpoint or LLM
context survives across paper runs. Nodes are plain functions over
`BlockAState` (msgpack-safe primitives only).

The hard-stop router is the ONLY place a strategy can be dropped — Stage A5
operator annotations carry zero deletion authority (No-Drop rule).
"""

from __future__ import annotations

import hashlib
import logging
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

try:  # instructor ≥1.6 canonical location; fall back for older pins
    from instructor.core import InstructorRetryException
except ImportError:  # pragma: no cover
    from instructor.exceptions import InstructorRetryException  # type: ignore[no-redef]

from black_box.block_a.catalog import check_catalog, ensure_catalog
from black_box.block_a.gatekeeper import confirm_gatekeeper
from black_box.block_a.llm_evaluator import LLMEvaluator
from black_box.block_a.models import (
    RESOURCE_INSUFFICIENT_ERROR,
    BlockAResult,
    BlockAState,
    CausalAbstractionSchema,
    ExecutableStrategySpec,
    PaperExtractionSchema,
    ResourceCheckResult,
    StrategyAnnotation,
)
from black_box.block_a.pdf_parser import parse_paper
from black_box.block_a.spec_compiler import (
    compile_specs,
    validate_and_export,
)
from black_box.block_a.traces import TraceWriter

logger = logging.getLogger(__name__)

Parser = Callable[[str | Path], list[dict[str, Any]]]
Gatekeeper = Callable[[BlockAState], Literal["complete", "rejected"]]

#: Status literal routing an LLM failure edge to the terminal `llm_failed` node.
LLM_FAILED_STATUS = "llm_extraction_failed"


def _call_stage(stage: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one LLM stage; convert exhausted-retry failures into state.

    Catches the two recoverable failure classes — `InstructorRetryException`
    (validation retries exhausted after `max_retries`) and `ValidationError`
    (schema drift) — and surfaces them as a terminal `llm_extraction_failed`
    state update carrying ``{stage, error_type, detail, trace_hash}``. The
    full traceback goes to the logs (and the P3 trace file); only the hash
    travels in the payload. Anything else propagates as an unexpected DAG
    failure (CLI exit 1).
    """
    try:
        return fn()
    except (InstructorRetryException, ValidationError) as exc:
        stack = traceback.format_exc()
        logger.exception("LLM stage %s failed after retries", stage)
        return {
            "stage_error": {
                "stage": stage,
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "trace_hash": hashlib.sha256(stack.encode("utf-8")).hexdigest()[:16],
            },
            "status": LLM_FAILED_STATUS,
        }


def _llm_router(next_node: str) -> Callable[[BlockAState], str]:
    """Conditional-edge router: exhausted LLM retries → `llm_failed`, else on."""

    def route(state: BlockAState) -> Literal["llm_failed", str]:
        return "llm_failed" if state.status == LLM_FAILED_STATUS else next_node

    return route


def build_block_a_graph(
    *,
    evaluator: LLMEvaluator,
    parser: Parser = parse_paper,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    approve: bool | None = None,
    gatekeeper: Gatekeeper = confirm_gatekeeper,
) -> Any:
    """Compile the Block A DAG with injected dependencies (testable, stateless)."""
    catalog_path = ensure_catalog(catalog_path)

    def parse_document(state: BlockAState) -> dict[str, Any]:
        chunks = parser(state.source_path)
        return {"chunks": chunks}

    def extract_requirements(state: BlockAState) -> dict[str, Any]:
        return _call_stage(
            "extract_requirements",
            lambda: {
                "extraction": evaluator.extract_paper(state.chunks).model_dump(
                    mode="json"
                )
            },
        )

    def check_resources(state: BlockAState) -> dict[str, Any]:
        extraction = PaperExtractionSchema.model_validate(state.extraction)
        result: ResourceCheckResult = check_catalog(
            extraction.datasets_used, catalog_path
        )
        return {"resource_check": result.model_dump(mode="json")}

    def hard_stop(state: BlockAState) -> dict[str, Any]:
        return {
            "status": "resource_insufficient",
            "error_code": RESOURCE_INSUFFICIENT_ERROR,
            "specs": [],
        }

    def abstract_mechanism(state: BlockAState) -> dict[str, Any]:
        return _call_stage(
            "abstract_mechanism",
            lambda: {
                "abstraction": evaluator.abstract_mechanism(
                    PaperExtractionSchema.model_validate(state.extraction),
                    state.chunks,
                ).model_dump(mode="json")
            },
        )

    def apply_operators(state: BlockAState) -> dict[str, Any]:
        return _call_stage(
            "apply_operators",
            lambda: {
                "annotations": [
                    a.model_dump(mode="json")
                    for a in evaluator.apply_operators(
                        PaperExtractionSchema.model_validate(state.extraction),
                        CausalAbstractionSchema.model_validate(state.abstraction),
                        state.chunks,
                    )
                ],
                "status": "gatekeeper",
            },
        )

    def compile_specs_node(state: BlockAState) -> dict[str, Any]:
        extraction = PaperExtractionSchema.model_validate(state.extraction)
        abstraction = (
            CausalAbstractionSchema.model_validate(state.abstraction)
            if state.abstraction
            else None
        )
        annotations = [StrategyAnnotation.model_validate(a) for a in state.annotations]
        resource = ResourceCheckResult.model_validate(state.resource_check)
        specs = compile_specs(
            extraction,
            abstraction,
            annotations,
            resource.verdicts,
            catalog_path=catalog_path,
            registry_path=registry_path,
        )
        return {"specs": [s.model_dump(mode="json") for s in specs]}

    def gatekeeper_node(state: BlockAState) -> dict[str, Any]:
        decide: Gatekeeper = confirm_gatekeeper
        if approve is True:
            decide = lambda _s: "complete"
        elif approve is False:
            decide = lambda _s: "rejected"
        elif gatekeeper is not None:
            decide = gatekeeper
        return {"status": decide(state)}

    def route_after_resource(
        state: BlockAState,
    ) -> Literal["abstract_mechanism", "hard_stop"]:
        if (
            state.resource_check
            and ResourceCheckResult.model_validate(state.resource_check).all_available
        ):
            return "abstract_mechanism"
        return "hard_stop"

    def llm_failed(state: BlockAState) -> dict[str, Any]:
        # Terminal: nothing further runs; partial results stay in state.
        return {"specs": []}

    workflow = StateGraph(BlockAState)
    workflow.add_node("parse_document", parse_document)
    workflow.add_node("extract_requirements", extract_requirements)
    workflow.add_node("check_resources", check_resources)
    workflow.add_node("hard_stop", hard_stop)
    workflow.add_node("abstract_mechanism", abstract_mechanism)
    workflow.add_node("apply_operators", apply_operators)
    workflow.add_node("llm_failed", llm_failed)
    workflow.add_node("compile_specs", compile_specs_node)
    workflow.add_node("gatekeeper", gatekeeper_node)

    workflow.set_entry_point("parse_document")
    workflow.add_edge("parse_document", "extract_requirements")
    # After every LLM stage: exhausted retries → terminal `llm_failed`, else
    # continue. Only the Stage A3 catalog check may hard-stop (it does NOT go
    # through this router).
    workflow.add_conditional_edges(
        "extract_requirements",
        _llm_router("check_resources"),
        {"check_resources": "check_resources", "llm_failed": "llm_failed"},
    )
    workflow.add_conditional_edges(
        "check_resources",
        route_after_resource,
        {"abstract_mechanism": "abstract_mechanism", "hard_stop": "hard_stop"},
    )
    workflow.add_conditional_edges(
        "abstract_mechanism",
        _llm_router("apply_operators"),
        {"apply_operators": "apply_operators", "llm_failed": "llm_failed"},
    )
    workflow.add_conditional_edges(
        "apply_operators",
        _llm_router("compile_specs"),
        {"compile_specs": "compile_specs", "llm_failed": "llm_failed"},
    )
    workflow.add_edge("hard_stop", END)
    workflow.add_edge("llm_failed", END)
    workflow.add_edge("compile_specs", "gatekeeper")
    workflow.add_edge("gatekeeper", END)

    return workflow.compile(checkpointer=MemorySaver())


class BlockAEngine:
    """Runs one paper through the Block A DAG and exports the spec matrix."""

    def __init__(
        self,
        *,
        evaluator: LLMEvaluator | None = None,
        parser: Parser = parse_paper,
        catalog_path: Path | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.parser = parser
        self.catalog_path = catalog_path
        self.registry_path = registry_path

    def run(
        self,
        paper_path: str | Path,
        *,
        approve: bool | None = None,
        out_dir: str | Path | None = None,
    ) -> BlockAResult:
        """Process one paper in complete isolation.

        Stateless by construction: a brand-new graph + MemorySaver is built
        per call, so nothing can leak between papers.
        """
        evaluator = self.evaluator
        if evaluator is None:
            from black_box.block_a.llm_evaluator import ollama_client

            evaluator = LLMEvaluator(ollama_client())

        graph = build_block_a_graph(
            evaluator=evaluator,
            parser=self.parser,
            catalog_path=self.catalog_path,
            registry_path=self.registry_path,
            approve=approve,
        )

        paper_id = uuid.uuid4().hex
        writer = _trace_writer_for(paper_id, out_dir)
        evaluator.set_trace(writer, paper_id)
        initial = BlockAState(
            paper_id=paper_id,
            source_path=str(paper_path),
            status="running",
        )
        config = {"configurable": {"thread_id": paper_id}}
        graph.invoke(initial, config=config)
        # `invoke` returns a plain dict; read the persisted state back from
        # the checkpoint (house pattern in strategy_routes.py) and re-validate
        # it into the Block A state model.
        saved = graph.get_state(config)
        final = BlockAState.model_validate(saved.values)

        title = ""
        if final.extraction:
            title = str(final.extraction.get("paper_title", ""))

        if final.status == "complete":
            specs = [ExecutableStrategySpec.model_validate(s) for s in final.specs]
            out_dir = out_dir or _default_out_dir()
            out_file = validate_and_export(specs, Path(out_dir) / "block_a_specs.json")
            tags = sorted({t for s in specs for t in s.risk_annotations})
            return BlockAResult(
                paper_id=paper_id,
                paper_title=title,
                status="complete",
                specs=specs,
                risk_tags=tags,
                out_path=str(out_file),
            )

        if final.status == "rejected":
            return BlockAResult(
                paper_id=paper_id,
                paper_title=title,
                status="rejected",
            )

        if final.status == LLM_FAILED_STATUS:
            # Partial results (e.g. extraction succeeded, A4 failed) remain in
            # final state — nothing dropped; only the export is withheld.
            return BlockAResult(
                paper_id=paper_id,
                paper_title=title,
                status=LLM_FAILED_STATUS,
                error=final.stage_error,
            )

        return BlockAResult(
            paper_id=paper_id,
            paper_title=title,
            status="resource_insufficient",
            error_code=final.error_code or RESOURCE_INSUFFICIENT_ERROR,
        )


def _default_out_dir() -> Path:
    from black_box.core.config import get_settings

    return get_settings().block_a_out_dir


def _trace_writer_for(paper_id: str, out_dir: str | Path | None) -> TraceWriter | None:
    """Per-run trace writer, or None when tracing is disabled (D3).

    With an explicit run `out_dir`, traces go to ``<out_dir>/traces/``; with
    the default settings they land in `block_a_trace_dir`. Default off keeps
    tests hermetic — no writer, no directory, no files.
    """
    from black_box.core.config import get_settings

    settings = get_settings()
    if not settings.block_a_trace_enabled:
        return None
    trace_dir = (
        Path(out_dir) / "traces" if out_dir is not None else settings.block_a_trace_dir
    )
    return TraceWriter(trace_dir / f"{paper_id}.jsonl")


__all__ = ["BlockAEngine", "build_block_a_graph"]
