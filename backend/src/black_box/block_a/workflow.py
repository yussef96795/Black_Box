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

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

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

logger = logging.getLogger(__name__)

Parser = Callable[[str | Path], list[dict[str, Any]]]
Gatekeeper = Callable[[BlockAState], Literal["complete", "rejected"]]


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
        extraction: PaperExtractionSchema = evaluator.extract_paper(state.chunks)
        return {"extraction": extraction.model_dump(mode="json")}

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
        extraction = PaperExtractionSchema.model_validate(state.extraction)
        abstraction: CausalAbstractionSchema = evaluator.abstract_mechanism(
            extraction, state.chunks
        )
        return {"abstraction": abstraction.model_dump(mode="json")}

    def apply_operators(state: BlockAState) -> dict[str, Any]:
        extraction = PaperExtractionSchema.model_validate(state.extraction)
        abstraction = CausalAbstractionSchema.model_validate(state.abstraction)
        annotations: list[StrategyAnnotation] = evaluator.apply_operators(
            extraction, abstraction, state.chunks
        )
        return {
            "annotations": [a.model_dump(mode="json") for a in annotations],
            "status": "gatekeeper",
        }

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

    workflow = StateGraph(BlockAState)
    workflow.add_node("parse_document", parse_document)
    workflow.add_node("extract_requirements", extract_requirements)
    workflow.add_node("check_resources", check_resources)
    workflow.add_node("hard_stop", hard_stop)
    workflow.add_node("abstract_mechanism", abstract_mechanism)
    workflow.add_node("apply_operators", apply_operators)
    workflow.add_node("compile_specs", compile_specs_node)
    workflow.add_node("gatekeeper", gatekeeper_node)

    workflow.set_entry_point("parse_document")
    workflow.add_edge("parse_document", "extract_requirements")
    workflow.add_edge("extract_requirements", "check_resources")
    workflow.add_conditional_edges("check_resources", route_after_resource)
    workflow.add_edge("hard_stop", END)
    workflow.add_edge("abstract_mechanism", "apply_operators")
    workflow.add_edge("apply_operators", "compile_specs")
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

        return BlockAResult(
            paper_id=paper_id,
            paper_title=title,
            status="resource_insufficient",
            error_code=final.error_code or RESOURCE_INSUFFICIENT_ERROR,
        )


def _default_out_dir() -> Path:
    from black_box.core.config import get_settings

    return get_settings().block_a_out_dir


__all__ = ["BlockAEngine", "build_block_a_graph"]
