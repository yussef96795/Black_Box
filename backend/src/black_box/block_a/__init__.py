"""Block A — Alpha Feasibility & Strategy Ingestion Engine.

Stages A1–A6 per BLOCK_A_SPECIFICATION.md:

    Module 1  Ingestion                  black_box.block_a.pdf_parser
    Module 2  Extraction + data check    llm_evaluator.extract_paper, catalog
    Module 3  Causal abstraction (A4)    llm_evaluator.abstract_mechanism
              8 quant operators (A5)      llm_evaluator.apply_operators
    Module 4  DSL compiler (A6)           spec_compiler
    Module 5  Human gatekeeper            gatekeeper
    DAG       LangGraph orchestration     workflow.BlockAEngine

Public entry points:
    BlockAEngine(...).run(paper_path, approve=..., out_dir=...) -> BlockAResult
    python -m black_box.block_a.cli <paper>
"""

from black_box.block_a.models import (
    RESOURCE_INSUFFICIENT_ERROR,
    BlockAResult,
    BlockAState,
    CausalAbstractionSchema,
    DataGranularity,
    DatasetRequirement,
    ExecutableStrategySpec,
    PaperExtractionSchema,
    ResourceCheckResult,
    ResourceVerdict,
    RiskTag,
    StrategyAnnotation,
    StrategyTier,
)
from black_box.block_a.workflow import BlockAEngine, build_block_a_graph

__all__ = [
    "RESOURCE_INSUFFICIENT_ERROR",
    "BlockAEngine",
    "BlockAResult",
    "BlockAState",
    "CausalAbstractionSchema",
    "DataGranularity",
    "DatasetRequirement",
    "ExecutableStrategySpec",
    "PaperExtractionSchema",
    "ResourceCheckResult",
    "ResourceVerdict",
    "RiskTag",
    "StrategyAnnotation",
    "StrategyTier",
    "build_block_a_graph",
]
