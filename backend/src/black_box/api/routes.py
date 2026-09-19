"""REST layer for the Black_Box API (Block A ingestion routes).

Serves the Docling-driven ingestion endpoint described in plan.md Block A
Step 1. Handlers stay thin: they validate the request shape, call the
service, and map exceptions — no document logic lives here
(Rules.md §1 SRP + §2 keep functions short).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from black_box.core.capabilities import DEFAULT_CAPABILITIES, Capabilities
from black_box.schemas import (
    FeasibilityAuditRequest,
    FeasibilityResult,
    IngestResponse,
)
from black_box.services.docling_service import DoclingService, get_docling_service
from black_box.services.feasibility_auditor import audit

api_router = APIRouter(prefix="/documents", tags=["documents"])
feasibility_router = APIRouter(prefix="/feasibility", tags=["feasibility"])

_UNPROCESSABLE = status.HTTP_422_UNPROCESSABLE_CONTENT

logger = logging.getLogger(__name__)

# Single mounted router object. `main.py` imports this as `api_router` and
# prefixes it with the configured API version (`Settings.api_prefix`).


@api_router.post(
    "/ingest", status_code=status.HTTP_201_CREATED, response_model=IngestResponse
)
async def ingest_document(
    file: Annotated[
        UploadFile, File(description="PDF / DOCX / HTML / Markdown document")
    ],
    service: Annotated[DoclingService, Depends(get_docling_service)],
) -> IngestResponse:
    """Parse and structure an uploaded document into Docling chunks.

    This is the Block A Step 1 entrypoint. Block C/D consume the returned
    `chunks` for subset stress validation and AST transpilation.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="A file part named 'file' is required.",
        )
    try:
        return await service.ingest(file)
    except HTTPException:
        raise
    except Exception as exc:  # map unexpected service failure -> 502
        logger.exception("Docling ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ingestion pipeline error: {exc}",
        ) from exc


# --- Block A Step 2: Feasibility Auditor Gate --------------------------------


@feasibility_router.get("/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    """Expose the platform capability contract the auditor tests against."""
    return DEFAULT_CAPABILITIES


@feasibility_router.post("/audit", response_model=FeasibilityResult)
async def audit_feasibility(request: FeasibilityAuditRequest) -> FeasibilityResult:
    """Run the four checkers (dependencies, compute, domain, tables) in sequence.

    Deterministic rule-based verdict: any REJECTED check rejects the strategy,
    else any REQUIRES_HITL check routes it to the human-in-the-loop gate
    (Block B), else the strategy PASSES.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=_UNPROCESSABLE,
            detail="Request 'text' must be non-empty.",
        )
    return audit(request.text, request.tables)
