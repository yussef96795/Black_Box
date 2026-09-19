"""Black Box — FastAPI application entrypoint (Block A Step 1 skeleton).

Brings up the ASGI app with lifespan-managed Docling converter, the /health
probe, and mounts the Block A ingestion router. Uses lazy converter
initialization so `uvicorn` can boot without a heavy Docling warm-up during
imports (Rules.md §3: async I/O, structured exceptions).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from black_box.api.routes import api_router
from black_box.core.config import get_settings
from black_box.services.docling_service import DoclingService

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the DoclingDocument converter lifecycle (loaded once per worker)."""
    docling = DoclingService()
    app.state.docling = docling
    logger.info("Docling service ready (accelerator=%s)", docling.accel())
    yield
    await docling.aclose()


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Quant strategy document engineering platform (Blocks A-D)",
    lifespan=lifespan,
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health", tags=["ops"], include_in_schema=False)
async def health() -> dict:
    """Liveness probe — returns service identity + version."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "docling": "ready"
        if getattr(app.state, "docling", None) is not None
        else "pending",
    }
