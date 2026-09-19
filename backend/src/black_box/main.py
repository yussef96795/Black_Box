"""Black Box — FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from black_box.api.routes import api_router, feasibility_router
from black_box.api.strategy_routes import strategy_router
from black_box.core.config import get_settings
from black_box.services.docling_service import DoclingService

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the Docling converter lifecycle (loaded once per worker)."""
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
app.include_router(feasibility_router, prefix=settings.api_prefix)
app.include_router(strategy_router, prefix=settings.api_prefix)


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
