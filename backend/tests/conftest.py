"""Shared pytest fixtures (Rules.md §testing)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from black_box.main import app

FIXTURES = Path(__file__).parent / "fixtures"
MARKDOWN_STRATEGY = FIXTURES / "strategy_smoke.md"


@pytest.fixture()
def client() -> TestClient:
    """TestClient booting the real FastAPI app (lifespan included)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def strategy_md_bytes() -> bytes:
    return MARKDOWN_STRATEGY.read_bytes()


@pytest.fixture()
def strategy_session_id() -> str:
    """A known test session ID for strategy endpoint tests."""
    return "test-session-block-b"
