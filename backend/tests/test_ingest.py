"""Integration tests for the ingestion pipeline (real Docling converter)."""

from __future__ import annotations

from black_box.core.config import Settings


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Black_Box"
    assert body["version"] == "0.1.0"
    assert body["docling"] in {"ready", "pending"}


def test_ingest_markdown_strategy(client, strategy_md_bytes) -> None:
    resp = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("strategy_smoke.md", strategy_md_bytes, "text/markdown")},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["document_id"]
    assert body["chunk_count"] >= 1
    assert body["total_tokens"] > 0
    assert len(body["chunks"]) == body["chunk_count"]
    for chunk in body["chunks"]:
        assert chunk["id"]
        assert chunk["text"]
        assert chunk["tokens"] >= 0
        assert "headings" in chunk["meta"]
    # The parameters table must surface as a dedicated table chunk
    table_chunks = [
        c for c in body["chunks"] if "table" in c["meta"]["doc_item_labels"]
    ]
    assert table_chunks, "expected at least one table-tagged chunk"
    # 1d: a validated parameters table must be reported as fit
    assert body["tables"], "expected a table validation report"
    assert body["tables"][0]["status"] == "fit"
    assert body["tables"][0]["rows"] >= 1
    # 1c: the volatility symbol (σ) must be resolved
    assert body["math"], "expected at least one math resolution"
    assert any("sigma" in r["resolved"] for r in body["math"])


def test_ingest_missing_file(client) -> None:
    resp = client.post("/api/v1/documents/ingest")
    assert resp.status_code == 422


def test_ingest_empty_file(client) -> None:
    resp = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()


def test_ingest_garbage_pdf(client) -> None:
    resp = client.post(
        "/api/v1/documents/ingest",
        files={"file": ("fake.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
    )
    assert resp.status_code == 422  # client error, not a server failure


def test_settings_allowed_formats() -> None:
    s = Settings(allowed_formats="pdf,docx,html,md")
    assert s.allowed_format_list == ["PDF", "DOCX", "HTML", "MD"]


def test_settings_max_upload_default() -> None:
    s = Settings()
    assert s.max_upload_size == 25 * 1024 * 1024
