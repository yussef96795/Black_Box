"""Tests for Block A Step 2 — Feasibility Auditor Gate."""

from __future__ import annotations

from black_box.schemas import TableValidationReport
from black_box.services.feasibility_auditor import audit, audit_dependencies

# A crypto-native, rule-based strategy with no TradFi or ML dependencies.
PASSED_TEXT = (
    "Long when RSI(14) < 30. Sell when close_t crosses the upper band. "
    "Entry uses Binance perpetual funding data with daily data granularity."
)


def test_audit_passes_crypto_native_strategy() -> None:
    result = audit(PASSED_TEXT)
    assert result.status == "PASSED"
    names = [c.name for c in result.checks]
    assert names == [
        "hard_dependencies",
        "algorithmic_compute",
        "domain_mapping",
        "tables",
    ]
    assert all(c.status == "PASSED" for c in result.checks)
    assert "fully feasible" in result.summary


def test_audit_requires_hitl_for_tradfi_provider_with_analog() -> None:
    text = (
        "React to FRED treasury yield announcements and trade only during "
        "traditional trading hours."
    )
    result = audit(text)
    assert result.status == "REQUIRES_HITL"
    dep = next(c for c in result.checks if c.name == "hard_dependencies")
    assert dep.status == "REQUIRES_HITL"
    deps = dep.details["dependencies"]
    fred = next(d for d in deps if d["name"] == "fred")
    assert fred["status"] == "partial"


def test_audit_rejects_black_box_model() -> None:
    text = (
        "Predict next-bar direction with an LSTM neural network, then trade "
        "Binance perpetuals."
    )
    result = audit(text)
    assert result.status == "REJECTED"
    compute = next(c for c in result.checks if c.name == "algorithmic_compute")
    assert compute.status == "REJECTED"
    assert any(m["name"] == "neural network" for m in compute.details["models"])


def test_audit_requires_hitl_for_compute_hints() -> None:
    text = (
        "Use GPU-accelerated low-latency inference for signal generation on "
        "Binance data."
    )
    result = audit(text)
    assert result.status == "REQUIRES_HITL"
    compute = next(c for c in result.checks if c.name == "algorithmic_compute")
    assert compute.status == "REQUIRES_HITL"
    hints = compute.details["compute_budget"]["hints"]
    assert "gpu" in hints and "low-latency inference" in hints


def test_audit_requires_hitl_for_weak_domain_analog() -> None:
    text = "Trigger trades after FOMC announcements using Coinbase feeds."
    result = audit(text)
    assert result.status == "REQUIRES_HITL"
    domain = next(c for c in result.checks if c.name == "domain_mapping")
    assert domain.status == "REQUIRES_HITL"
    mapping = domain.details["mapping"]
    fomc = next(m for m in mapping if m["tradfi"] == "fomc")
    assert fomc["confidence"] < 0.7


def test_audit_requires_hitl_for_unreadable_tables() -> None:
    bad_table = TableValidationReport(
        table_id="table_0",
        status="parse-error",
        rows=1,
        columns=3,
        errors=["row 2 col 'Value' (abc): 'abc' is not a number"],
    )
    result = audit(PASSED_TEXT, tables=[bad_table])
    assert result.status == "REQUIRES_HITL"
    tables = next(c for c in result.checks if c.name == "tables")
    assert tables.status == "REQUIRES_HITL"


def test_dependencies_missing_when_no_analog() -> None:
    deps = audit_dependencies("Pull equity data from Bloomberg terminal.")
    bloomberg = next(d for d in deps if d.name == "bloomberg")
    assert bloomberg.status == "missing"


def test_dependencies_satisfied_granularity() -> None:
    deps = audit_dependencies("Need tick data for the arbitrage bot on Binance.")
    names = [d.name for d in deps]
    assert any(n == "granularity:tick (L3)" for n in names)
    assert all(d.status == "satisfied" for d in deps)


# --- API integration ---------------------------------------------------------


def test_get_capabilities(client) -> None:
    resp = client.get("/api/v1/feasibility/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "1.0.0"
    assert len(body["capabilities"]) >= 3
    kinds = {c["type"] for c in body["capabilities"]}
    assert {"data", "compute", "domain"} <= kinds


def test_post_audit_endpoint(client) -> None:
    resp = client.post("/api/v1/feasibility/audit", json={"text": PASSED_TEXT})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "PASSED"
    assert len(body["checks"]) == 4


def test_post_audit_empty_text_422(client) -> None:
    resp = client.post("/api/v1/feasibility/audit", json={"text": "", "tables": []})
    assert resp.status_code == 422


def test_post_audit_tables_reported(client) -> None:
    payload = {
        "text": PASSED_TEXT,
        "tables": [
            {
                "table_id": "table_0",
                "status": "parse-error",
                "rows": 1,
                "columns": 3,
                "errors": ["bad"],
            }
        ],
    }
    resp = client.post("/api/v1/feasibility/audit", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "REQUIRES_HITL"
