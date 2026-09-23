"""Tests for Block A Stage A3 — DuckDB catalog + hard data check."""

from __future__ import annotations

from pathlib import Path

from black_box.block_a.catalog import (
    build_catalog,
    check_catalog,
    granularities_available,
    load_catalog,
)
from black_box.block_a.models import (
    DataGranularity,
    DatasetRequirement,
)
from black_box.services.feasibility_auditor import audit_dependencies

# A small catalog exercising all three verdict paths in one file.
ROWS = [
    {
        "asset_class": "Crypto",
        "symbol": "BTCUSDT",
        "granularity": "tick",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "BTCUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "BTCUSDT",
        "granularity": "1d",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ADAUSDT",
        "granularity": "1m",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
]


def make_catalog(tmp_path: Path) -> Path:
    return build_catalog(tmp_path / "catalog.parquet", rows=ROWS)


def test_build_catalog_roundtrip(tmp_path) -> None:
    path = make_catalog(tmp_path)
    df = load_catalog(path)
    assert df.shape == (4, 6)
    assert set(df["symbol"].to_list()) == {"BTCUSDT", "ADAUSDT"}


def test_granularities_available(tmp_path) -> None:
    path = make_catalog(tmp_path)
    assert granularities_available(path) == {"tick", "1m", "1d"}


def test_available_verdict(tmp_path) -> None:
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="BTCUSDT",
                required_granularity=DataGranularity.MINUTE_1,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is True
    v = result.verdicts[0]
    assert v.status == "available"
    assert "2020" in v.note


def test_l2_proxy_verdict(tmp_path) -> None:
    """L2 demanded but only OHLCV → approved proxy (volume delta on 1m)."""
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="ADAUSDT",
                required_granularity=DataGranularity.MINUTE_1,
                requires_l2_book=True,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is True  # proxy keeps the paper actionable
    v = result.verdicts[0]
    assert v.status == "proxy"
    assert v.proxy_source == "volume_delta:1m"


def test_missing_symbol_hard_stop(tmp_path) -> None:
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Equities",
                symbol="QQQ",
                required_granularity=DataGranularity.MINUTE_1,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is False
    assert result.verdicts[0].status == "missing"


def test_missing_granularity_hard_stop(tmp_path) -> None:
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="BTCUSDT",
                required_granularity=DataGranularity.SECOND_1,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is False
    assert "1s" in result.verdicts[0].note


def test_missing_order_flow_hard_stop(tmp_path) -> None:
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="ADAUSDT",
                required_granularity=DataGranularity.MINUTE_1,
                requires_order_flow=True,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is False
    assert result.verdicts[0].status == "missing"


def test_start_year_too_late_hard_stop(tmp_path) -> None:
    """Data starting after the paper's required year is not sufficient."""
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="ADAUSDT",
                required_granularity=DataGranularity.MINUTE_1,
                start_year=2019,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is False
    assert "2021" in result.verdicts[0].note


def test_symbol_normalized_to_upper(tmp_path) -> None:
    result = check_catalog(
        [
            DatasetRequirement(
                asset_class="Crypto",
                symbol="btcusdt",
                required_granularity=DataGranularity.MINUTE_1,
            )
        ],
        make_catalog(tmp_path),
    )
    assert result.all_available is True


def test_auditor_granularity_resolves_against_catalog(tmp_path) -> None:
    """Auditor wired to the catalog (integration wiring, path unchanged)."""
    path = make_catalog(tmp_path)
    deps = audit_dependencies(
        "Need daily data for the arbitrage bot on Binance.", catalog_path=path
    )
    daily = next(d for d in deps if d.name == "granularity:daily")
    assert daily.status == "satisfied"
    deps2 = audit_dependencies(
        "Need order book data for the flow bot on Binance.", catalog_path=path
    )
    ob = next(d for d in deps2 if d.name == "granularity:orderbook")
    assert ob.status == "satisfied"  # tick rows imply orderbook capability


def test_auditor_catalog_absent_granularity_missing(tmp_path) -> None:
    """Catalog without tick/orderbook rows → orderbook demand is missing."""
    path = build_catalog(
        tmp_path / "catalog_ohlcv.parquet",
        rows=[
            {
                "asset_class": "Crypto",
                "symbol": "BTCUSDT",
                "granularity": "1m",
                "has_l2_book": False,
                "has_order_flow": False,
                "start_year": 2020,
            },
        ],
    )
    deps = audit_dependencies(
        "Need order book data for the flow bot on Binance.", catalog_path=path
    )
    ob = next(d for d in deps if d.name == "granularity:orderbook")
    assert ob.status == "missing"  # catalog authority: no tick rows → no book


def test_auditor_without_catalog_falls_back(tmp_path) -> None:
    """No catalog → legacy static capability table (backwards compatible)."""
    deps = audit_dependencies(
        "Need tick data for the scalp bot on Binance.",
        catalog_path=tmp_path / "absent.parquet",
    )
    tick = next(d for d in deps if d.name == "granularity:tick (L3)")
    assert tick.status == "satisfied"
