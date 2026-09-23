"""Stage A3 — Data Catalog Interrogator (deterministic, no LLM).

Answers one question per `DatasetRequirement`: can the local Parquet data
lake serve this symbol at this granularity (with L2 / order flow when
demanded)? Verdicts are `available | proxy | missing`.

Proxy rule (spec §4 Module 2): when `requires_l2_book` is True but the
catalog only holds OHLCV for the symbol, query the approved proxy
`volume_delta` on 1-minute bars. If the symbol has 1m OHLCV, the verdict
becomes `proxy`; otherwise the pipeline hard-stops with
`RESOURCE_INSUFFICIENT_ERROR`.

The catalog is a DuckDB-readable Parquet metadata index
(`block_a/config/data_catalog.parquet`) with columns:
    asset_class, symbol, granularity, has_l2_book, has_order_flow, start_year
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from black_box.block_a.models import (
    DataGranularity,
    DatasetRequirement,
    ResourceCheckResult,
    ResourceVerdict,
)

logger = logging.getLogger(__name__)

CATALOG_COLUMNS = [
    "asset_class",
    "symbol",
    "granularity",
    "has_l2_book",
    "has_order_flow",
    "start_year",
]

#: Seed catalog — honest to the platform capability contract in
#: `core/capabilities.py` (crypto L3/1m available; equity/rates unsupported).
#: Symbols marked without L2 exercise the approved volume-delta proxy path.
DEFAULT_CATALOG_ROWS: list[dict[str, Any]] = [
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
        "granularity": "1h",
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
        "symbol": "ETHUSDT",
        "granularity": "tick",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ETHUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ETHUSDT",
        "granularity": "1h",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ETHUSDT",
        "granularity": "1d",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "SOLUSDT",
        "granularity": "tick",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "SOLUSDT",
        "granularity": "1m",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "SOLUSDT",
        "granularity": "1h",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    {
        "asset_class": "Crypto",
        "symbol": "SOLUSDT",
        "granularity": "1d",
        "has_l2_book": True,
        "has_order_flow": True,
        "start_year": 2020,
    },
    # OHLCV-only symbols (no L2 / order flow) — proxy path only.
    {
        "asset_class": "Crypto",
        "symbol": "ADAUSDT",
        "granularity": "1m",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ADAUSDT",
        "granularity": "1h",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
    {
        "asset_class": "Crypto",
        "symbol": "ADAUSDT",
        "granularity": "1d",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
    {
        "asset_class": "Crypto",
        "symbol": "DOGEUSDT",
        "granularity": "1m",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
    {
        "asset_class": "Crypto",
        "symbol": "DOGEUSDT",
        "granularity": "1h",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
    {
        "asset_class": "Crypto",
        "symbol": "DOGEUSDT",
        "granularity": "1d",
        "has_l2_book": False,
        "has_order_flow": False,
        "start_year": 2021,
    },
]

#: Approved Stage A3 proxy when L2 book is demanded but only OHLCV exists.
PROXY_L2 = "volume_delta"

#: Granularity the approved L2 proxy is built on.
PROXY_L2_GRANULARITY = DataGranularity.MINUTE_1


def default_catalog_path() -> Path:
    """Path of the seeded catalog shipped inside the package config dir."""
    return Path(__file__).resolve().parent / "config" / "data_catalog.parquet"


def build_catalog(path: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    """Write the Parquet metadata index (idempotent, overwrites in place)."""
    data = rows if rows is not None else DEFAULT_CATALOG_ROWS
    df = pl.DataFrame(data, schema=CATALOG_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("data catalog written: %s (%d rows)", path, len(data))
    return path


def ensure_catalog(path: Path | None = None) -> Path:
    """Seed the shipped catalog if it does not exist yet."""
    target = path or default_catalog_path()
    if not target.exists():
        build_catalog(target)
    return target


def load_catalog(path: Path) -> pl.DataFrame:
    """Load the catalog into a polars DataFrame (DuckDB-readable parquet)."""
    return pl.read_parquet(path)


def granularities_available(path: Path) -> set[str]:
    """Distinct granularities present in the catalog (auditor integration)."""
    try:
        df = load_catalog(path)
    except Exception:  # noqa: BLE001 — missing/unreadable catalog degrades
        return set()
    return set(df["granularity"].to_list())


def check_catalog(
    requirements: list[DatasetRequirement], path: Path
) -> ResourceCheckResult:
    """Stage A3: verify every dataset requirement against the catalog.

    Deterministic and side-effect free. Callers (the DAG router) hard-stop
    when `all_available` is False — the No-Drop rule only permits *this*
    gate to halt a strategy.
    """
    df = load_catalog(path)
    verdicts: list[ResourceVerdict] = []

    for req in requirements:
        verdicts.append(_check_one(df, req))

    return ResourceCheckResult(
        verdicts=verdicts,
        all_available=all(v.status != "missing" for v in verdicts),
    )


def _check_one(df: pl.DataFrame, req: DatasetRequirement) -> ResourceVerdict:
    symbol, granularity = req.symbol.upper(), req.required_granularity
    symbol_rows = df.filter(pl.col("symbol") == symbol)

    if symbol_rows.is_empty():
        return ResourceVerdict(
            asset_class=req.asset_class,
            symbol=symbol,
            granularity=granularity,
            status="missing",
            note=f"symbol '{symbol}' absent from data catalog",
        )

    # The catalog is the authority on asset class once a symbol resolves —
    # never propagate an LLM-mislabeled class downstream (deterministic fix).
    asset_class = str(symbol_rows["asset_class"][0])

    exact = symbol_rows.filter(pl.col("granularity") == granularity.value)
    if exact.is_empty():
        return ResourceVerdict(
            asset_class=asset_class,
            symbol=symbol,
            granularity=granularity,
            status="missing",
            note=f"no {granularity.value} data for '{symbol}' in catalog",
        )

    row = exact.row(0, named=True)

    if req.start_year is not None and row["start_year"] > req.start_year:
        return ResourceVerdict(
            asset_class=asset_class,
            symbol=symbol,
            granularity=granularity,
            status="missing",
            note=(
                f"data for '{symbol}' starts {row['start_year']}, later than "
                f"required {req.start_year}"
            ),
        )

    if req.requires_l2_book and not row["has_l2_book"]:
        has_proxy_bar = not symbol_rows.filter(
            pl.col("granularity") == PROXY_L2_GRANULARITY.value
        ).is_empty()
        if has_proxy_bar:
            return ResourceVerdict(
                asset_class=asset_class,
                symbol=symbol,
                granularity=granularity,
                status="proxy",
                proxy_source=f"{PROXY_L2}:{PROXY_L2_GRANULARITY.value}",
                note=(
                    f"no L2 book for '{symbol}'; approved proxy "
                    f"{PROXY_L2} on {PROXY_L2_GRANULARITY.value} bars"
                ),
            )
        return ResourceVerdict(
            asset_class=asset_class,
            symbol=symbol,
            granularity=granularity,
            status="missing",
            note=f"L2 book demanded but not available for '{symbol}', and no proxy exists in catalog",
        )

    if req.requires_order_flow and not row["has_order_flow"]:
        return ResourceVerdict(
            asset_class=asset_class,
            symbol=symbol,
            granularity=granularity,
            status="missing",
            note=f"order flow demanded but not available for '{symbol}' (no approved proxy)",
        )

    return ResourceVerdict(
        asset_class=asset_class,
        symbol=symbol,
        granularity=granularity,
        status="available",
        note=f"verified: {symbol} {granularity.value} in catalog from {row['start_year']}",
    )


__all__ = [
    "CATALOG_COLUMNS",
    "DEFAULT_CATALOG_ROWS",
    "build_catalog",
    "check_catalog",
    "default_catalog_path",
    "ensure_catalog",
    "granularities_available",
    "load_catalog",
]
