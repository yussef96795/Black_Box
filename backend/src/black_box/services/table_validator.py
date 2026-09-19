"""Table Schema Validator (Block A Step 1d).

Extracts Docling `TABLE` items from a converted document and validates each
against an expected strategy-parameter schema, emitting a per-table report
(TableValidationReport — defined in schemas.py as the API contract):

    table_id   — stable id (table_<idx>)
    status     — fit | orphan | parse-error
    rows, cols — grid dimensions
    errors     — human-readable violations

Rules.md §1 SRP: this module only knows about table validation; the service
layer feeds it a DoclingDocument and serializes the report.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from black_box.schemas import TableValidationReport


class ColumnSpec(BaseModel):
    """Expected type contract for a single table column."""

    name: str
    kind: Literal["string", "number", "range", "percent"] = "string"


class TableSpec(BaseModel):
    """Expected schema for a strategy-parameters table."""

    name: str
    match_headers: list[str]
    columns: list[ColumnSpec] = Field(default_factory=list)


# Default expected schema for the canonical Black_Box strategy doc:
# a parameters table with Param / Value / Bounds columns.
DEFAULT_TABLE_SCHEMAS: list[TableSpec] = [
    TableSpec(
        name="parameters",
        match_headers=["Param", "Value", "Bounds"],
        columns=[
            ColumnSpec(name="Param", kind="string"),
            ColumnSpec(name="Value", kind="number"),
            ColumnSpec(name="Bounds", kind="range"),
        ],
    )
]

_RANGE_RE = re.compile(r"^\s*\[?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]?\s*$")


def _check_cell(value: str, kind: str) -> str | None:
    """Return an error message when `value` violates `kind`, else None."""
    value = value.strip()
    if not value:
        return "empty cell"
    if kind == "number":
        try:
            float(value)
        except ValueError:
            return f"'{value}' is not a number"
    elif kind == "percent":
        cleaned = value.rstrip("%").strip()
        try:
            float(cleaned)
        except ValueError:
            return f"'{value}' is not a percent"
    elif kind == "range":
        m = _RANGE_RE.match(value)
        if not m:
            return f"'{value}' is not a [min, max] range"
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            return f"range {value}: lower bound > upper bound"
    return None


def _column_index(spec: TableSpec, name: str) -> int | None:
    """Index of the grid column named `name` (None when not present)."""
    try:
        return spec.match_headers.index(name)
    except ValueError:
        return None


def validate_tables(
    doc: Any, specs: list[TableSpec] | None = None
) -> list[TableValidationReport]:
    """Validate every table in a DoclingDocument against `specs`.

    Falls back to `DEFAULT_TABLE_SCHEMAS`. Tables whose headers match a spec
    are checked row-by-row (status ``fit`` or ``parse-error``); tables with no
    matching spec are reported as ``orphan`` so Block C can decide whether to
    consume or ignore them. Columns are matched by header name, so specs may
    cover a subset of the grid columns.
    """
    specs = specs or DEFAULT_TABLE_SCHEMAS
    reports: list[TableValidationReport] = []
    tables = list(getattr(doc, "tables", []) or [])

    for idx, table in enumerate(tables):
        table_id = f"table_{idx}"
        data = getattr(table, "data", None)
        grid = getattr(data, "grid", None) if data else None
        if not grid or not grid[0]:
            reports.append(
                TableValidationReport(
                    table_id=table_id,
                    status="parse-error",
                    errors=["empty/unreadable grid"],
                )
            )
            continue

        header = [c.text.strip() for c in grid[0]]
        spec = next((s for s in specs if s.match_headers == header), None)

        if spec is None:
            reports.append(
                TableValidationReport(
                    table_id=table_id,
                    status="orphan",
                    rows=max(len(grid) - 1, 0),
                    columns=len(header),
                    errors=[f"no schema matches headers {header}"],
                )
            )
            continue

        errors: list[str] = []
        col_index = {col.name: _column_index(spec, col.name) for col in spec.columns}
        for r, row in enumerate(grid[1:], start=2):  # 1-indexed data rows
            cells = [c.text if hasattr(c, "text") else str(c) for c in row]
            for col in spec.columns:
                pos = col_index[col.name]
                if pos is None:
                    errors.append(
                        f"row {r}: column '{col.name}' not found in headers {header}"
                    )
                    continue
                if pos >= len(cells):
                    errors.append(
                        f"row {r}: expected column '{col.name}', row truncated"
                    )
                    continue
                msg = _check_cell(cells[pos], col.kind)
                if msg:
                    errors.append(f"row {r} col '{col.name}' ({cells[pos]}): {msg}")

        status: Literal["fit", "parse-error"] = "fit" if not errors else "parse-error"
        reports.append(
            TableValidationReport(
                table_id=table_id,
                status=status,
                rows=max(len(grid) - 1, 0),
                columns=len(header),
                errors=errors[:20],
            )
        )
    return reports
