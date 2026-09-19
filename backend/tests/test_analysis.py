"""Unit tests for math resolution (Block A Step 1c) and table validation (1d)."""

from __future__ import annotations

from black_box.services.math_resolver import SYMBOL_TABLE, resolve_math_in_text
from black_box.services.table_validator import (
    ColumnSpec,
    TableSpec,
    validate_tables,
)


class _FakeCell:
    def __init__(self, text: str) -> None:
        self.text = text


# --- 1c: math resolution -----------------------------------------------------


def test_resolve_unicode_symbol():
    res = resolve_math_in_text("annualized volatility target is σ = 15%")
    assert res is not None
    assert res.symbol_table["σ"] == SYMBOL_TABLE["σ"]
    assert "sigma" in res.resolved


def test_resolve_inline_latex():
    res = resolve_math_in_text("position = (risk * equity) / $\\sigma$")
    assert res is not None
    assert "sigma" in res.resolved
    assert r"\sigma" in res.symbol_table


def test_no_math_returns_none():
    assert resolve_math_in_text("plain text without symbols") is None


def test_resolution_context_carried():
    res = resolve_math_in_text("σ = 15%", context="Signal")
    assert res is not None
    assert res.context == "Signal"


# --- 1d: table validation -----------------------------------------------------


def _fake_table(header: list[str], rows: list[list[_FakeCell]]):
    class _Table:
        def __init__(self) -> None:
            self.data = type(
                "Data", (), {"grid": [[_FakeCell(h) for h in header]] + rows}
            )()

    return _Table()


def _validate(specs, table):
    doc = type("Doc", (), {"tables": [table]})()
    return validate_tables(doc, specs)


def test_table_fit():
    header = ["Param", "Value", "Bounds"]
    rows = [
        [_FakeCell("lookback"), _FakeCell("20"), _FakeCell("[10, 60]")],
        [_FakeCell("entry_std"), _FakeCell("2.0"), _FakeCell("[1.0, 3.0]")],
    ]
    reports = _validate(None, _fake_table(header, rows))
    assert len(reports) == 1
    assert reports[0].status == "fit"
    assert reports[0].rows == 2
    assert reports[0].errors == []


def test_table_parse_error_on_bad_value():
    header = ["Param", "Value", "Bounds"]
    rows = [[_FakeCell("lookback"), _FakeCell("twenty"), _FakeCell("[10, 60]")]]
    reports = _validate(None, _fake_table(header, rows))
    assert reports[0].status == "parse-error"
    assert any("not a number" in e for e in reports[0].errors)


def test_table_parse_error_on_inverted_range():
    header = ["Param", "Value", "Bounds"]
    rows = [[_FakeCell("lookback"), _FakeCell("20"), _FakeCell("[60, 10]")]]
    reports = _validate(None, _fake_table(header, rows))
    assert reports[0].status == "parse-error"
    assert any("lower bound > upper bound" in e for e in reports[0].errors)


def test_table_orphan_on_unknown_headers():
    header = ["Foo", "Bar"]
    rows = [[_FakeCell("a"), _FakeCell("b")]]
    reports = _validate(None, _fake_table(header, rows))
    assert reports[0].status == "orphan"


def test_table_unreadable_grid():
    class _Broken:
        def __init__(self) -> None:
            self.data = type("Data", (), {"grid": []})()

    doc = type("Doc", (), {"tables": [_Broken()]})()
    reports = validate_tables(doc)
    assert reports[0].status == "parse-error"


def test_custom_spec_respected():
    header = ["id", "weight"]
    spec = TableSpec(
        name="weights",
        match_headers=["id", "weight"],
        columns=[ColumnSpec(name="weight", kind="number")],
    )
    rows = [[_FakeCell("a"), _FakeCell("0.5")]]
    reports = _validate([spec], _fake_table(header, rows))
    assert reports[0].status == "fit"
