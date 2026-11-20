# -*- coding: utf-8 -*-
"""Regression tests for MyhhubFetcher (2026-11-20).

No network: the adapter's ``chip_and_patterns`` is monkeypatched.
Live numbers (603928 / 518880) are recorded in
``trading-agents/myhhub/验证记录-20261120.md``, not asserted here.
"""

from data_provider.myhhub_fetcher import MyhhubFetcher
from data_provider.myhhub_vendor import adapter as _adapter
from data_provider.base import _is_meaningful_chip_distribution


def _ok_chip(**over):
    base = {"ok": True, "row_date": "2026-11-19",
            "profit_ratio": 0.8158, "avg_cost": 11.59,
            "cost_90_low": 10.44, "cost_90_high": 13.63,
            "conc90": 0.1327, "conc70": 0.0811,
            "cost_70_low": 10.77, "cost_70_high": 12.68,
            "present": [], "absent": [], "anomalies": [],
            "error": None, "source": "myhhub"}
    base.update(over)
    return {"bars_rows": 507, "bars_latency_ms": 10,
            "chip": base,
            "patterns": {"ok": True, "row_date": "2026-11-19",
                         "triggered": [], "checked": 61, "error": None,
                         "source": "myhhub"},
            "fetch_error": None}


def test_get_chip_distribution_returns_myhhub_source(monkeypatch):
    monkeypatch.setattr(_adapter, "chip_and_patterns",
                        lambda code, on_date: _ok_chip())
    chip = MyhhubFetcher().get_chip_distribution("603928")
    assert chip is not None
    assert chip.source == "myhhub"
    assert chip.code == "603928"
    assert chip.date == "2026-11-19"
    assert chip.profit_ratio == 0.8158
    assert chip.avg_cost == 11.59
    assert chip.concentration_90 == 0.1327
    assert chip.concentration_70 == 0.0811
    assert _is_meaningful_chip_distribution(chip) is True


def test_get_chip_distribution_etf_returns_none(monkeypatch):
    monkeypatch.setattr(_adapter, "chip_and_patterns",
                        lambda code, on_date: _ok_chip())
    assert MyhhubFetcher().get_chip_distribution("518880") is None


def test_get_chip_distribution_us_returns_none(monkeypatch):
    monkeypatch.setattr(_adapter, "chip_and_patterns",
                        lambda code, on_date: _ok_chip())
    assert MyhhubFetcher().get_chip_distribution("AAPL") is None


def test_get_chip_distribution_failed_calc_returns_none(monkeypatch):
    bad = _ok_chip(ok=False, error="模拟计算失败",
                   profit_ratio=None, avg_cost=None,
                   conc90=None, conc70=None)
    monkeypatch.setattr(_adapter, "chip_and_patterns",
                        lambda code, on_date: bad)
    assert MyhhubFetcher().get_chip_distribution("603928") is None


def test_fetcher_priority_is_last():
    assert MyhhubFetcher.priority == 98
    assert MyhhubFetcher.name == "MyhhubFetcher"
