# -*- coding: utf-8 -*-
"""Regression tests for AkShare concept rankings Eastmoney -> Sina fallback.

Eastmoney push2 (ak.stock_board_concept_name_em) returns 502 under rate
limiting; concept rankings previously had no second source at all, while
supplementary fetchers such as Miaoxiang never implement concept rankings.
The Sina spot table (ak.stock_sector_spot, indicator='概念') returns the
same '板块'/'涨跌幅' columns as the industry spot table the sector fallback
already uses.
"""

import sys
from types import SimpleNamespace

import pandas as pd

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from data_provider.akshare_fetcher import AkshareFetcher


def _em_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "板块名称": ["机器人概念", "转基因", "消费电子"],
            "涨跌幅": [4.2, -2.0, 1.5],
        }
    )


def _sina_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "板块": ["人形机器人", "固态电池", "医药"],
            "涨跌幅": [3.1, -1.2, 0.8],
        }
    )


def _fetcher_with(monkeypatch, *, eastmoney, sina) -> AkshareFetcher:
    monkeypatch.setitem(
        sys.modules,
        "akshare",
        SimpleNamespace(
            stock_board_concept_name_em=eastmoney,
            stock_sector_spot=sina,
        ),
    )
    return AkshareFetcher(sleep_min=0, sleep_max=0)


def test_concept_rankings_prefers_eastmoney(monkeypatch) -> None:
    sina_calls = []

    def fake_sina(*, indicator=None):
        sina_calls.append(indicator)
        return _sina_frame()

    fetcher = _fetcher_with(
        monkeypatch,
        eastmoney=lambda: _em_frame(),
        sina=fake_sina,
    )

    top, bottom = fetcher.get_concept_rankings(2)

    assert [row["name"] for row in top] == ["机器人概念", "消费电子"]
    assert [row["name"] for row in bottom] == ["转基因", "消费电子"]
    assert sina_calls == []


def test_concept_rankings_falls_back_to_sina_when_eastmoney_502(monkeypatch) -> None:
    sina_calls = []

    def fake_sina(*, indicator=None):
        sina_calls.append(indicator)
        return _sina_frame()

    def fake_eastmoney():
        raise RuntimeError("502 Bad Gateway")

    fetcher = _fetcher_with(
        monkeypatch,
        eastmoney=fake_eastmoney,
        sina=fake_sina,
    )

    top, bottom = fetcher.get_concept_rankings(2)

    assert sina_calls == ["概念"]
    assert [row["name"] for row in top] == ["人形机器人", "医药"]
    assert [row["name"] for row in bottom] == ["固态电池", "医药"]


def test_concept_rankings_falls_back_to_sina_when_eastmoney_empty(monkeypatch) -> None:
    sina_calls = []

    def fake_sina(*, indicator=None):
        sina_calls.append(indicator)
        return _sina_frame()

    fetcher = _fetcher_with(
        monkeypatch,
        eastmoney=lambda: pd.DataFrame(),
        sina=fake_sina,
    )

    top, _bottom = fetcher.get_concept_rankings(1)

    assert sina_calls == ["概念"]
    assert [row["name"] for row in top] == ["人形机器人"]


def test_concept_rankings_returns_none_when_both_sources_fail(monkeypatch) -> None:
    def fake_sina(*, indicator=None):
        raise RuntimeError("sina down")

    fetcher = _fetcher_with(
        monkeypatch,
        eastmoney=lambda: (_ for _ in ()).throw(RuntimeError("502 Bad Gateway")),
        sina=fake_sina,
    )

    assert fetcher.get_concept_rankings(5) is None
