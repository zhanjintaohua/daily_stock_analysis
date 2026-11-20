# -*- coding: utf-8 -*-
"""MyhhubFetcher — myhhub/stock 本地筹码模型作 DSA 筹码分布最后兜底.

Upstream: myhhub/stock (Apache-2.0), commit
b6e0ca2268cfbadd02f5ed052159c187b6670231. Vendored modules live in
``data_provider/myhhub_vendor/`` (``cyq.py``,
``pattern_recognitions.py``, ``patterns.py``, ``adapter.py``) — the same
files as ``trading-agents/ashare/myhhub_vendor/``; only the two modules
(chip + pattern) were taken, nothing else from the upstream repo.

Position in the chain: priority 98, last. The manager
(``base.DataFetcherManager.get_chip_distribution``) tries fetchers in
priority order and takes the first *meaningful* result
(``_is_meaningful_chip_distribution``: avg_cost > 0 and a
concentration present). This fetcher is therefore reached only when
AkShare ``stock_cyq_em`` (Eastmoney CYQ, push2), Miaoxiang and
Tushare all fail or return nothing usable — exactly the "两边都标缺失"
gap. It needs no key and no extra endpoint beyond the push2his daily
bars the adapter already fetches (single attempt, no retry).

Model label: the values are a LOCAL recomputation (210-trading-day
triangular distribution, ``CYQCalculator``), not Eastmoney CYQ. The
returned ``ChipDistribution.source`` is ``"myhhub"`` so every consumer
can tell them apart. 2026-11-20 cross-check on 603928: myhhub vs EM
CYQ 2026-11-19 — profit 81.58% vs 81.11%, avg cost 11.59 vs 11.58,
conc90 0.1327 vs 0.1328, conc70 0.0811 vs 0.0811 (see
``trading-agents/myhhub/验证记录-20261120.md``).

Scope guard: ETF / US / HK codes return None (no chip distribution,
same rule as AkshareFetcher). Pattern recognition is NOT wired here —
DSA already has a heuristic ``analyze_pattern`` tool; the TA-Lib
61-pattern layer is added additively inside that tool's handler (see
``src/agent/tools/analysis_tools.py``).
"""

import logging
import os
from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseFetcher
from .myhhub_vendor import adapter as _myhhub
from .realtime_types import ChipDistribution

logger = logging.getLogger(__name__)


class MyhhubFetcher(BaseFetcher):
    """Local-model chip fallback. Priority 98 (last in the chain)."""

    name = "MyhhubFetcher"
    priority = int(os.getenv("MYHHUB_PRIORITY", "98"))

    def _fetch_raw_data(self, stock_code: str, start_date: str,
                        end_date: str) -> pd.DataFrame:
        """Raw daily bars from Eastmoney push2his (single attempt).

        Only used to satisfy the ``BaseFetcher`` ABC and as the shared
        bars source for the local chip computation. ETF codes go through
        the fund history endpoint (``stock_zh_a_hist`` returns empty
        for ETFs).
        """
        import akshare as ak

        code = (stock_code or "").strip()
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        if _myhhub.is_etf_or_index(code):
            return ak.fund_etf_hist_em(symbol=code, period="daily",
                                       start_date=start, end_date=end,
                                       adjust="qfq")
        return ak.stock_zh_a_hist(symbol=code, period="daily",
                                  start_date=start, end_date=end,
                                  adjust="qfq")

    def _normalize_data(self, df: pd.DataFrame,
                        stock_code: str) -> pd.DataFrame:
        """Standard columns: date/open/high/low/close/volume/amount (+turnover)."""
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume",
                         "amount", "turnover"])
        rename = {"日期": "date", "开盘": "open", "收盘": "close",
                  "最高": "high", "最低": "low", "成交量": "volume",
                  "成交额": "amount", "涨跌幅": "pct_chg",
                  "换手率": "turnover"}
        df = df.rename(columns=rename)
        keep = [c for c in ["date", "open", "high", "low", "close",
                            "volume", "amount", "pct_chg", "turnover"]
                if c in df.columns]
        df = df[keep].copy()
        df["date"] = df["date"].astype(str).str[:10]
        return df.sort_values("date").reset_index(drop=True)

    def get_chip_distribution(
            self, stock_code: str) -> Optional[ChipDistribution]:
        """Chip distribution computed locally from daily bars.

        Returns a ``ChipDistribution`` (source="myhhub") or None when
        the code has no chip concept (ETF/US/HK) or the computation
        produced nothing usable. Never raises.
        """
        from .akshare_fetcher import (_is_etf_code, _is_hk_code,
                                      _is_us_code)

        code = (stock_code or "").strip()
        if _is_us_code(code) or _is_hk_code(code) or _is_etf_code(code):
            logger.debug(f"[筹码分布-myhhub] {code} 无筹码分布概念，跳过")
            return None
        try:
            today = date.today().strftime("%Y-%m-%d")
            res = _myhhub.chip_and_patterns(code, today)
            chip = res["chip"]
            if not chip["ok"]:
                logger.warning(
                    f"[筹码分布-myhhub] {code} 本地计算无可用结果: "
                    f"{chip.get('error')}")
                return None
            out = ChipDistribution(
                code=code,
                date=str(chip["row_date"] or ""),
                source="myhhub",
                profit_ratio=float(chip["profit_ratio"] or 0.0),
                avg_cost=float(chip["avg_cost"] or 0.0),
                cost_90_low=float(chip["cost_90_low"] or 0.0),
                cost_90_high=float(chip["cost_90_high"] or 0.0),
                concentration_90=float(chip["conc90"] or 0.0),
                cost_70_low=float(chip["cost_70_low"] or 0.0),
                cost_70_high=float(chip["cost_70_high"] or 0.0),
                concentration_70=float(chip["conc70"] or 0.0),
            )
            logger.info(
                f"[筹码分布-myhhub] {code} 日期={out.date}: "
                f"获利比例={out.profit_ratio:.1%}, 平均成本={out.avg_cost}, "
                f"90%集中度={out.concentration_90:.2%}, "
                f"70%集中度={out.concentration_70:.2%}")
            return out
        except Exception as exc:  # noqa: BLE001 - fallback must never break the chain
            logger.warning(f"[筹码分布-myhhub] {code} 计算失败: {exc}")
            return None
