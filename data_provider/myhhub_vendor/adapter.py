"""Adapter: myhhub/stock chip + pattern modules as a supplement source.

Upstream: myhhub/stock (Apache-2.0), commit
b6e0ca2268cfbadd02f5ed052159c187b6670231.
Only two upstream modules are vendored here (``cyq.py``,
``pattern_recognitions.py``) plus the 61-pattern table (``patterns.py``,
extracted verbatim from upstream ``tablestructure.py``). Nothing else
from the upstream repo is imported — no DB, no web server, no trading
engine, no strategy code.

Feeding (free, no key):
  - Daily bars come from AkShare ``stock_zh_a_hist`` — the Eastmoney
    push2his history endpoint, NOT the push2 quote endpoint that
    rate-limits. One attempt, never retried inline.
  - Bars are point-in-time by construction: only rows dated on/before
    ``on_date`` are kept, and the chip window is the last 210 trading
    days ending on ``on_date``.
  - Prices are qfq (前复权), matching upstream ``stockfetch`` behaviour.

Models (labelled as such everywhere they surface):
  - Chip: upstream ``CYQCalculator`` — a local triangular-distribution
    recomputation over 210 trading days. It is NOT Eastmoney CYQ
    (120-day window, different model), so its numbers are reported
    under the source name ``myhhub 本地筹码模型`` and must never be
    presented as 东财 CYQ values.
  - Patterns: upstream ``get_pattern_recognitions`` over TA-Lib
    CDL* functions (61 patterns). Value +100 = bullish trigger,
    -100 = bearish trigger, 0 = no trigger, evaluated on the
    ``on_date`` close using the last 12 bars.

Failure contract: every public function returns a dict and never
raises. ``ok=False`` carries ``error`` (cause, in Chinese) and the
caller marks the field missing with that cause. No retries, no
fill-forward, no invented values.

ETF / index guard: chip distribution is not applicable to ETFs or
indices (same rule as DSA's AkshareFetcher) — ``compute_chip`` is
simply not called for them; pattern recognition still runs (OHLC only).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import cyq as _cyq_mod
from . import pattern_recognitions as _pr_mod
from .patterns import PATTERNS

# 210 trading days for the chip window; fetch ~2 calendar years of bars
# so the window is available even after long holidays / suspensions.
_CHIP_MIN_BARS = 210
_BARS_LOOKBACK_DAYS = 760
_ETF_PREFIXES = ("5", "15", "16")

SOURCE_CHIP = "myhhub 本地筹码模型（210 交易日三角分布，源自 myhhub/stock CYQCalculator）"
SOURCE_PATTERN = "myhhub/TA-Lib 61 种 K 线形态（源自 myhhub/stock pattern_recognitions）"


def is_etf_or_index(code: str) -> bool:
    """True for ETF codes (5xxxxx / 15xxxx / 16xxxx)."""
    c = (code or "").strip()
    return c.startswith(_ETF_PREFIXES)


def fetch_bars(code: str, on_date: str) -> dict:
    """Fetch daily bars ending on ``on_date`` (single attempt, no retry).

    Returns {"ok": bool, "bars": DataFrame|None, "rows": int,
    "error": str|None, "latency_ms": int|None}.
    Columns on success: date(str YYYY-MM-DD) / open / high / low /
    close / volume / amount / turnover (%, as reported by AkShare).
    """
    import time

    t0 = time.time()
    out = {"ok": False, "bars": None, "rows": 0, "error": None,
           "latency_ms": None}
    try:
        import akshare as ak
        import pandas as pd

        d = datetime.strptime(on_date, "%Y-%m-%d").date()
        start = (d - timedelta(days=_BARS_LOOKBACK_DAYS)).strftime("%Y%m%d")
        end = d.strftime("%Y%m%d")
        if is_etf_or_index(code):
            # ETF 日线不在 stock_zh_a_hist 里（返回空），走基金历史接口
            # fund_etf_hist_em（同为 push2his，同列名）；单次调用。
            df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                     start_date=start, end_date=end,
                                     adjust="qfq")
        else:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start, end_date=end, adjust="qfq")
        out["latency_ms"] = int((time.time() - t0) * 1000)
        if df is None or len(df) == 0:
            out["error"] = "东财 push2his 日线返回空"
            return out
        rename = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                  "最低": "low", "成交量": "volume", "成交额": "amount",
                  "换手率": "turnover"}
        df = df.rename(columns=rename)
        need = ["date", "open", "high", "low", "close", "volume", "amount",
                "turnover"]
        missing_cols = [c for c in need if c not in df.columns]
        if missing_cols:
            out["error"] = f"日线缺列：{'、'.join(missing_cols)}"
            return out
        df = df[need].copy()
        df["date"] = df["date"].astype(str).str[:10]
        for c in ["open", "high", "low", "close", "volume", "amount",
                  "turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["date"] <= on_date].sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) == 0:
            out["error"] = f"日线无行日期≤{on_date}"
            return out
        out.update({"ok": True, "bars": df, "rows": len(df)})
        return out
    except Exception as exc:  # noqa: BLE001 - fetch failure => missing entry
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["error"] = f"日线拉取失败（{type(exc).__name__}: {exc}）"
        return out


def compute_chip(bars, on_date: str) -> dict:
    """Local chip distribution as of ``on_date`` from ``bars``.

    ``bars`` is the frame returned by :func:`fetch_bars` (rows <= on_date).
    Returns {"ok", "row_date", "profit_ratio" (0-1), "avg_cost",
    "cost_90_low/high", "conc90"/"conc70" (0-1 fractions),
    "present"/"absent", "anomalies", "error", "source"}.
    """
    result = {"ok": False, "row_date": None, "profit_ratio": None,
              "avg_cost": None, "cost_90_low": None, "cost_90_high": None,
              "conc90": None, "conc70": None, "present": [], "absent": [],
              "anomalies": [], "error": None, "source": SOURCE_CHIP}
    try:
        if bars is None or len(bars) < _CHIP_MIN_BARS:
            n = 0 if bars is None else len(bars)
            result["error"] = f"筹码窗口不足：仅 {n} 根日线，需 ≥{_CHIP_MIN_BARS} 根"
            return result
        calc = _cyq_mod.CYQCalculator(bars)
        # index 119 with default range=120 gives end==0, i.e. the tail
        # 210 trading days ending on the last bar (upstream chart usage).
        r = calc.calc(119)
        result["row_date"] = str(r.d)[:10]
        if result["row_date"] > on_date:
            result["error"] = f"筹码行日期 {result['row_date']} 晚于分析日 {on_date}"
            return result
        profit = float(r.benefit_part)
        avg_cost = float(r.avg_cost)
        p90 = r.percent_chips["90"]
        p70 = r.percent_chips["70"]
        c90_low, c90_high = float(p90["priceRange"][0]), float(p90["priceRange"][1])
        c70_low, c70_high = float(p70["priceRange"][0]), float(p70["priceRange"][1])
        conc90 = float(p90["concentration"])
        conc70 = float(p70["concentration"])

        def _check(name, value, lo, hi):
            if value is None or value != value:
                result["anomalies"].append(f"{name} 非数值（不作真值）")
                return None
            if not (lo <= value <= hi):
                result["anomalies"].append(
                    f"{name} {value}（异常：超出 {lo}–{hi} 合理区间，不作真值）")
                return None
            return value

        result["profit_ratio"] = _check("获利比例", profit, 0.0, 1.0)
        result["avg_cost"] = _check("平均成本", avg_cost, 0.0, float("inf"))
        result["cost_90_low"] = _check("90%成本-低", c90_low, 0.0, float("inf"))
        result["cost_90_high"] = _check("90%成本-高", c90_high, 0.0, float("inf"))
        result["conc90"] = _check("90%筹码集中度", conc90, 0.0, 1.0)
        result["conc70"] = _check("70%筹码集中度", conc70, 0.0, 1.0)
        result["cost_70_low"] = _check("70%成本-低", c70_low, 0.0, float("inf"))
        result["cost_70_high"] = _check("70%成本-高", c70_high, 0.0, float("inf"))
        for key, name in (("profit_ratio", "获利比例"), ("avg_cost", "平均成本"),
                          ("cost_90_low", "90%成本-低"),
                          ("cost_90_high", "90%成本-高"),
                          ("conc90", "90%筹码集中度"),
                          ("cost_70_low", "70%成本-低"),
                          ("cost_70_high", "70%成本-高"),
                          ("conc70", "70%筹码集中度")):
            (result["present"] if result[key] is not None
             else result["absent"]).append(name)
        result["ok"] = result["avg_cost"] is not None and (
            result["conc90"] is not None or result["conc70"] is not None)
        if not result["ok"]:
            result["error"] = "本地筹码计算未产出可用核心指标（平均成本/集中度缺失或异常）"
        return result
    except Exception as exc:  # noqa: BLE001 - calc failure => missing entry
        result["error"] = f"本地筹码计算失败（{type(exc).__name__}: {exc}）"
        return result


def detect_patterns(bars, on_date: str) -> dict:
    """61-pattern TA-Lib scan on the ``on_date`` close (last 12 bars).

    Returns {"ok", "row_date", "triggered": [{"key","cn","value"}],
    "checked": 61, "error", "source"}. ``triggered == []`` with ok=True
    means "no pattern fired" — a real answer, not a failure. ok=False
    (e.g. TA-Lib not installed) means the field is missing with cause.
    """
    result = {"ok": False, "row_date": None, "triggered": [],
              "checked": len(PATTERNS), "error": None, "source": SOURCE_PATTERN}
    try:
        import talib
    except ImportError:
        result["error"] = "形态识别缺失：TA-Lib 未安装（需 pip install TA_Lib==0.6.8）"
        return result
    try:
        if bars is None or len(bars) < 2:
            result["error"] = "形态识别缺失：日线不足 2 根"
            return result
        stock_column = {k: {"func": getattr(talib, fname)}
                        for k, (_, fname) in PATTERNS.items()}
        out = _pr_mod.get_pattern_recognitions(
            bars, stock_column, end_date=on_date, threshold=1,
            calc_threshold=12)
        if out is None or len(out) == 0:
            result["error"] = f"形态识别无返回行（分析日 {on_date} 前无日线）"
            return result
        row = out.iloc[0]
        result["row_date"] = str(row["date"])[:10]
        for key, (cn, _fname) in PATTERNS.items():
            try:
                v = int(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if v != 0:
                result["triggered"].append(
                    {"key": key, "cn": cn, "value": v})
        result["ok"] = True
        return result
    except Exception as exc:  # noqa: BLE001 - scan failure => missing entry
        result["error"] = f"形态识别失败（{type(exc).__name__}: {exc}）"
        return result


def chip_and_patterns(code: str, on_date: str) -> dict:
    """One fetch, both modules. Returns {"bars_rows", "bars_latency_ms",
    "chip": {...}, "patterns": {...}, "fetch_error": str|None}."""
    code = (code or "").strip()
    fetched = fetch_bars(code, on_date)
    chip = {"ok": False, "row_date": None, "profit_ratio": None,
            "avg_cost": None, "cost_90_low": None, "cost_90_high": None,
            "conc90": None, "conc70": None, "cost_70_low": None,
            "cost_70_high": None, "present": [], "absent": [],
            "anomalies": [], "error": None, "source": SOURCE_CHIP}
    patterns = {"ok": False, "row_date": None, "triggered": [],
                "checked": len(PATTERNS), "error": None,
                "source": SOURCE_PATTERN}
    if not fetched["ok"]:
        chip["error"] = f"筹码：{fetched['error']}"
        patterns["error"] = f"形态：{fetched['error']}"
        return {"bars_rows": 0, "bars_latency_ms": fetched["latency_ms"],
                "chip": chip, "patterns": patterns,
                "fetch_error": fetched["error"]}
    bars = fetched["bars"]
    if is_etf_or_index(code):
        chip["error"] = "ETF/指数无筹码分布（不适用，非失败）"
        chip["absent"] = ["获利比例", "平均成本", "90%成本-低", "90%成本-高",
                          "90%筹码集中度", "70%成本-低", "70%成本-高",
                          "70%筹码集中度"]
    else:
        chip = compute_chip(bars, on_date)
    patterns = detect_patterns(bars, on_date)
    return {"bars_rows": fetched["rows"],
            "bars_latency_ms": fetched["latency_ms"],
            "chip": chip, "patterns": patterns, "fetch_error": None}
