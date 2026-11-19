# -*- coding: utf-8 -*-
"""
LLM JSON extraction robustness tests (fix (c)).

Regression coverage for the run #2 gate failures (2026-11-19): DeepSeek-V4-Pro
returned a complete dashboard JSON wrapped in an intro line, a ```json fenced
block, and trailing markdown — the old extractor rejected that shape as
``ambiguous_json`` even though the JSON object itself was complete.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import GeminiAnalyzer

DASHBOARD_JSON = """{
    "stock_name": "宇树科技（688836）",
    "sentiment_score": 39,
    "trend_prediction": "看空",
    "operation_advice": "减仓",
    "decision_type": "sell",
    "dashboard": {
        "core_conclusion": {"one_sentence": "趋势破位走弱，持仓者减仓避险"},
        "intelligence": {"risk_alerts": ["风险点1：跌破全部均线"]},
        "phase_decision": {
            "phase_context": {"phase": "premarket"},
            "action_window": "盘前计划",
            "immediate_action": "止损止盈预警",
            "watch_conditions": ["600元关口是否失守"],
            "next_check_time": "2026-11-19 09:30",
            "confidence_reason": "盘前阶段，数据降级",
            "data_limitations": ["行情数据降级"]
        }
    },
    "analysis_summary": "跌破均线，弱势空头。"
}"""


def _analyzer() -> GeminiAnalyzer:
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        return GeminiAnalyzer()


class TestExtractAnalysisJsonObject(unittest.TestCase):
    """_extract_analysis_json_object tolerates real LLM output shapes."""

    def test_fenced_json_with_intro_and_trailing_markdown(self) -> None:
        """Run #2 shape: prose + ```json fence + trailing markdown extracts."""
        response = (
            "**分析标的：宇树科技（688836）** —— 科创板上市公司，以下为盘前决策仪表盘。\n\n"
            "```json\n" + DASHBOARD_JSON + "\n```\n\n"
            "**盘前执行要点提示**：\n1. **多头排列检查**：❌ 不满足；\n2. **今日关键动作**：开盘紧盯600元整数关口。\n"
        )
        analyzer = _analyzer()
        _json_str, data = analyzer._extract_analysis_json_object(response)
        self.assertEqual(data["sentiment_score"], 39)
        self.assertEqual(
            data["dashboard"]["core_conclusion"]["one_sentence"],
            "趋势破位走弱，持仓者减仓避险",
        )
        self.assertEqual(
            data["dashboard"]["intelligence"]["risk_alerts"],
            ["风险点1：跌破全部均线"],
        )
        self.assertEqual(
            data["dashboard"]["phase_decision"]["action_window"], "盘前计划"
        )

    def test_bare_json_with_trailing_markdown(self) -> None:
        """Unfenced JSON followed by markdown extracts via raw-decode prefix."""
        analyzer = _analyzer()
        _json_str, data = analyzer._extract_analysis_json_object(
            DASHBOARD_JSON + "\n\n以上为分析结论，仅供参考。\n"
        )
        self.assertEqual(data["sentiment_score"], 39)

    def test_truncated_json_repaired(self) -> None:
        """Missing closing braces are repaired by the json-repair fallback."""
        analyzer = _analyzer()
        truncated = DASHBOARD_JSON.rstrip()[:-3]  # drop closing "}\n}"
        _json_str, data = analyzer._extract_analysis_json_object(truncated)
        self.assertEqual(data["sentiment_score"], 39)

    def test_two_different_valid_json_objects_is_ambiguous(self) -> None:
        """Two genuinely different dashboards stay ambiguous."""
        other = DASHBOARD_JSON.replace('"sentiment_score": 39', '"sentiment_score": 70')
        analyzer = _analyzer()
        with self.assertRaises(ValueError) as ctx:
            analyzer._extract_analysis_json_object(
                "```json\n" + DASHBOARD_JSON + "\n```\n```json\n" + other + "\n```\n"
            )
        self.assertEqual(str(ctx.exception), "ambiguous_json")

    def test_no_json_is_invalid(self) -> None:
        analyzer = _analyzer()
        with self.assertRaises(ValueError) as ctx:
            analyzer._extract_analysis_json_object("今天天气不错，没有分析结果。")
        self.assertEqual(str(ctx.exception), "invalid_json")

    def test_empty_is_empty_response(self) -> None:
        analyzer = _analyzer()
        with self.assertRaises(ValueError) as ctx:
            analyzer._extract_analysis_json_object("   \n")
        self.assertEqual(str(ctx.exception), "empty_response")

    def test_validate_accepts_run2_shape(self) -> None:
        """The response validator used by _call_litellm accepts the run #2 shape."""
        response = (
            "分析标的：**英伟达（NVDA）**\n\n```json\n"
            + DASHBOARD_JSON
            + "\n```\n\n**执行要点**：紧盯关键价位。\n"
        )
        analyzer = _analyzer()
        analyzer._validate_json_response(response)  # must not raise


if __name__ == "__main__":
    unittest.main()
