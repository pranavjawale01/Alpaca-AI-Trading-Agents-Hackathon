"""
tests/test_opportunity_scorer.py — Unit tests for OpportunityScorer.

Tests criteria stacking, cap logic, greedy disable toggle, strategy-specific
EMA logic, VIX sweet spots, IVR thresholds, and edge cases.
"""

from __future__ import annotations

from core.opportunity_scorer import OpportunityScorer


class TestOpportunityScorer:

    def test_all_criteria_fired_reaches_cap(self):
        """When all 6 criteria match, multiplier reaches max cap (2.0)."""
        scorer = OpportunityScorer()
        context = {
            "symbol": "NVDA",
            "strategy": "momo",
            "ivr": 75.0,                   # +0.20
            "ema_signal": "bullish",        # +0.20
            "vix": 18.0,                    # +0.15
            "volume_surge_ratio": 2.5,      # +0.15
        }
        # not in open positions: +0.15
        # session_pnl > 0: +0.15
        # total = 1.0 + 0.20 + 0.20 + 0.15 + 0.15 + 0.15 + 0.15 = 2.00
        mult = scorer.score(
            market_context=context,
            open_positions=["SPY", "QQQ"],
            session_pnl=500.0,
        )
        assert mult == 2.0

    def test_no_criteria_fired_returns_base(self):
        """When no criteria match, multiplier remains base 1.0."""
        scorer = OpportunityScorer()
        context = {
            "symbol": "SPY",
            "strategy": "momo",
            "ivr": 30.0,                    # <= 50 -> 0
            "ema_signal": "bearish",        # bearish for momo -> 0
            "vix": 32.0,                    # > 22 -> 0
            "volume_surge_ratio": 1.2,      # <= 1.8 -> 0
        }
        # in open positions: SPY in ["SPY"] -> 0
        # session_pnl <= 0: -50.0 -> 0
        mult = scorer.score(
            market_context=context,
            open_positions=["SPY"],
            session_pnl=-50.0,
        )
        assert mult == 1.0

    def test_greedy_disabled_returns_1_0(self):
        """When greedy mode is disabled, score always returns 1.0 regardless of context."""
        scorer = OpportunityScorer(greedy_enabled=False)
        context = {
            "symbol": "NVDA",
            "strategy": "momo",
            "ivr": 90.0,
            "ema_signal": "bullish",
            "vix": 18.0,
            "volume_surge_ratio": 3.0,
        }
        mult = scorer.score(
            market_context=context,
            open_positions=[],
            session_pnl=1000.0,
        )
        assert mult == 1.0

    def test_custom_cap(self):
        """Custom max_greedy_multiplier is respected."""
        scorer = OpportunityScorer(max_greedy_multiplier=1.50)
        context = {
            "symbol": "NVDA",
            "strategy": "momo",
            "ivr": 80.0,
            "ema_signal": "bullish",
            "vix": 19.0,
            "volume_surge_ratio": 2.2,
        }
        mult = scorer.score(
            market_context=context,
            open_positions=[],
            session_pnl=200.0,
        )
        assert mult == 1.50

    def test_iv_crush_strategy_ema_waiver(self):
        """iv_crush strategy receives EMA alignment boost regardless of signal."""
        scorer = OpportunityScorer()
        context = {
            "symbol": "AAPL",
            "strategy": "iv_crush",
            "ivr": 40.0,                    # 0
            "ema_signal": "bearish",        # iv_crush gets +0.20 anyway
            "vix": 30.0,                    # 0
            "volume_surge_ratio": 1.0,      # 0
        }
        # symbol AAPL in open_positions -> 0
        # session_pnl = 0 -> 0
        mult = scorer.score(
            market_context=context,
            open_positions=["AAPL"],
            session_pnl=0.0,
        )
        # base (1.0) + EMA (+0.20) = 1.20
        assert mult == 1.20

    def test_nested_context_structure(self):
        """Extracts fields correctly from nested dictionary formats like SignalEnhancer."""
        scorer = OpportunityScorer()
        context = {
            "symbol": "TSLA",
            "strategy": "theta",
            "volatility": {
                "ivr": 60.0,                # +0.20
                "vix": 17.5,                # +0.15
            },
            "technical_signals": {
                "ema_signal": "bullish",    # +0.20
                "volume_surge_ratio": 2.0,  # +0.15
            },
        }
        # symbol TSLA not in ["SPY"] -> +0.15
        # session_pnl = 0.0 -> +0.0
        # total = 1.0 + 0.20 + 0.15 + 0.20 + 0.15 + 0.15 = 1.85
        mult = scorer.score(
            market_context=context,
            open_positions=["SPY"],
            session_pnl=0.0,
        )
        assert mult == 1.85

    def test_vix_boundary_conditions(self):
        """VIX exactly 15.0 and 22.0 should be included in sweet spot."""
        scorer = OpportunityScorer()

        # VIX 15.0
        c15 = {"symbol": "A", "vix": 15.0}
        m15 = scorer.score(c15, open_positions=["A"], session_pnl=0.0)
        assert m15 >= 1.15  # VIX +0.15

        # VIX 22.0
        c22 = {"symbol": "A", "vix": 22.0}
        m22 = scorer.score(c22, open_positions=["A"], session_pnl=0.0)
        assert m22 >= 1.15  # VIX +0.15

        # VIX 14.99 (outside)
        c14 = {"symbol": "A", "vix": 14.99}
        m14 = scorer.score(c14, open_positions=["A"], session_pnl=0.0)
        assert m14 == 1.00

        # VIX 22.01 (outside)
        c22_out = {"symbol": "A", "vix": 22.01}
        m22_out = scorer.score(c22_out, open_positions=["A"], session_pnl=0.0)
        assert m22_out == 1.00
