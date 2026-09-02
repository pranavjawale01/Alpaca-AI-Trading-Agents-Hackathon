"""
tests/test_two_sided_trading.py — Unit tests for two-sided trading capabilities:
buying vs selling, long calls vs long puts (short delta), cash-secured puts,
and portfolio delta Greek sign correctness.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.market_data import MarketData
from core.signal_enhancer import SignalEnhancer
from core.llm_council import LLMCouncil, _STRATEGY_PROMPTS
from agents.orchestrator import Orchestrator
from agents.momo_breakout import MomoBreakoutAgent
from agents.theta_collector import ThetaCollectorAgent
from core.risk_manager import RiskManager


class TestEMACrossovers:
    """Test MarketData EMA signal for both bullish and bearish crossovers."""

    def test_bullish_crossover(self):
        client = MagicMock()
        # Fast EMA crosses ABOVE slow EMA on the last bar
        prices = [100.0] * 30 + [99.0, 98.0, 97.0, 96.0, 105.0]
        client.get_bars.return_value = [{"c": p, "v": 1000} for p in prices]

        md = MarketData(client)
        sig = md.get_ema_signal("TEST", fast=5, slow=10)

        assert sig["crossover"] is True
        assert sig["crossover_bullish"] is True
        assert sig["crossover_bearish"] is False
        assert sig["crossover_type"] == "bullish"
        assert sig["signal"] == "bullish"

    def test_bearish_crossover(self):
        client = MagicMock()
        # Fast EMA crosses BELOW slow EMA on the last bar
        prices = [100.0] * 30 + [101.0, 102.0, 103.0, 104.0, 95.0]
        client.get_bars.return_value = [{"c": p, "v": 1000} for p in prices]

        md = MarketData(client)
        sig = md.get_ema_signal("TEST", fast=5, slow=10)

        assert sig["crossover"] is True
        assert sig["crossover_bearish"] is True
        assert sig["crossover_bullish"] is False
        assert sig["crossover_type"] == "bearish"
        assert sig["signal"] == "bearish"


class TestSignalEnhancerTwoSided:
    """Test SignalEnhancer produces correct context for both calls and puts."""

    def test_bullish_context(self):
        ema_sig = {"crossover": True, "crossover_type": "bullish", "signal": "bullish", "ema_fast": 105, "ema_slow": 100}
        vol_surge = {"is_surging": True, "surge_ratio": 2.5}
        ctx = SignalEnhancer.build_momo_context("AAPL", ema_sig, vol_surge, vix=18.0, price=105.0, direction="bullish")

        assert ctx["trade_parameters"]["option_type"] == "call"
        assert ctx["direction"] == "bullish"
        assert "Call Buy" in ctx["strategy"]

    def test_bearish_context(self):
        ema_sig = {"crossover": True, "crossover_type": "bearish", "signal": "bearish", "ema_fast": 95, "ema_slow": 100}
        vol_surge = {"is_surging": True, "surge_ratio": 2.5}
        ctx = SignalEnhancer.build_momo_context("AAPL", ema_sig, vol_surge, vix=28.0, price=95.0, direction="bearish")

        assert ctx["trade_parameters"]["option_type"] == "put"
        assert ctx["direction"] == "bearish"
        assert "Put Buy" in ctx["strategy"]
        assert "Short Delta" in ctx["strategy"]


class TestLLMCouncilPrompts:
    """Verify LLM Council supports both momentum_call and momentum_put."""

    def test_strategy_prompts_exist(self):
        assert "momentum_call" in _STRATEGY_PROMPTS
        assert "momentum_put" in _STRATEGY_PROMPTS
        assert "theta_put" in _STRATEGY_PROMPTS
        assert "iv_crush_straddle" in _STRATEGY_PROMPTS


class TestPortfolioDeltaEstimation:
    """Verify MasterOrchestrator._estimate_portfolio_delta accounts for long/short and call/put."""

    @patch("agents.orchestrator.AlpacaClient")
    def test_delta_signs(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_account.return_value = {"equity": 100000}
        orchestrator = Orchestrator(client=mock_client)

        positions = [
            # Long Stock: +1 delta per share
            {"symbol": "SPY", "qty": 10, "asset_class": "us_equity"},
            # Short Stock: -1 delta per share
            {"symbol": "QQQ", "qty": -5, "asset_class": "us_equity"},
            # Long Call: positive delta (+50 per contract)
            {"symbol": "AAPL260918C00200000", "qty": 1, "asset_class": "us_option"},
            # Short Call: negative delta (-50 per contract)
            {"symbol": "MSFT260918C00450000", "qty": -1, "asset_class": "us_option"},
            # Long Put: negative delta (-50 per contract)
            {"symbol": "NVDA260918P00120000", "qty": 1, "asset_class": "us_option"},
            # Short Put: positive delta (+50 per contract)
            {"symbol": "GOOG260918P00170000", "qty": -1, "asset_class": "us_option"},
        ]

        # Calculation:
        # SPY: +10
        # QQQ: -5
        # AAPL Long Call: +1 * 50 = +50
        # MSFT Short Call: -1 * 50 = -50
        # NVDA Long Put: +1 * (-50) = -50
        # GOOG Short Put: -1 * (-50) = +50
        # Net: 10 - 5 + 50 - 50 - 50 + 50 = 5.0
        delta = orchestrator._estimate_portfolio_delta(positions)
        assert delta == 5.0


class TestMomoTwoSidedExecution:
    """Verify MomoBreakoutAgent executes both calls (long) and puts (short delta)."""

    def test_buy_call_and_buy_put_delta_impact(self):
        client = MagicMock()
        client.get_account.return_value = {"equity": 100000}
        md = MagicMock()
        rm = MagicMock()

        momo = MomoBreakoutAgent(client=client, market_data=md, risk_manager=rm)

        contract_call = {"symbol": "AAPL260918C00200000", "expiration": "2026-09-18"}
        action_call = momo._buy_option(
            symbol="AAPL",
            equity=100000,
            contract=contract_call,
            target_strike=200.0,
            est_premium=3.0,
            option_type="call",
        )

        assert action_call is not None
        assert action_call["action"] == "buy_call"
        assert action_call["direction"] == "bullish"
        # Verify RiskManager approve_order called with positive delta for call
        call_rm_args = rm.approve_order.call_args_list[-1][1]
        assert call_rm_args["delta_impact"] > 0

        contract_put = {"symbol": "AAPL260918P00180000", "expiration": "2026-09-18"}
        action_put = momo._buy_option(
            symbol="AAPL",
            equity=100000,
            contract=contract_put,
            target_strike=180.0,
            est_premium=3.0,
            option_type="put",
        )

        assert action_put is not None
        assert action_put["action"] == "buy_put"
        assert action_put["direction"] == "bearish"
        # Verify RiskManager approve_order called with negative delta for put
        put_rm_args = rm.approve_order.call_args_list[-1][1]
        assert put_rm_args["delta_impact"] < 0


class TestThetaCollectorPositiveDelta:
    """Verify ThetaCollector short put assigns positive delta."""

    def test_short_put_positive_delta(self):
        client = MagicMock()
        client.get_account.return_value = {"equity": 100000}
        md = MagicMock()
        rm = MagicMock()

        theta = ThetaCollectorAgent(client=client, market_data=md, risk_manager=rm)

        # Mock entry scan conditions
        md.get_price.return_value = 500.0
        md.estimate_historical_vol.return_value = 0.25
        rm.current_vix = 18.0
        client.get_option_contracts.return_value = [
            {"symbol": "SPY260918P00450000", "strike": 450.0, "expiration": "2026-09-18"}
        ]
        md.find_otm_strike.return_value = 450.0

        signal = theta._evaluate_entry("SPY", 100000)
        assert signal is not None
        assert signal["action"] == "sell_put"

        # Check that approve_order was called with positive delta
        approve_call = rm.approve_order.call_args_list[-1][1]
        assert approve_call["delta_impact"] > 0


class TestThreeAgentQuantitativeCouncil:
    """Verify 3-Agent Quantitative Strategy Council requiring unanimous approval."""

    def test_unanimous_approval_theta_csp(self):
        council = LLMCouncil()
        ctx = {
            "volatility": {"ivr": 45.0, "vix": 18.0, "historical_vol_30d": 0.22},
            "market_conditions": {"vix": 18.0},
            "trade_parameters": {"target_dte": 35, "target_delta": "~0.20"},
        }
        res = council.vote("SPY", ctx, strategy="theta_put")

        assert res.agreed is True
        assert res.conviction_tier == "strong"
        assert res.size_multiplier == 1.00
        assert res.action == "sell"
        assert len(res.votes) == 3
        assert len(res.dissenting_models) == 0

    def test_veto_when_volatility_agent_dissents(self):
        council = LLMCouncil()
        # IVR = 15.0 (below 30 minimum threshold)
        ctx = {
            "volatility": {"ivr": 15.0, "vix": 18.0, "historical_vol_30d": 0.22},
            "market_conditions": {"vix": 18.0},
            "trade_parameters": {"target_dte": 35, "target_delta": "~0.20"},
        }
        res = council.vote("SPY", ctx, strategy="theta_put")

        # Must be vetoed because VolatilityPricingAgent requires IVR >= 30
        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert res.size_multiplier == 0.00
        assert "VolatilityPricingAgent" in res.dissenting_models

    def test_unanimous_approval_momo_call(self):
        council = LLMCouncil()
        ctx = {
            "direction": "bullish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bullish",
                "volume_surging": True,
                "volume_surge_ratio": 2.2,
            },
            "volatility": {"ivr": 25.0},
            "market_conditions": {"vix": 19.0, "historical_volatility_30d": 0.28},
            "trade_parameters": {"option_type": "call", "target_dte": 35},
        }
        res = council.vote("NVDA", ctx, strategy="momentum_call")

        assert res.agreed is True
        assert res.conviction_tier == "strong"
        assert res.size_multiplier == 1.00
        assert res.action == "buy"

    def test_veto_when_trend_agent_dissents_no_volume(self):
        council = LLMCouncil()
        # volume_surging = False
        ctx = {
            "direction": "bullish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bullish",
                "volume_surging": False,
                "volume_surge_ratio": 1.1,
            },
            "volatility": {"ivr": 25.0},
            "market_conditions": {"vix": 19.0, "historical_volatility_30d": 0.28},
            "trade_parameters": {"option_type": "call", "target_dte": 35},
        }
        res = council.vote("NVDA", ctx, strategy="momentum_call")

        # TrendMomentumAgent dissents due to no volume confirmation
        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert "TrendMomentumAgent" in res.dissenting_models

    def test_unanimous_approval_momo_put_breakdown(self):
        council = LLMCouncil()
        ctx = {
            "direction": "bearish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bearish",
                "volume_surging": True,
                "volume_surge_ratio": 2.4,
            },
            "volatility": {"ivr": 32.0},
            "market_conditions": {"vix": 26.0, "historical_volatility_30d": 0.32},
            "trade_parameters": {"option_type": "put", "target_dte": 35},
        }
        res = council.vote("TSLA", ctx, strategy="momentum_put")

        assert res.agreed is True
        assert res.conviction_tier == "strong"
        assert res.size_multiplier == 1.00
        assert res.action == "buy"

    def test_veto_overbought_rsi_trap(self):
        council = LLMCouncil()
        # Bullish crossover with volume surge, BUT RSI = 78.5 (overbought exhaustion trap)
        ctx = {
            "direction": "bullish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bullish",
                "volume_surging": True,
                "volume_surge_ratio": 2.5,
                "rsi_14": 78.5,
            },
            "volatility": {"ivr": 22.0},
            "market_conditions": {"vix": 16.0, "historical_volatility_30d": 0.22},
            "trade_parameters": {"option_type": "call", "target_dte": 35},
        }
        res = council.vote("NVDA", ctx, strategy="momentum_call")

        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert "TrendMomentumAgent" in res.dissenting_models

    def test_veto_oversold_rsi_trap(self):
        council = LLMCouncil()
        # Bearish breakdown with volume surge, BUT RSI = 22.0 (oversold bounce risk)
        ctx = {
            "direction": "bearish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bearish",
                "volume_surging": True,
                "volume_surge_ratio": 2.5,
                "rsi_14": 22.0,
            },
            "volatility": {"ivr": 25.0},
            "market_conditions": {"vix": 22.0, "historical_volatility_30d": 0.28},
            "trade_parameters": {"option_type": "put", "target_dte": 35},
        }
        res = council.vote("TSLA", ctx, strategy="momentum_put")

        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert "TrendMomentumAgent" in res.dissenting_models

    def test_veto_portfolio_heat_limit(self):
        council = LLMCouncil()
        # Perfect technical & volatility signals, but portfolio options exposure at 96%
        ctx = {
            "direction": "bullish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bullish",
                "volume_surging": True,
                "volume_surge_ratio": 2.2,
                "rsi_14": 55.0,
            },
            "volatility": {"ivr": 25.0},
            "market_conditions": {"vix": 18.0, "historical_volatility_30d": 0.25},
            "trade_parameters": {"option_type": "call", "target_dte": 35},
            "portfolio_state": {
                "portfolio_heat": 0.96,  # > 0.95 cap
                "pnl_budget_used": 0.20,
                "correlated_tech_count": 1,
            },
        }
        res = council.vote("AAPL", ctx, strategy="momentum_call")

        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert "RiskGreeksAgent" in res.dissenting_models

    def test_veto_correlated_tech_limit(self):
        council = LLMCouncil()
        # 4 tech positions already open -> 5th tech position rejected
        ctx = {
            "direction": "bullish",
            "technical_signals": {
                "ema_crossover": True,
                "ema_crossover_type": "bullish",
                "volume_surging": True,
                "volume_surge_ratio": 2.2,
                "rsi_14": 52.0,
            },
            "volatility": {"ivr": 25.0},
            "market_conditions": {"vix": 17.0, "historical_volatility_30d": 0.24},
            "trade_parameters": {"option_type": "call", "target_dte": 35},
            "portfolio_state": {
                "portfolio_heat": 0.50,
                "pnl_budget_used": 0.10,
                "correlated_tech_count": 4,  # Sector limit reached
            },
        }
        res = council.vote("NVDA", ctx, strategy="momentum_call")

        assert res.agreed is False
        assert res.conviction_tier == "veto"
        assert "RiskGreeksAgent" in res.dissenting_models


class TestMarketDataTechnicalIndicators:
    """Test the newly added quantitative indicators on MarketData."""

    def test_rsi_calculation(self):
        from unittest.mock import MagicMock
        from core.market_data import MarketData

        client = MagicMock()
        # 25 upward bars -> RSI should be high (> 70)
        bars = [{"c": 100.0 + i * 2.0} for i in range(25)]
        client.get_bars.return_value = bars
        md = MarketData(client)

        res = md.get_rsi("NVDA", period=14)
        assert res["symbol"] == "NVDA"
        assert res["rsi"] > 70.0
        assert res["zone"] == "overbought"

    def test_macd_calculation(self):
        from unittest.mock import MagicMock
        from core.market_data import MarketData

        client = MagicMock()
        bars = [{"c": 100.0 + i * 1.5} for i in range(45)]
        client.get_bars.return_value = bars
        md = MarketData(client)

        res = md.get_macd("NVDA")
        assert res["symbol"] == "NVDA"
        assert "macd_line" in res
        assert "signal_line" in res
        assert "histogram" in res

    def test_bollinger_bands_calculation(self):
        from unittest.mock import MagicMock
        from core.market_data import MarketData

        client = MagicMock()
        bars = [{"c": 100.0 + (i % 3)} for i in range(35)]
        client.get_bars.return_value = bars
        client.get_latest_quote.return_value = {"mid": 101.0}
        md = MarketData(client)

        res = md.get_bollinger_bands("SPY")
        assert res["symbol"] == "SPY"
        assert res["upper"] > res["lower"]
        assert "bandwidth" in res
        assert "is_squeeze" in res

    def test_sma200_calculation(self):
        from unittest.mock import MagicMock
        from core.market_data import MarketData

        client = MagicMock()
        bars = [{"c": 100.0 + i * 0.5} for i in range(215)]
        client.get_bars.return_value = bars
        md = MarketData(client)

        res = md.get_sma200("SPY")
        assert res["symbol"] == "SPY"
        assert res["above_sma200"] is True
        assert res["distance_pct"] > 0

    def test_atr_and_momentum_calculation(self):
        from unittest.mock import MagicMock
        from core.market_data import MarketData

        client = MagicMock()
        bars = [{"o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0 + i} for i in range(30)]
        client.get_bars.return_value = bars
        md = MarketData(client)

        atr_res = md.get_atr("AAPL")
        assert atr_res["symbol"] == "AAPL"
        assert atr_res["atr"] > 0

        momo_res = md.get_price_momentum("AAPL")
        assert momo_res["symbol"] == "AAPL"
        assert "roc_10" in momo_res
        assert "roc_20" in momo_res
        assert momo_res["momentum_strength"] in [
            "strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"
        ]


