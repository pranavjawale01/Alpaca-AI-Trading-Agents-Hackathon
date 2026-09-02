"""
core/signal_enhancer.py — Market context builder for LLM Council prompts.

Converts raw market data from MarketData / RiskManager into structured,
human-readable context dicts that the LLM Council can reason about effectively.

Each strategy has its own context builder that includes only the data
relevant to that trade type — keeping prompts focused and token-efficient.
"""

from __future__ import annotations

from datetime import date
from typing import Any


class SignalEnhancer:
    """
    Thin adapter that packages raw market indicators into LLM-ready context dicts.
    Centralises all prompt-engineering for the LLM Council.

    Usage:
        ctx = SignalEnhancer.build_momo_context("NVDA", ema_signal, vol_surge, vix, price)
        result = council.vote("NVDA", ctx, strategy="momentum_call")
    """

    @staticmethod
    def build_momo_context(
        symbol: str,
        ema_signal: dict,
        vol_surge: dict,
        vix: float,
        price: float,
        hist_vol: float | None = None,
        direction: str = "bullish",
        rsi: dict | None = None,
        macd: dict | None = None,
        bollinger: dict | None = None,
        sma200: dict | None = None,
        atr: dict | None = None,
        momentum: dict | None = None,
        portfolio_state: dict | None = None,
    ) -> dict[str, Any]:
        """
        Context for MomoBreakout OTM option buy decisions (Calls for bullish breakouts, Puts for bearish breakdowns).

        Args:
            symbol: Ticker symbol
            ema_signal: Dict from MarketData.get_ema_signal()
            vol_surge: Dict from MarketData.get_volume_surge()
            vix: Current CBOE VIX level
            price: Current underlying price
            hist_vol: Historical 30-day realised volatility (0.0–1.0), optional
            direction: 'bullish' (call) or 'bearish' (put)
            rsi: Dict from MarketData.get_rsi(), optional
            macd: Dict from MarketData.get_macd(), optional
            bollinger: Dict from MarketData.get_bollinger_bands(), optional
            sma200: Dict from MarketData.get_sma200(), optional
            atr: Dict from MarketData.get_atr(), optional
            momentum: Dict from MarketData.get_price_momentum(), optional
            portfolio_state: Dict from SignalEnhancer.build_portfolio_context(), optional
        """
        is_bearish = direction.lower() == "bearish"
        strategy_desc = (
            "Momentum Breakdown — OTM Put Buy (Short Delta)"
            if is_bearish
            else "Momentum Breakout — OTM Call Buy (Long Delta)"
        )
        opt_type = "put" if is_bearish else "call"
        key_risks = [
            "False breakdown if volume surge is short-covering"
            if is_bearish
            else "False breakout if volume surge is sector-wide not stock-specific",
            "Premium decay accelerates if momentum stalls",
            "Elevated VIX inflates premium cost",
        ]

        # ── Build enriched technical signals ──────────────────────────
        tech_signals = {
            "ema_crossover": ema_signal.get("crossover", False),
            "ema_crossover_type": ema_signal.get("crossover_type", "unknown"),
            "ema_signal": ema_signal.get("signal", "unknown"),
            "ema_20": round(ema_signal.get("ema_fast", ema_signal.get("ema20", 0)), 2),
            "ema_50": round(ema_signal.get("ema_slow", ema_signal.get("ema50", 0)), 2),
            "volume_surging": vol_surge.get("is_surging", False),
            "volume_surge_ratio": round(vol_surge.get("surge_ratio", 1.0), 2),
        }

        # Advanced indicators (available when MarketData provides them)
        if rsi is not None:
            tech_signals["rsi_14"] = round(rsi.get("rsi", 50.0), 1)
            tech_signals["rsi_zone"] = rsi.get("zone", "neutral")
        if macd is not None:
            tech_signals["macd_histogram"] = round(macd.get("histogram", 0.0), 4)
            tech_signals["macd_bullish_cross"] = macd.get("macd_bullish_cross", False)
            tech_signals["macd_bearish_cross"] = macd.get("macd_bearish_cross", False)
        if bollinger is not None:
            tech_signals["bollinger_pct_b"] = round(bollinger.get("pct_b", 0.5), 3)
            tech_signals["bollinger_squeeze"] = bollinger.get("is_squeeze", False)
            tech_signals["bollinger_bandwidth"] = round(bollinger.get("bandwidth", 0.0), 2)
        if sma200 is not None:
            tech_signals["price_above_sma200"] = sma200.get("above_sma200", True)
            tech_signals["sma200_distance_pct"] = round(sma200.get("distance_pct", 0.0), 2)
        if atr is not None:
            tech_signals["atr_14"] = round(atr.get("atr", 0.0), 2)
            tech_signals["atr_pct"] = round(atr.get("atr_pct", 0.0), 2)
        if momentum is not None:
            tech_signals["momentum_roc_10"] = round(momentum.get("roc_10", 0.0), 2)
            tech_signals["momentum_strength"] = momentum.get("momentum_strength", "neutral")

        ctx = {
            "strategy": strategy_desc,
            "symbol": symbol,
            "direction": direction,
            "current_price": round(price, 2),
            "date": date.today().isoformat(),
            "technical_signals": tech_signals,
            "market_conditions": {
                "vix": round(vix, 1),
                "vix_regime": _classify_vix(vix),
                "historical_volatility_30d": round(hist_vol, 4) if hist_vol else None,
            },
            "trade_parameters": {
                "option_type": opt_type,
                "moneyness": "5% OTM",
                "target_dte": "30-45 days",
                "max_premium_pct_equity": "1%",
                "profit_target": "100% (2x premium)",
                "stop_loss": "50% of premium",
            },
            "key_risks": key_risks,
        }

        if portfolio_state is not None:
            ctx["portfolio_state"] = portfolio_state

        return ctx

    @staticmethod
    def build_theta_context(
        symbol: str,
        price: float,
        ivr: float,
        vix: float,
        hist_vol: float,
        target_strike: float | None = None,
        dte: int = 35,
        premium_annualised_yield: float | None = None,
        portfolio_state: dict | None = None,
    ) -> dict[str, Any]:
        """
        Context for ThetaCollector cash-secured put sell decisions.

        Args:
            symbol: Ticker symbol (ETF like SPY, QQQ)
            price: Current underlying price
            ivr: Implied Volatility Rank (0–100)
            vix: Current VIX
            hist_vol: 30-day realised volatility
            target_strike: Proposed put strike (10% OTM), optional
            dte: Days to expiration for the target contract
            premium_annualised_yield: Annualised premium yield %, optional
            portfolio_state: Dict from SignalEnhancer.build_portfolio_context(), optional
        """
        ctx = {
            "strategy": "Cash-Secured Put — Theta Collection",
            "symbol": symbol,
            "current_price": round(price, 2),
            "date": date.today().isoformat(),
            "volatility": {
                "ivr": round(ivr, 1),
                "ivr_interpretation": _classify_ivr(ivr),
                "vix": round(vix, 1),
                "vix_regime": _classify_vix(vix),
                "historical_vol_30d": round(hist_vol, 4),
                "iv_hv_ratio": round(ivr / max(hist_vol * 200, 1), 2) if hist_vol else None,
            },
            "trade_parameters": {
                "option_type": "put (selling)",
                "target_strike": round(target_strike, 2) if target_strike else "~10% OTM",
                "target_dte": dte,
                "target_delta": "~0.20",
                "profit_target": "50% of premium",
                "time_stop_dte": 21,
                "stop_loss": "2x premium received",
                "annualised_yield_pct": premium_annualised_yield,
            },
            "entry_checklist": {
                "ivr_above_30": ivr >= 30,
                "vix_below_30": vix < 30,
                "premium_income_favorable": ivr >= 40,
            },
            "key_risks": [
                "Underlying drops sharply → put goes ITM → assignment risk",
                "IV expands further → unrealised loss on short put",
                "Low IVR means premium income not worth the risk",
            ],
        }

        if portfolio_state is not None:
            ctx["portfolio_state"] = portfolio_state

        return ctx

    @staticmethod
    def build_iv_crush_context(
        symbol: str,
        price: float,
        vix: float,
        days_to_earnings: int,
        ivr: float,
        atm_strike: float | None = None,
        dte: int = 10,
        implied_move_pct: float | None = None,
        portfolio_state: dict | None = None,
    ) -> dict[str, Any]:
        """
        Context for IVCrush ATM straddle sell ahead of earnings.

        Args:
            symbol: Ticker with upcoming earnings
            price: Current underlying price
            vix: Current VIX
            days_to_earnings: Calendar days until earnings announcement
            ivr: Current Implied Volatility Rank (0–100)
            atm_strike: The ATM strike for the straddle, optional
            dte: Days to expiration of the options contracts
            implied_move_pct: Market-implied ±% move for earnings, optional
            portfolio_state: Dict from SignalEnhancer.build_portfolio_context(), optional
        """
        ctx = {
            "strategy": "Earnings IV Crush — ATM Straddle Sell",
            "symbol": symbol,
            "current_price": round(price, 2),
            "date": date.today().isoformat(),
            "earnings_event": {
                "days_to_earnings": days_to_earnings,
                "timing_assessment": _classify_earnings_timing(days_to_earnings),
            },
            "volatility": {
                "ivr": round(ivr, 1),
                "ivr_for_iv_crush": _classify_ivr_for_iv_crush(ivr),
                "vix": round(vix, 1),
                "vix_regime": _classify_vix(vix),
                "implied_move_pct": implied_move_pct,
            },
            "trade_parameters": {
                "structure": "short ATM straddle (sell call + sell put)",
                "atm_strike": round(atm_strike, 2) if atm_strike else "ATM",
                "target_dte": dte,
                "profit_target": "40% of combined premium",
                "hold_until": "1 trading day post-earnings",
                "stop_loss": "1.5x combined premium",
            },
            "entry_checklist": {
                "ivr_above_60": ivr >= 60,
                "vix_below_30": vix < 30,
                "earnings_1_to_3_days": 1 <= days_to_earnings <= 3,
                "delta_neutral_entry": True,
            },
            "key_risks": [
                "Actual earnings move exceeds implied move → large loss on either leg",
                "IV does not collapse post-earnings (rare but possible)",
                "Gap risk: earnings outside market hours make exit difficult",
            ],
        }

        if portfolio_state is not None:
            ctx["portfolio_state"] = portfolio_state

        return ctx

    @staticmethod
    def build_portfolio_context(
        risk_manager: Any,
        open_positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Build portfolio-level risk context for council evaluators.
        """
        equity = getattr(risk_manager, "equity", 100000.0)
        daily_pnl = getattr(risk_manager, "daily_pnl", 0.0)
        portfolio_delta = getattr(risk_manager, "portfolio_delta", 0.0)
        options_exp = getattr(risk_manager, "total_options_exposure", 0.0)
        vix = getattr(risk_manager, "current_vix", 18.0)

        # Max allowed limits
        max_exp = equity * 0.30
        portfolio_heat = options_exp / max_exp if max_exp > 0 else 0.0
        daily_loss_limit = equity * 0.02
        pnl_budget_used = abs(daily_pnl) / daily_loss_limit if (daily_pnl < 0 and daily_loss_limit > 0) else 0.0

        tech_symbols = {"NVDA", "TSLA", "AAPL", "META", "AMZN", "MSFT", "GOOGL"}
        pos_symbols: list[str] = []
        if open_positions:
            for p in open_positions:
                sym = p.get("symbol", "")
                if "/" in sym:
                    sym = sym.split()[0]
                pos_symbols.append(sym)
        correlated_count = sum(1 for s in pos_symbols if any(tech in s for tech in tech_symbols))

        return {
            "equity": round(equity, 2),
            "daily_pnl": round(daily_pnl, 2),
            "portfolio_delta": round(portfolio_delta, 2),
            "options_exposure": round(options_exp, 2),
            "portfolio_heat": round(min(portfolio_heat, 1.0), 3),
            "pnl_budget_used": round(min(pnl_budget_used, 1.0), 3),
            "vix": round(vix, 1),
            "open_positions_count": len(pos_symbols),
            "correlated_tech_count": correlated_count,
            "delta_headroom": round(50.0 - abs(portfolio_delta), 1),
        }


# ── Helper classifiers for LLM context ───────────────────────

def _classify_vix(vix: float) -> str:
    if vix < 15:
        return "very_low — complacency, ideal for premium selling"
    elif vix < 20:
        return "low — risk-on, normal conditions"
    elif vix < 28:
        return "elevated — neutral/cautious"
    elif vix < 35:
        return "high — risk-off, avoid aggressive trades"
    else:
        return "extreme — VIX kill switch territory"


def _classify_ivr(ivr: float) -> str:
    if ivr < 20:
        return "very_low — premium is cheap, avoid selling"
    elif ivr < 40:
        return "low — below average, cautious on premium selling"
    elif ivr < 60:
        return "moderate — acceptable for premium selling"
    elif ivr < 80:
        return "high — good premium selling environment"
    else:
        return "very_high — excellent for premium selling, high edge"


def _classify_ivr_for_iv_crush(ivr: float) -> str:
    if ivr < 50:
        return "insufficient — IV crush trade likely unprofitable"
    elif ivr < 65:
        return "borderline — marginal IV crush opportunity"
    elif ivr < 80:
        return "good — strong earnings premium, favourable for straddle sell"
    else:
        return "excellent — very high earnings premium, high-conviction IV crush"


def _classify_earnings_timing(days: int) -> str:
    if days == 1:
        return "imminent — earnings tomorrow, optimal timing"
    elif days == 2:
        return "near — 2 days, good timing for straddle entry"
    elif days == 3:
        return "acceptable — 3 days, slight theta decay before event"
    else:
        return f"too_far — {days} days, too early for earnings trade"
