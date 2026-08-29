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
    ) -> dict[str, Any]:
        """
        Context for MomoBreakout OTM call buy decisions.

        Args:
            symbol: Ticker symbol
            ema_signal: Dict from MarketData.get_ema_signal() with keys:
                        crossover (bool), signal (str), ema20, ema50
            vol_surge: Dict from MarketData.get_volume_surge() with keys:
                       is_surging (bool), surge_ratio (float)
            vix: Current CBOE VIX level
            price: Current underlying price
            hist_vol: Historical 30-day realised volatility (0.0–1.0), optional
        """
        return {
            "strategy": "Momentum Breakout — OTM Call Buy",
            "symbol": symbol,
            "current_price": round(price, 2),
            "date": date.today().isoformat(),
            "technical_signals": {
                "ema_crossover": ema_signal.get("crossover", False),
                "ema_signal": ema_signal.get("signal", "unknown"),
                "ema_20": round(ema_signal.get("ema20", 0), 2),
                "ema_50": round(ema_signal.get("ema50", 0), 2),
                "volume_surging": vol_surge.get("is_surging", False),
                "volume_surge_ratio": round(vol_surge.get("surge_ratio", 1.0), 2),
            },
            "market_conditions": {
                "vix": round(vix, 1),
                "vix_regime": _classify_vix(vix),
                "historical_volatility_30d": round(hist_vol, 4) if hist_vol else None,
            },
            "trade_parameters": {
                "option_type": "call",
                "moneyness": "5% OTM",
                "target_dte": "30-45 days",
                "max_premium_pct_equity": "1%",
                "profit_target": "100% (2x premium)",
                "stop_loss": "50% of premium",
            },
            "key_risks": [
                "False breakout if volume surge is sector-wide not stock-specific",
                "Premium decay accelerates if momentum stalls",
                "Elevated VIX inflates premium cost",
            ],
        }

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
        """
        return {
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
        """
        return {
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
