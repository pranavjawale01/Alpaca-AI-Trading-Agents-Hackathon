"""
core/market_data.py — Market data helpers.

Provides:
  - get_vix(): Current VIX level (fear gauge)
  - get_price(): Latest price for any symbol
  - get_moving_averages(): EMA crossover signals
  - get_iv_rank(): Implied volatility rank for premium-selling signals
  - get_earnings_calendar(): Upcoming earnings dates
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import requests

from core.alpaca_client import AlpacaClient

log = logging.getLogger(__name__)


class MarketData:
    """
    Market data accessor built on top of Alpaca's Data API.

    Usage:
        md = MarketData(client)
        vix = md.get_vix()
        price = md.get_price("SPY")
        signal = md.get_ema_signal("NVDA")
    """

    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    # ─────────────────────────────────────────
    # Prices
    # ─────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Get current mid-price for a symbol."""
        quote = self.client.get_latest_quote(symbol)
        return quote["mid"]

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get prices for multiple symbols."""
        return {sym: self.get_price(sym) for sym in symbols}

    # ─────────────────────────────────────────
    # VIX
    # ─────────────────────────────────────────

    def get_vix(self) -> float:
        """
        Fetch VIX level via Alpaca (VIX = ticker 'VIXY' as proxy,
        or fallback to a static value during pre-market).
        Returns VIX as a float (e.g. 18.5).
        """
        try:
            # VIXY is the VIX ETF — a reasonable real-time proxy
            quote = self.client.get_latest_quote("VIXY")
            # VIXY ≈ VIX/10 roughly; adjust scale heuristically
            vix_approx = quote["mid"] * 10
            log.info(f"VIX proxy (VIXY): {vix_approx:.1f}")
            return vix_approx
        except Exception as e:
            log.warning(f"VIX fetch failed, using default: {e}")
            return 18.0  # reasonable default

    # ─────────────────────────────────────────
    # Moving Averages / EMA Signal
    # ─────────────────────────────────────────

    def _ema(self, prices: list[float], period: int) -> float:
        """Compute EMA of price series."""
        k = 2 / (period + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema

    def get_ema_signal(
        self,
        symbol: str,
        fast: int = 20,
        slow: int = 50,
    ) -> dict:
        """
        Compute EMA crossover signal.

        Returns:
            {
                "symbol": str,
                "ema_fast": float,
                "ema_slow": float,
                "signal": "bullish" | "bearish" | "neutral",
                "crossover": bool
            }
        """
        bars = self.client.get_bars(symbol, "1Day", limit=slow + 10)
        closes = [b["c"] for b in bars]

        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)

        prev_fast = self._ema(closes[:-1], fast)
        prev_slow = self._ema(closes[:-1], slow)

        crossover = (prev_fast < prev_slow) and (ema_fast > ema_slow)
        if ema_fast > ema_slow * 1.002:
            signal = "bullish"
        elif ema_fast < ema_slow * 0.998:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "symbol": symbol,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "signal": signal,
            "crossover": crossover,
        }

    # ─────────────────────────────────────────
    # Volume Surge (Momo Signal)
    # ─────────────────────────────────────────

    def get_volume_surge(self, symbol: str, lookback: int = 20) -> dict:
        """
        Detect volume surge vs 20-day average.
        Surge ratio > 2.0 → strong momentum signal.
        """
        bars = self.client.get_bars(symbol, "1Day", limit=lookback + 1)
        volumes = [b["v"] for b in bars]
        avg_volume = np.mean(volumes[:-1])
        today_volume = volumes[-1]
        surge_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            "symbol": symbol,
            "today_volume": today_volume,
            "avg_volume": avg_volume,
            "surge_ratio": surge_ratio,
            "is_surging": surge_ratio >= 2.0,
        }

    # ─────────────────────────────────────────
    # IV Rank (approx from historical bars)
    # ─────────────────────────────────────────

    def estimate_historical_vol(self, symbol: str, window: int = 30) -> float:
        """
        Estimate 30-day historical volatility (annualised) from daily closes.
        Used as a proxy for IV when live options data isn't available.
        """
        bars = self.client.get_bars(symbol, "1Day", limit=window + 5)
        closes = np.array([b["c"] for b in bars])
        log_returns = np.diff(np.log(closes))
        daily_vol = np.std(log_returns)
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)

    # ─────────────────────────────────────────
    # ATM Strike Finder
    # ─────────────────────────────────────────

    def find_atm_strike(
        self,
        symbol: str,
        available_strikes: list[float],
    ) -> float:
        """Find the strike closest to current price (ATM)."""
        price = self.get_price(symbol)
        return min(available_strikes, key=lambda k: abs(k - price))

    def find_otm_strike(
        self,
        symbol: str,
        available_strikes: list[float],
        option_type: str = "call",
        otm_pct: float = 0.05,
    ) -> float:
        """
        Find a strike OTM by approximately otm_pct%.
        E.g. otm_pct=0.05 finds a strike 5% OTM.
        """
        price = self.get_price(symbol)
        if option_type == "call":
            target = price * (1 + otm_pct)
            candidates = [k for k in available_strikes if k > price]
        else:
            target = price * (1 - otm_pct)
            candidates = [k for k in available_strikes if k < price]

        if not candidates:
            return self.find_atm_strike(symbol, available_strikes)
        return min(candidates, key=lambda k: abs(k - target))
