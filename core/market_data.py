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
        Estimate VIX using VIXY ETF price as a proxy.
        VIXY tracks short-term VIX futures — its price is roughly in the 15-30
        range mirroring VIX. We use it directly (no scaling needed).
        Falls back to 18.0 if unavailable.
        """
        try:
            # VIXY price ≈ VIX level (both typically 12-40 range)
            quote = self.client.get_latest_quote("VIXY")
            vix_approx = quote["mid"]
            log.info(f"VIX proxy (VIXY price): {vix_approx:.1f}")
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

        if len(closes) < slow + 2:
            log.warning(f"[{symbol}] Not enough bars ({len(closes)}) for EMA signal")
            return {"symbol": symbol, "ema_fast": 0, "ema_slow": 0, "signal": "neutral", "crossover": False}

        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)

        prev_fast = self._ema(closes[:-1], fast)
        prev_slow = self._ema(closes[:-1], slow)


        crossover_bullish = (prev_fast < prev_slow) and (ema_fast > ema_slow)
        crossover_bearish = (prev_fast > prev_slow) and (ema_fast < ema_slow)
        crossover = crossover_bullish or crossover_bearish
        crossover_type = "bullish" if crossover_bullish else ("bearish" if crossover_bearish else "none")

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
            "crossover_type": crossover_type,
            "crossover_bullish": crossover_bullish,
            "crossover_bearish": crossover_bearish,
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
        if len(volumes) < 2:
            return {
                "symbol": symbol,
                "today_volume": 0.0,
                "avg_volume": 1.0,
                "surge_ratio": 1.0,
                "is_surging": False,
            }
        avg_volume = float(np.mean(volumes[:-1]))
        today_volume = float(volumes[-1])
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
        Returns a default of 0.20 (20%) if insufficient data.
        """
        bars = self.client.get_bars(symbol, "1Day", limit=window + 5)
        closes = np.array([b["c"] for b in bars])
        if len(closes) < 5:
            log.warning(f"[{symbol}] Not enough bars for vol estimate, using 20% default")
            return 0.20
        log_returns = np.diff(np.log(closes))
        daily_vol = np.std(log_returns)
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol) if annual_vol > 0 else 0.20


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

    # ─────────────────────────────────────────
    # Technical Indicators
    # ─────────────────────────────────────────

    def get_rsi(self, symbol: str, period: int = 14) -> dict:
        """Compute RSI using Wilder's smoothed average of gains/losses."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=period + 5)
            closes = [b["c"] for b in bars]
            if len(closes) < period + 1:
                return {"symbol": symbol, "rsi": 50.0, "zone": "neutral"}
            
            diffs = np.diff(closes)
            gains = np.where(diffs > 0, diffs, 0)
            losses = np.where(diffs < 0, -diffs, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100.0
            zone = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
            return {"symbol": symbol, "rsi": float(rsi), "zone": zone}
        except Exception as e:
            log.warning(f"[{symbol}] get_rsi failed: {e}")
            return {"symbol": symbol, "rsi": 50.0, "zone": "neutral"}

    def get_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal_period: int = 9) -> dict:
        """Compute MACD, Signal line, and Histogram with crossover detection."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=slow + signal_period + 5)
            closes = [b["c"] for b in bars]
            if len(closes) < slow + signal_period:
                return {"symbol": symbol, "macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0, "macd_bullish_cross": False, "macd_bearish_cross": False}
                
            ema_fast = [self._ema(closes[:i+1], fast) for i in range(slow-1, len(closes))]
            ema_slow = [self._ema(closes[:i+1], slow) for i in range(slow-1, len(closes))]
            macd_series = [f - s for f, s in zip(ema_fast, ema_slow)]
            
            macd_line = macd_series[-1]
            prev_macd_line = macd_series[-2]
            
            signal_line = self._ema(macd_series[-signal_period:], signal_period)
            prev_signal_line = self._ema(macd_series[-signal_period-1:-1], signal_period)
            
            histogram = macd_line - signal_line
            bullish_cross = (prev_macd_line <= prev_signal_line) and (macd_line > signal_line)
            bearish_cross = (prev_macd_line >= prev_signal_line) and (macd_line < signal_line)
            
            return {
                "symbol": symbol,
                "macd_line": float(macd_line),
                "signal_line": float(signal_line),
                "histogram": float(histogram),
                "macd_bullish_cross": bool(bullish_cross),
                "macd_bearish_cross": bool(bearish_cross)
            }
        except Exception as e:
            log.warning(f"[{symbol}] get_macd failed: {e}")
            return {"symbol": symbol, "macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0, "macd_bullish_cross": False, "macd_bearish_cross": False}

    def get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict:
        """Compute Bollinger Bands, bandwidth, %B, and squeeze detection."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=period + 10)
            closes = [b["c"] for b in bars]
            if len(closes) < period:
                return {"symbol": symbol, "upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0, "pct_b": 0.0, "is_squeeze": False}
                
            bandwidths = []
            for i in range(period, len(closes) + 1):
                window = closes[i-period:i]
                mid = np.mean(window)
                std = np.std(window, ddof=0)
                up = mid + std_dev * std
                dn = mid - std_dev * std
                bw = (up - dn) / mid * 100 if mid > 0 else 0
                bandwidths.append(bw)
                if i == len(closes):
                    upper = up
                    middle = mid
                    lower = dn
                    bandwidth = bw
                    
            price = closes[-1]
            pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5
            is_squeeze = bandwidth <= np.percentile(bandwidths, 20)
            
            return {
                "symbol": symbol,
                "upper": float(upper),
                "middle": float(middle),
                "lower": float(lower),
                "bandwidth": float(bandwidth),
                "pct_b": float(pct_b),
                "is_squeeze": bool(is_squeeze)
            }
        except Exception as e:
            log.warning(f"[{symbol}] get_bollinger_bands failed: {e}")
            return {"symbol": symbol, "upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0, "pct_b": 0.0, "is_squeeze": False}

    def get_sma200(self, symbol: str) -> dict:
        """Compute SMA of last 200 closes and distance percentage."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=210)
            closes = [b["c"] for b in bars]
            if len(closes) < 200:
                return {"symbol": symbol, "sma200": 0.0, "price": 0.0, "above_sma200": True, "distance_pct": 0.0}
            sma200 = float(np.mean(closes[-200:]))
            price = closes[-1]
            distance_pct = (price - sma200) / sma200 * 100 if sma200 > 0 else 0.0
            return {
                "symbol": symbol,
                "sma200": sma200,
                "price": price,
                "above_sma200": price > sma200,
                "distance_pct": distance_pct
            }
        except Exception as e:
            log.warning(f"[{symbol}] get_sma200 failed: {e}")
            return {"symbol": symbol, "sma200": 0.0, "price": 0.0, "above_sma200": True, "distance_pct": 0.0}

    def get_atr(self, symbol: str, period: int = 14) -> dict:
        """Compute Average True Range using Wilder's smoothing."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=period + 5)
            if len(bars) < period + 1:
                return {"symbol": symbol, "atr": 0.0, "atr_pct": 0.0}
            
            trs = []
            for i in range(1, len(bars)):
                h = bars[i]["h"]
                l = bars[i]["l"]
                prev_c = bars[i-1]["c"]
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                trs.append(tr)
                
            atr = np.mean(trs[:period])
            for i in range(period, len(trs)):
                atr = (atr * (period - 1) + trs[i]) / period
                
            price = bars[-1]["c"]
            atr_pct = atr / price * 100 if price > 0 else 0.0
            
            return {"symbol": symbol, "atr": float(atr), "atr_pct": float(atr_pct)}
        except Exception as e:
            log.warning(f"[{symbol}] get_atr failed: {e}")
            return {"symbol": symbol, "atr": 0.0, "atr_pct": 0.0}

    def get_price_momentum(self, symbol: str) -> dict:
        """Compute Rate of Change and classify momentum strength."""
        try:
            bars = self.client.get_bars(symbol, "1Day", limit=25)
            closes = [b["c"] for b in bars]
            if len(closes) < 21:
                return {"symbol": symbol, "roc_10": 0.0, "roc_20": 0.0, "momentum_strength": "neutral"}
                
            today = closes[-1]
            c10 = closes[-11]
            c20 = closes[-21]
            roc_10 = (today - c10) / c10 * 100 if c10 > 0 else 0.0
            roc_20 = (today - c20) / c20 * 100 if c20 > 0 else 0.0
            
            if roc_10 > 5:
                strength = "strong_bullish"
            elif roc_10 > 2:
                strength = "bullish"
            elif roc_10 < -5:
                strength = "strong_bearish"
            elif roc_10 < -2:
                strength = "bearish"
            else:
                strength = "neutral"
                
            return {
                "symbol": symbol,
                "roc_10": float(roc_10),
                "roc_20": float(roc_20),
                "momentum_strength": strength
            }
        except Exception as e:
            log.warning(f"[{symbol}] get_price_momentum failed: {e}")
            return {"symbol": symbol, "roc_10": 0.0, "roc_20": 0.0, "momentum_strength": "neutral"}
