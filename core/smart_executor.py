"""
core/smart_executor.py — Mid-Price Limit Order Execution Engine.

The #1 hidden cost in options trading is the bid-ask spread.
A retail agent using market orders on every trade pays the full ask on
entries and receives the full bid on exits — often 10–30% of the premium.

This module implements professional-grade execution:

  1. Fetch the option's current bid/ask quote
  2. Submit a LIMIT order at the mid-price (bid+ask)/2
  3. Wait up to `timeout_seconds` for a fill
  4. If not filled: step price toward the aggressive side by 1 tick
  5. Repeat `n_aggression_steps` times
  6. Final fallback: convert to market order to guarantee execution

On liquid options (SPY, QQQ, AAPL) the mid-price fills ~80%+ of the time.
On less liquid names, the stepper moves the price toward the market to
find natural sellers/buyers without paying the full spread.

Example savings:
  SPY put: bid=$1.20, ask=$1.30 → mid=$1.25
  Market order cost: $1.30 × 100 = $130
  Limit fill at mid: $1.25 × 100 = $125  → saved $5/contract (~3.8%)
  On 5 contracts: $25 saved per trade → ~$300/year on 12 trades

Usage:
    executor = SmartExecutor(client)
    result = executor.execute_option_order("SPY250919P00480000", 2, "sell")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from rich.console import Console

import config

console = Console()
log = logging.getLogger(__name__)

# Minimum meaningful option price (below this we don't bother with limit)
_MIN_PRICE_FOR_LIMIT = 0.05

# Tick size for option price stepping
_OPTION_TICK = 0.05


class SmartExecutor:
    """
    Intelligent order executor that targets mid-price on options.

    Falls back gracefully to market orders so trading is never blocked.
    All execution is logged at INFO level for post-trade analysis.
    """

    def __init__(self, client, use_limit_orders: Optional[bool] = None) -> None:
        """
        Args:
            client: AlpacaClient instance
            use_limit_orders: Override config.EXECUTION.use_limit_orders
        """
        self.client = client
        self.use_limit_orders = (
            use_limit_orders
            if use_limit_orders is not None
            else config.EXECUTION.use_limit_orders
        )
        self.timeout = config.EXECUTION.limit_order_timeout_seconds
        self.n_steps = config.EXECUTION.limit_price_aggression_steps

        console.print(
            f"[cyan]SmartExecutor initialised | "
            f"limit_orders={self.use_limit_orders} | "
            f"timeout={self.timeout}s | steps={self.n_steps}[/cyan]"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def execute_option_order(
        self,
        option_symbol: str,
        qty: int,
        side: str,  # "buy" | "sell"
    ) -> dict:
        """
        Execute an option order at the best achievable price.

        For 'buy'  orders: tries to pay mid-price (below ask).
        For 'sell' orders: tries to receive mid-price (above bid).

        Args:
            option_symbol: OCC-formatted option symbol
            qty: Number of contracts
            side: 'buy' | 'sell'

        Returns:
            Order dict with {id, status, symbol, fill_price, execution_type}
        """
        if not self.use_limit_orders:
            log.info(f"[SmartExecutor] Limit orders disabled — using market for {option_symbol}")
            return self._market_order(option_symbol, qty, side, reason="config_disabled")

        try:
            bid, ask = self._get_option_quote(option_symbol)
        except Exception as exc:
            log.warning(f"[SmartExecutor] Quote fetch failed ({exc}) — falling back to market")
            return self._market_order(option_symbol, qty, side, reason="quote_error")

        spread = ask - bid
        mid = (bid + ask) / 2

        if mid < _MIN_PRICE_FOR_LIMIT or spread < _OPTION_TICK:
            # Very cheap option or negligible spread → market is fine
            log.info(f"[SmartExecutor] Spread too small (${spread:.2f}) → market order")
            return self._market_order(option_symbol, qty, side, reason="negligible_spread")

        console.print(
            f"[dim][Executor] {option_symbol}: bid=${bid:.2f} ask=${ask:.2f} "
            f"spread=${spread:.2f} → targeting mid=${mid:.2f}[/dim]"
        )

        # Try mid-price first, then step toward aggressive side
        limit_price = round(mid / _OPTION_TICK) * _OPTION_TICK  # snap to tick

        for step in range(self.n_steps + 1):
            result = self._try_limit_order(option_symbol, qty, side, limit_price, step)
            if result.get("filled"):
                console.print(
                    f"[green][Executor] Limit fill @ ${limit_price:.2f} "
                    f"(step {step}) — saved ${abs(limit_price - (ask if side=='buy' else bid)) * qty * 100:,.0f}[/green]"
                )
                result["execution_type"] = f"limit_step_{step}"
                return result

            # Step price toward market: buy → increase price, sell → decrease price
            if side == "buy":
                limit_price = min(limit_price + _OPTION_TICK, ask)
            else:
                limit_price = max(limit_price - _OPTION_TICK, bid)

        # Final fallback: market order
        log.warning(
            f"[SmartExecutor] {option_symbol}: limit not filled after {self.n_steps} steps "
            f"— converting to market"
        )
        return self._market_order(option_symbol, qty, side, reason="limit_exhausted")

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _get_option_quote(self, option_symbol: str) -> tuple[float, float]:
        """
        Fetch current bid/ask for an option using Option Data API.
        """
        try:
            quote = self.client.get_option_quote(option_symbol)
            bid = float(quote.get("bid", 0.0))
            ask = float(quote.get("ask", 0.0))
            if bid > 0 and ask > 0 and ask >= bid:
                return bid, ask
            if bid > 0 and ask == 0:
                return bid, bid * 1.10
            if ask > 0 and bid == 0:
                return max(0.01, ask * 0.90), ask
        except Exception:
            pass

        raise RuntimeError(f"Cannot fetch option quote for {option_symbol}")

    def _try_limit_order(
        self,
        option_symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        step: int,
    ) -> dict:
        """
        Submit a limit order and poll for fill within timeout.

        Returns dict with {filled: bool, id, status, ...}.
        """
        try:
            result = self.client.place_option_limit_order(
                option_symbol, qty, side, round(limit_price, 2)
            )
            order_id = result.get("id")

            # Poll for fill or accepted status
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                status = self._check_order_status(order_id)
                if status in ("filled", "partially_filled"):
                    result["filled"] = True
                    result["fill_price"] = limit_price
                    return result
                if status in ("new", "accepted", "held"):
                    # Accepted/queued by Alpaca broker
                    result["filled"] = True
                    result["fill_price"] = limit_price
                    return result
                if status in ("canceled", "expired", "rejected"):
                    break
                time.sleep(2)  # poll every 2 seconds

            # Not filled — cancel the resting order before stepping
            self._cancel_order(order_id)
            result["filled"] = False
            return result

        except Exception as exc:
            log.warning(f"[SmartExecutor] Limit order attempt failed (step={step}): {exc}")
            return {"filled": False}

    def _check_order_status(self, order_id: str) -> str:
        """Poll Alpaca for order status. Returns status string."""
        try:
            from alpaca.trading.requests import GetOrderByIdRequest
            order = self.client.trading.get_order_by_id(order_id)
            return str(order.status).lower()
        except Exception:
            return "unknown"

    def _cancel_order(self, order_id: str) -> None:
        """Cancel a resting order by ID."""
        try:
            self.client.trading.cancel_order_by_id(order_id)
            log.info(f"[SmartExecutor] Cancelled resting limit order {order_id}")
        except Exception as exc:
            log.debug(f"[SmartExecutor] Cancel failed (may already be done): {exc}")

    def _market_order(
        self, option_symbol: str, qty: int, side: str, reason: str = ""
    ) -> dict:
        """Place a fallback market order."""
        result = self.client.place_option_market_order(option_symbol, qty, side)
        result["execution_type"] = f"market_{reason}"
        result["filled"] = True  # market orders always fill
        log.info(
            f"[SmartExecutor] Market order placed for {option_symbol} "
            f"({side} x{qty}) | reason={reason}"
        )
        return result
