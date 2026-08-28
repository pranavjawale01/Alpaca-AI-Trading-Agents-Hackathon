"""
core/risk_manager.py — Enforcement of all risk gates.

Every agent MUST call RiskManager.approve_order() before placing any trade.
If any gate fails, the method raises RiskViolation with a clear reason.

Risk Gates:
  1. Max single position size (5% of equity)
  2. Max total options exposure (30% of equity)
  3. Daily loss limit (-2% of starting equity)
  4. Portfolio delta limits (-50 to +50)
  5. VIX kill switch (halt if VIX > 35)
  6. Earnings cooldown (no trades 2h before/after events)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console

import config
from config import RISK

console = Console()
log = logging.getLogger(__name__)


class RiskViolation(Exception):
    """Raised when a risk gate blocks an order."""
    pass


class RiskManager:
    """
    Stateful risk manager — tracks daily P&L, current exposure, and
    portfolio Greeks to enforce all configured risk gates.

    Usage:
        rm = RiskManager(equity=100_000)
        rm.update_daily_pnl(-500)
        rm.approve_order("SPY", order_value=4000, delta_impact=5.0)
    """

    def __init__(self, equity: float = RISK.starting_balance) -> None:
        self.equity = equity
        self.daily_pnl: float = 0.0
        self.portfolio_delta: float = 0.0
        self.total_options_exposure: float = 0.0
        self.current_vix: float = 15.0   # updated by market data module
        self._earnings_events: list[datetime] = []

        console.print(
            f"[cyan]RiskManager initialised | equity=${equity:,.0f}[/cyan]"
        )

    # ─────────────────────────────────────────
    # State Updaters (called by agents)
    # ─────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        self.equity = equity

    def update_daily_pnl(self, pnl: float) -> None:
        """Set absolute daily P&L (not incremental)."""
        self.daily_pnl = pnl

    def update_vix(self, vix: float) -> None:
        self.current_vix = vix

    def update_portfolio_delta(self, delta: float) -> None:
        self.portfolio_delta = delta

    def update_options_exposure(self, exposure: float) -> None:
        """Total notional value of open options positions."""
        self.total_options_exposure = exposure

    def add_earnings_event(self, event_time: datetime) -> None:
        """Register an upcoming earnings announcement to enforce cooldown."""
        self._earnings_events.append(event_time)

    # ─────────────────────────────────────────
    # Gate Checks
    # ─────────────────────────────────────────

    def _check_vix_kill_switch(self) -> None:
        if self.current_vix >= RISK.vix_kill_switch:
            raise RiskViolation(
                f"VIX kill switch triggered: VIX={self.current_vix:.1f} >= {RISK.vix_kill_switch}"
            )

    def _check_daily_loss_limit(self) -> None:
        loss_limit = -RISK.daily_loss_limit_pct * RISK.starting_balance
        if self.daily_pnl <= loss_limit:
            raise RiskViolation(
                f"Daily loss limit hit: P&L=${self.daily_pnl:,.0f} <= limit=${loss_limit:,.0f}"
            )

    def _check_position_size(self, order_value: float) -> None:
        max_size = RISK.max_position_pct * self.equity
        if order_value > max_size:
            raise RiskViolation(
                f"Position too large: ${order_value:,.0f} > max ${max_size:,.0f} "
                f"({RISK.max_position_pct*100:.0f}% of equity)"
            )

    def _check_options_exposure(self, additional_exposure: float) -> None:
        max_exposure = RISK.max_options_exposure_pct * self.equity
        projected = self.total_options_exposure + additional_exposure
        if projected > max_exposure:
            raise RiskViolation(
                f"Options exposure too high: projected=${projected:,.0f} > max=${max_exposure:,.0f}"
            )

    def _check_delta_limits(self, delta_impact: float) -> None:
        projected_delta = self.portfolio_delta + delta_impact
        if not (RISK.min_portfolio_delta <= projected_delta <= RISK.max_portfolio_delta):
            raise RiskViolation(
                f"Delta out of bounds: projected={projected_delta:.1f} "
                f"(allowed: {RISK.min_portfolio_delta} to {RISK.max_portfolio_delta})"
            )

    def _check_earnings_cooldown(self) -> None:
        now = datetime.now(timezone.utc)
        cooldown_mins = RISK.earnings_cooldown_minutes
        for event in self._earnings_events:
            diff_mins = abs((now - event).total_seconds() / 60)
            if diff_mins < cooldown_mins:
                raise RiskViolation(
                    f"Earnings cooldown active: event in {diff_mins:.0f}m "
                    f"(cooldown={cooldown_mins}m)"
                )

    # ─────────────────────────────────────────
    # Main Approval Gate
    # ─────────────────────────────────────────

    def approve_order(
        self,
        symbol: str,
        order_value: float,
        delta_impact: float = 0.0,
        is_option: bool = False,
        skip_earnings_check: bool = False,
    ) -> None:
        """
        Approve or reject an order based on all active risk gates.

        Args:
            symbol: Ticker symbol
            order_value: Notional value of the order in USD
            delta_impact: Change in portfolio delta this order causes
            is_option: Whether this is an options order
            skip_earnings_check: Set True for closing/hedge orders

        Raises:
            RiskViolation: If any gate fails — caller must catch this.
        """
        checks = [
            self._check_vix_kill_switch,
            self._check_daily_loss_limit,
            lambda: self._check_position_size(order_value),
            lambda: self._check_delta_limits(delta_impact),
        ]
        if is_option:
            checks.append(lambda: self._check_options_exposure(order_value))
        if not skip_earnings_check:
            checks.append(self._check_earnings_cooldown)

        for check in checks:
            check()

        log.info(
            f"[APPROVED] Order APPROVED: {symbol} | value=${order_value:,.0f} "
            f"| delta_impact={delta_impact:+.2f} | option={is_option}"
        )

    # ─────────────────────────────────────────
    # Portfolio Summary
    # ─────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "equity": self.equity,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / RISK.starting_balance * 100,
            "portfolio_delta": self.portfolio_delta,
            "options_exposure": self.total_options_exposure,
            "options_exposure_pct": self.total_options_exposure / self.equity * 100,
            "vix": self.current_vix,
            "daily_loss_limit_remaining": (
                -RISK.daily_loss_limit_pct * RISK.starting_balance - self.daily_pnl
            ),
        }
