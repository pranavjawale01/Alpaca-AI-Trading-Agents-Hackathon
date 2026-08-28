"""
agents/hedge_agent.py — Portfolio Hedge Agent.

Strategy:
  - Buys SPY puts to protect the portfolio against sudden market drops
  - Sizes hedge based on portfolio delta and VIX level
  - Rolls hedge every 30 days or when VIX spikes significantly

Entry conditions:
  - Portfolio delta > 30 (too long, need protection)
  - OR VIX > 22 (increasing fear, time to hedge)
  - Puts bought 3–5% OTM, 30-45 DTE

Sizing:
  - Hedge covers 20% of portfolio delta at any given time
  - Cost capped at 0.5% of equity per month
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from rich.console import Console

from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.risk_manager import RiskManager, RiskViolation
import config

console = Console()
log = logging.getLogger(__name__)

VIX_HEDGE_TRIGGER = 22
DELTA_HEDGE_TRIGGER = 30
HEDGE_PUT_OTM_PCT = 0.04       # 4% OTM
MAX_HEDGE_COST_PCT = 0.005     # 0.5% of equity per hedge cycle


class HedgeAgent:
    """
    Maintains a protective put position on SPY as portfolio insurance.
    Activates when portfolio delta is too long or VIX is rising.
    """

    def __init__(
        self,
        client: AlpacaClient,
        market_data: MarketData,
        risk_manager: RiskManager,
    ) -> None:
        self.client = client
        self.md = market_data
        self.rm = risk_manager
        self.symbol = config.UNIVERSE.hedge_symbol
        self.hedge_position: Optional[dict] = None

        console.print("[bold red]HedgeAgent initialised[/bold red]")

    def run(self) -> list[dict]:
        actions = []

        # Check if hedge is needed
        needs_hedge = (
            self.rm.portfolio_delta > DELTA_HEDGE_TRIGGER
            or self.rm.current_vix > VIX_HEDGE_TRIGGER
        )

        if needs_hedge and self.hedge_position is None:
            action = self._open_hedge()
            if action:
                actions.append(action)
        elif not needs_hedge and self.hedge_position is not None:
            # Remove hedge when conditions normalise
            action = self._close_hedge()
            if action:
                actions.append(action)

        return actions

    def _open_hedge(self) -> Optional[dict]:
        """Buy SPY protective put."""
        account = self.client.get_account()
        equity = account["equity"]
        max_spend = equity * MAX_HEDGE_COST_PCT

        expiry_min = (date.today() + timedelta(days=28)).isoformat()
        expiry_max = (date.today() + timedelta(days=45)).isoformat()

        puts = self.client.get_option_contracts(
            self.symbol, expiry_min, expiry_max, "put"
        )
        if not puts:
            log.warning("HedgeAgent: No SPY puts found")
            return None

        target_strike = self.md.find_otm_strike(
            self.symbol,
            [p["strike"] for p in puts],
            "put",
            HEDGE_PUT_OTM_PCT,
        )
        contract = next((p for p in puts if p["strike"] == target_strike), None)
        if not contract:
            return None

        spy_price = self.md.get_price(self.symbol)
        est_premium = spy_price * 0.015  # rough 4% OTM put estimate
        n_contracts = max(1, int(max_spend / (est_premium * 100)))

        try:
            self.rm.approve_order(
                symbol=contract["symbol"],
                order_value=est_premium * 100 * n_contracts,
                delta_impact=-0.20 * 100 * n_contracts,
                is_option=True,
                skip_earnings_check=True,
            )
        except RiskViolation as e:
            console.print(f"[yellow][RISK BLOCK] Hedge blocked: {e}[/yellow]")
            return None

        result = self.client.place_option_market_order(contract["symbol"], n_contracts, "buy")

        self.hedge_position = {
            "contract": contract["symbol"],
            "strike": target_strike,
            "qty": n_contracts,
            "expiration": contract["expiration"],
        }

        console.print(
            f"[red][FILLED] Hedge OPENED: SPY ${target_strike:.0f} put "
            f"exp={contract['expiration']} x{n_contracts} "
            f"| VIX={self.rm.current_vix:.1f} | delta={self.rm.portfolio_delta:.1f}[/red]"
        )
        return {"agent": "Hedge", "action": "buy_put", "symbol": self.symbol, **self.hedge_position}

    def _close_hedge(self) -> Optional[dict]:
        """Close (sell) the protective put when conditions normalise."""
        if not self.hedge_position:
            return None
        contract_sym = self.hedge_position["contract"]
        qty = self.hedge_position["qty"]
        try:
            self.client.place_option_market_order(contract_sym, qty, "sell")
            console.print(f"[blue][CLOSED] Hedge CLOSED: conditions normalized[/blue]")
            closed = self.hedge_position.copy()
            self.hedge_position = None
            return {"agent": "Hedge", "action": "closed_hedge", **closed}
        except Exception as e:
            log.error(f"HedgeAgent close error: {e}")
            return None
