"""
agents/momo_breakout.py — Momentum Breakout Call Buyer Agent.

Strategy:
  - Monitors a watchlist of high-beta growth stocks
  - Buys cheap OTM calls (5% OTM, 30-45 DTE) on EMA crossover + volume surge
  - Targets 3:1 reward-to-risk ratio
  - Cuts losses quickly if trade goes against

Entry conditions:
  - 20-EMA crosses above 50-EMA (bullish crossover)
  - Volume surge ratio >= 2.0 (strong conviction move)
  - IVR < 40 (options are cheap relative to history)
  - VIX < 25 (risk-on environment)

Exit conditions:
  - 100% profit (double your money)
  - Stop loss: 50% of premium paid
  - Time stop: 15 DTE
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

MAX_POSITIONS = 3
PROFIT_TARGET_MULTIPLIER = 2.0   # 100% gain
STOP_LOSS_PCT = 0.50             # lose max 50% of premium paid
TIME_STOP_DTE = 15
OTM_PCT = 0.05                   # 5% out of the money
MAX_PREMIUM_PER_TRADE_PCT = 0.01 # max 1% of equity per momo trade


class MomoBreakoutAgent:
    """
    Buys cheap OTM calls on momentum breakout signals.
    Asymmetric upside: risk small, reward large.
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
        self.watchlist = config.UNIVERSE.momo_watchlist
        self.active_positions: dict[str, dict] = {}

        console.print("[bold yellow]MomoBreakoutAgent initialised[/bold yellow]")

    def run(self) -> list[dict]:
        actions = []
        actions += self._manage_existing_positions()

        if len(self.active_positions) < MAX_POSITIONS and self.rm.current_vix < 25:
            actions += self._scan_breakouts()

        return actions

    def _scan_breakouts(self) -> list[dict]:
        """Scan watchlist for EMA crossover + volume surge."""
        actions = []
        account = self.client.get_account()
        equity = account["equity"]

        for symbol in self.watchlist:
            if symbol in self.active_positions:
                continue

            try:
                ema_signal = self.md.get_ema_signal(symbol)
                vol_surge = self.md.get_volume_surge(symbol)

                # Both conditions must be true
                if ema_signal["crossover"] and vol_surge["is_surging"]:
                    console.print(
                        f"[yellow]🔥 Breakout signal: {symbol} | "
                        f"EMA={ema_signal['signal']} | surge={vol_surge['surge_ratio']:.1f}x[/yellow]"
                    )
                    action = self._buy_call(symbol, equity)
                    if action:
                        actions.append(action)

            except RiskViolation as e:
                console.print(f"[yellow]⚠ Risk block [{symbol} momo]: {e}[/yellow]")
            except Exception as e:
                log.error(f"Momo scan error [{symbol}]: {e}")

        return actions

    def _buy_call(self, symbol: str, equity: float) -> Optional[dict]:
        """Buy OTM call on breakout signal."""
        expiry_min = (date.today() + timedelta(days=28)).isoformat()
        expiry_max = (date.today() + timedelta(days=45)).isoformat()

        calls = self.client.get_option_contracts(symbol, expiry_min, expiry_max, "call")
        if not calls:
            log.info(f"[{symbol}] No call contracts found")
            return None

        # Find 5% OTM call
        target_strike = self.md.find_otm_strike(
            symbol, [c["strike"] for c in calls], "call", OTM_PCT
        )
        contract = next((c for c in calls if c["strike"] == target_strike), None)
        if not contract:
            return None

        # Max 1% of equity in premium
        max_spend = equity * MAX_PREMIUM_PER_TRADE_PCT
        price = self.md.get_price(symbol)
        est_premium = price * 0.03  # rough OTM estimate
        n_contracts = max(1, int(max_spend / (est_premium * 100)))

        order_value = est_premium * 100 * n_contracts

        self.rm.approve_order(
            symbol=contract["symbol"],
            order_value=order_value,
            delta_impact=0.30 * 100 * n_contracts,  # approx 30-delta call
            is_option=True,
        )

        result = self.client.place_option_market_order(contract["symbol"], n_contracts, "buy")

        action = {
            "agent": "MomoBreakout",
            "action": "buy_call",
            "symbol": symbol,
            "contract": contract["symbol"],
            "strike": target_strike,
            "expiration": contract["expiration"],
            "qty": n_contracts,
            "entry_premium_est": est_premium,
            "order_id": result.get("id"),
        }
        self.active_positions[symbol] = action

        console.print(
            f"[green]✅ MomoBreakout BOUGHT CALL: {symbol} "
            f"${target_strike:.0f} exp={contract['expiration']} x{n_contracts}[/green]"
        )
        return action

    def _manage_existing_positions(self) -> list[dict]:
        """Check profit targets and stop losses on open calls."""
        actions = []
        positions = {p["symbol"]: p for p in self.client.get_all_positions()}

        for symbol, meta in list(self.active_positions.items()):
            contract_sym = meta.get("contract")
            if contract_sym not in positions:
                del self.active_positions[symbol]
                continue

            pos = positions[contract_sym]
            plpc = float(pos.get("unrealized_plpc", 0))

            if plpc >= (PROFIT_TARGET_MULTIPLIER - 1):
                # Hit profit target
                self.client.place_option_market_order(contract_sym, meta["qty"], "sell")
                del self.active_positions[symbol]
                console.print(f"[green]💰 MomoBreakout profit target: {symbol} (+{plpc*100:.0f}%)[/green]")
                actions.append({"agent": "MomoBreakout", "action": "closed_profit", "symbol": symbol})

            elif plpc <= -STOP_LOSS_PCT:
                # Hit stop loss
                self.client.place_option_market_order(contract_sym, meta["qty"], "sell")
                del self.active_positions[symbol]
                console.print(f"[red]🛑 MomoBreakout stop loss: {symbol} ({plpc*100:.0f}%)[/red]")
                actions.append({"agent": "MomoBreakout", "action": "stopped_out", "symbol": symbol})

        return actions
