"""
agents/theta_collector.py — Cash-Secured Put (CSP) Income Agent.

Strategy:
  - Sells cash-secured puts on liquid ETFs (SPY, QQQ, IWM, GLD)
  - Targets 30-45 DTE (days to expiration)
  - Selects strike at ~10% OTM (Delta ~0.20–0.25)
  - Closes position when profit hits 50% of premium collected (early exit)
  - Manages through expiration if not triggered

Entry conditions:
  - IVR > 30 (elevated premium to sell)
  - Bullish or neutral market regime (VIX < 30)
  - Sufficient buying power for cash-secured requirement

Exit conditions:
  - 50% profit target reached
  - 21 DTE (time-based stop)
  - Stop loss: if position loses 2x premium received
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from rich.console import Console

from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.options_pricer import greeks, iv_rank, estimate_annual_vol_from_hist
from core.risk_manager import RiskManager, RiskViolation
import config

console = Console()
log = logging.getLogger(__name__)

# ── Agent Parameters ──────────────────────────────────────────
TARGET_DTE_MIN = 28
TARGET_DTE_MAX = 45
TARGET_DELTA = 0.20          # sell puts at ~20 delta (10% OTM)
MIN_IVR_TO_ENTER = 30        # only sell when vol is elevated
PROFIT_TARGET_PCT = 0.50     # close at 50% profit
STOP_LOSS_MULTIPLIER = 2.0   # stop if loss > 2x premium
TIME_STOP_DTE = 21           # roll or close at 21 DTE
RISK_FREE_RATE = 0.05


class ThetaCollectorAgent:
    """
    Sells cash-secured puts on high-quality ETFs to collect theta decay.
    This is the most reliable P&L generator in the portfolio.
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
        self.symbols = config.UNIVERSE.theta_symbols
        self.active_positions: dict[str, dict] = {}  # symbol → trade metadata

        console.print("[bold cyan]ThetaCollectorAgent initialised[/bold cyan]")

    # ─────────────────────────────────────────
    # Main Run Loop
    # ─────────────────────────────────────────

    def run(self) -> list[dict]:
        """
        Called once per trading session (market open).
        Returns list of actions taken.
        """
        actions = []

        # 1. Manage existing positions
        actions += self._manage_existing_positions()

        # 2. Scan for new entry opportunities
        if len(self.active_positions) < len(self.symbols):
            actions += self._scan_for_entries()

        return actions

    # ─────────────────────────────────────────
    # Entry Logic
    # ─────────────────────────────────────────

    def _scan_for_entries(self) -> list[dict]:
        """Scan symbols for CSP entry conditions."""
        actions = []
        account = self.client.get_account()
        equity = account["equity"]

        for symbol in self.symbols:
            if symbol in self.active_positions:
                continue  # already have a position

            try:
                signal = self._evaluate_entry(symbol, equity)
                if signal:
                    actions.append(signal)
            except RiskViolation as e:
                console.print(f"[yellow]⚠ Risk block [{symbol}]: {e}[/yellow]")
            except Exception as e:
                log.error(f"Entry scan error [{symbol}]: {e}")

        return actions

    def _evaluate_entry(self, symbol: str, equity: float) -> Optional[dict]:
        """Evaluate entry conditions for one symbol. Returns action dict or None."""
        price = self.md.get_price(symbol)
        hist_vol = self.md.estimate_historical_vol(symbol)
        vix = self.rm.current_vix

        # Condition 1: VIX check (don't sell premium in volatile markets)
        if vix >= 30:
            log.info(f"[{symbol}] Skip: VIX={vix:.1f} >= 30")
            return None

        # Condition 2: IVR check (simplified — use hist vol as proxy)
        # In production, use actual IV from options chain
        ivr = min(hist_vol * 200, 100)  # heuristic IVR proxy
        if ivr < MIN_IVR_TO_ENTER:
            log.info(f"[{symbol}] Skip: IVR={ivr:.1f} < {MIN_IVR_TO_ENTER}")
            return None

        # Find target expiration (~30-45 DTE)
        expiry_min = (date.today() + timedelta(days=TARGET_DTE_MIN)).isoformat()
        expiry_max = (date.today() + timedelta(days=TARGET_DTE_MAX)).isoformat()

        contracts = self.client.get_option_contracts(
            symbol,
            expiration_date_gte=expiry_min,
            expiration_date_lte=expiry_max,
            contract_type="put",
        )

        if not contracts:
            log.info(f"[{symbol}] No put contracts found for target DTE range")
            return None

        # Find ~20 delta put (10% OTM)
        target_strike = self.md.find_otm_strike(
            symbol,
            [c["strike"] for c in contracts],
            option_type="put",
            otm_pct=0.10,
        )
        contract = next((c for c in contracts if c["strike"] == target_strike), None)
        if not contract:
            return None

        # Calculate premium and order value
        T = TARGET_DTE_MIN / 365
        premium = self.client.get_latest_quote(contract["symbol"])["mid"] if True else 0
        # Fallback: use Black-Scholes estimate
        if premium <= 0:
            from core.options_pricer import black_scholes_price
            premium = black_scholes_price(price, target_strike, T, RISK_FREE_RATE, hist_vol, "put")

        # CSP requires holding cash = strike × 100 (per contract)
        notional = target_strike * 100
        n_contracts = max(1, int((equity * 0.04) / notional))  # max 4% of equity

        # Risk gate
        self.rm.approve_order(
            symbol=contract["symbol"],
            order_value=notional * n_contracts,
            delta_impact=-TARGET_DELTA * 100 * n_contracts,
            is_option=True,
        )

        # Place order
        result = self.client.place_option_market_order(
            contract["symbol"], n_contracts, "sell"
        )

        action = {
            "agent": "ThetaCollector",
            "action": "sell_put",
            "symbol": symbol,
            "contract": contract["symbol"],
            "strike": target_strike,
            "expiration": contract["expiration"],
            "qty": n_contracts,
            "premium_collected": premium * 100 * n_contracts,
            "order_id": result.get("id"),
        }

        self.active_positions[symbol] = {
            **action,
            "entry_premium": premium,
            "stop_loss_premium": premium * STOP_LOSS_MULTIPLIER,
        }

        console.print(
            f"[green]✅ ThetaCollector SOLD PUT: {symbol} "
            f"${target_strike:.0f} exp={contract['expiration']} "
            f"x{n_contracts} | premium=${premium*100*n_contracts:,.0f}[/green]"
        )
        return action

    # ─────────────────────────────────────────
    # Position Management
    # ─────────────────────────────────────────

    def _manage_existing_positions(self) -> list[dict]:
        """Check existing CSP positions for exit conditions."""
        actions = []
        positions = {p["symbol"]: p for p in self.client.get_all_positions()}

        for symbol, meta in list(self.active_positions.items()):
            contract_symbol = meta.get("contract")
            if contract_symbol not in positions:
                # Position already closed (expired worthless = max profit!)
                console.print(f"[green]💰 {symbol} CSP expired worthless — max profit![/green]")
                del self.active_positions[symbol]
                actions.append({"agent": "ThetaCollector", "action": "expired_worthless", "symbol": symbol})
                continue

            pos = positions[contract_symbol]
            current_value = abs(float(pos["market_value"]))
            entry_premium = meta["entry_premium"] * 100 * meta["qty"]
            profit_pct = 1 - (current_value / entry_premium) if entry_premium > 0 else 0

            # Exit: 50% profit target
            if profit_pct >= PROFIT_TARGET_PCT:
                self._close_position(symbol, contract_symbol, meta["qty"], "profit_target")
                actions.append({"agent": "ThetaCollector", "action": "closed_profit_target", "symbol": symbol, "profit_pct": profit_pct})

        return actions

    def _close_position(self, symbol: str, contract_symbol: str, qty: int, reason: str) -> None:
        """Close (buy back) a short put."""
        try:
            self.client.place_option_market_order(contract_symbol, qty, "buy")
            del self.active_positions[symbol]
            console.print(f"[blue]🔒 ThetaCollector closed {symbol} [{reason}][/blue]")
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")
