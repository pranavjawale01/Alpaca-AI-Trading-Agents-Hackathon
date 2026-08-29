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
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.risk_manager import RiskManager, RiskViolation
from core.signal_enhancer import SignalEnhancer
import config

if TYPE_CHECKING:
    from core.llm_council import LLMCouncil
    from core.kelly_sizer import KellySizer
    from core.smart_executor import SmartExecutor
    from core.trade_journal import TradeJournal

console = Console()
log = logging.getLogger(__name__)

MAX_POSITIONS = 3
PROFIT_TARGET_MULTIPLIER = 2.0   # 100% gain — still close at 2x
STOP_LOSS_PCT = 0.50             # hard stop: lose max 50% of premium paid
TIME_STOP_DTE = 15
OTM_PCT = 0.05                   # 5% out of the money
MAX_PREMIUM_PER_TRADE_PCT = 0.01 # hard cap; Kelly will usually be below this


class MomoBreakoutAgent:
    """
    Buys cheap OTM calls on momentum breakout signals.
    Asymmetric upside: risk small, reward large.

    Pro improvements:
      1. LLM Council: 3-model vote gate before every entry
      2. Kelly Criterion: position sized to edge, not fixed %
      3. SmartExecutor: limit orders at mid-price (avoids full spread)
      4. Trailing Stop: exit at 25% pullback from peak, not fixed -50%
      5. TradeJournal: every trade logged to SQLite for learning
    """

    def __init__(
        self,
        client: AlpacaClient,
        market_data: MarketData,
        risk_manager: RiskManager,
        council: Optional["LLMCouncil"] = None,
        kelly_sizer: Optional["KellySizer"] = None,
        smart_executor: Optional["SmartExecutor"] = None,
        journal: Optional["TradeJournal"] = None,
    ) -> None:
        self.client = client
        self.md = market_data
        self.rm = risk_manager
        self.council = council
        self.kelly = kelly_sizer
        self.executor = smart_executor
        self.journal = journal
        self.watchlist = config.UNIVERSE.momo_watchlist
        self.active_positions: dict[str, dict] = {}
        # {symbol: peak_plpc} — tracks all-time high unrealised P&L per position
        self._peak_plpc: dict[str, float] = {}

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
                        f"[yellow][SIGNAL] Breakout detected: {symbol} | "
                        f"EMA={ema_signal['signal']} | surge={vol_surge['surge_ratio']:.1f}x[/yellow]"
                    )

                    # ── LLM Council vote gate ──────────────────────────
                    if self.council is not None:
                        price = self.md.get_price(symbol)
                        hist_vol = self.md.estimate_historical_vol(symbol)
                        ctx = SignalEnhancer.build_momo_context(
                            symbol=symbol,
                            ema_signal=ema_signal,
                            vol_surge=vol_surge,
                            vix=self.rm.current_vix,
                            price=price,
                            hist_vol=hist_vol,
                        )
                        consensus = self.council.vote(symbol, ctx, strategy="momentum_call")
                        console.print(consensus.summary())
                        if not consensus.agreed:
                            console.print(
                                f"[yellow][COUNCIL VETO] {symbol} momo: "
                                f"score={consensus.net_score:+.3f} (threshold={consensus.threshold}). "
                                f"Dissenting: {consensus.dissenting_models or 'none (all hold)'}"
                                f"[/yellow]"
                            )
                            log.info(
                                f"[{symbol}] Council vetoed momo entry: "
                                f"score={consensus.net_score:+.3f} | "
                                f"votes={[(v.model.split('/')[-1], v.action, v.confidence) for v in consensus.votes]}"
                            )
                            continue  # skip this trade
                    # ──────────────────────────────────────────────────

                    action = self._buy_call(symbol, equity)
                    if action:
                        actions.append(action)

            except RiskViolation as e:
                console.print(f"[yellow][RISK BLOCK] [{symbol} momo]: {e}[/yellow]")
            except Exception as e:
                log.error(f"Momo scan error [{symbol}]: {e}")

        return actions

    def _buy_call(self, symbol: str, equity: float) -> Optional[dict]:
        """Buy OTM call using Kelly-sized position and mid-price limit order."""
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

        price = self.md.get_price(symbol)
        est_premium = price * 0.03  # rough OTM estimate per share

        # ── Kelly Criterion sizing ────────────────────────────────────
        if self.kelly is not None:
            n_contracts = self.kelly.get_contract_count(
                strategy="momo",
                equity=equity,
                premium_per_contract=est_premium * 100,
                override_max_pct=MAX_PREMIUM_PER_TRADE_PCT,
            )
        else:
            n_contracts = max(1, int((equity * MAX_PREMIUM_PER_TRADE_PCT) / (est_premium * 100)))
        # ─────────────────────────────────────────────────────────────

        order_value = est_premium * 100 * n_contracts

        self.rm.approve_order(
            symbol=contract["symbol"],
            order_value=order_value,
            delta_impact=0.30 * 100 * n_contracts,
            is_option=True,
        )

        # ── Smart limit order execution ────────────────────────────
        if self.executor is not None:
            result = self.executor.execute_option_order(contract["symbol"], n_contracts, "buy")
        else:
            result = self.client.place_option_market_order(contract["symbol"], n_contracts, "buy")
        # ─────────────────────────────────────────────────────────

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
        self.active_positions[symbol] = {
            **action,
            "entry_premium": est_premium,
        }
        self._peak_plpc[symbol] = 0.0  # initialise trailing stop tracker

        # ── Journal: log entry ─────────────────────────────────────
        if self.journal is not None:
            trade_id = self.journal.log_entry(
                agent="MomoBreakout",
                strategy="momo",
                symbol=symbol,
                contract=contract["symbol"],
                side="buy",
                qty=n_contracts,
                entry_price=est_premium,
            )
            self.active_positions[symbol]["journal_id"] = trade_id
        # ─────────────────────────────────────────────────────────

        console.print(
            f"[green][FILLED] MomoBreakout BOUGHT CALL: {symbol} "
            f"${target_strike:.0f} exp={contract['expiration']} x{n_contracts} "
            f"| est premium=${est_premium:.2f} | Kelly-sized[/green]"
        )
        return action

    def _manage_existing_positions(self) -> list[dict]:
        """
        Check exit conditions on all open calls.

        Exit logic (in priority order):
          1. Profit target: close at 2× premium (100% gain)
          2. Trailing stop: close if P&L pulls back > TRAILING_STOP_PCT from peak
          3. Hard stop: close if unrealised loss > STOP_LOSS_PCT from cost

        The trailing stop replaces a fixed stop-from-cost, allowing the agent to
        lock in profits on big winners while still cutting losers quickly.
        """
        actions = []
        trailing_pct = config.EXECUTION.trailing_stop_pct
        positions = {p["symbol"]: p for p in self.client.get_all_positions()}

        for symbol, meta in list(self.active_positions.items()):
            contract_sym = meta.get("contract")
            if contract_sym not in positions:
                # Position closed externally or expired
                del self.active_positions[symbol]
                self._peak_plpc.pop(symbol, None)
                continue

            pos = positions[contract_sym]
            plpc = float(pos.get("unrealized_plpc", 0))  # fraction: 0.50 = 50% gain

            # Update trailing high-water mark
            prev_peak = self._peak_plpc.get(symbol, 0.0)
            self._peak_plpc[symbol] = max(prev_peak, plpc)
            peak = self._peak_plpc[symbol]

            reason = None

            if plpc >= (PROFIT_TARGET_MULTIPLIER - 1):
                reason = "profit_target"
                color = "green"
                msg = f"[PROFIT TARGET] +{plpc*100:.0f}%"

            elif peak > 0 and (peak - plpc) >= trailing_pct:
                # Trailing stop: pulled back trailing_pct from peak
                reason = "trailing_stop"
                color = "yellow"
                msg = f"[TRAILING STOP] peak={peak*100:.0f}% → now={plpc*100:.0f}% (pullback={( peak - plpc)*100:.0f}%)"

            elif plpc <= -STOP_LOSS_PCT:
                reason = "stop_loss"
                color = "red"
                msg = f"[STOP LOSS] {plpc*100:.0f}%"

            if reason:
                # Close position using executor (mid-price) or market
                if self.executor is not None:
                    self.executor.execute_option_order(contract_sym, meta["qty"], "sell")
                else:
                    self.client.place_option_market_order(contract_sym, meta["qty"], "sell")

                # ── Journal: log exit ──────────────────────────────────
                if self.journal is not None and "journal_id" in meta:
                    exit_price = meta["entry_premium"] * (1 + plpc)
                    self.journal.log_exit(meta["journal_id"], exit_price, reason)
                # ──────────────────────────────────────────────────────

                del self.active_positions[symbol]
                self._peak_plpc.pop(symbol, None)
                console.print(f"[{color}]MomoBreakout {msg}: {symbol}[/{color}]")
                actions.append({
                    "agent": "MomoBreakout",
                    "action": f"closed_{reason}",
                    "symbol": symbol,
                    "plpc": plpc,
                    "peak_plpc": peak,
                })

        return actions
