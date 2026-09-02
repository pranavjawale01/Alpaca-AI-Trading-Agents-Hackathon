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
    from core.opportunity_scorer import OpportunityScorer

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
        opportunity_scorer: Optional["OpportunityScorer"] = None,
    ) -> None:
        self.client = client
        self.md = market_data
        self.rm = risk_manager
        self.council = council
        self.kelly = kelly_sizer
        self.executor = smart_executor
        self.journal = journal
        self.opportunity_scorer = opportunity_scorer
        self.watchlist = config.UNIVERSE.momo_watchlist
        self.active_positions: dict[str, dict] = {}
        # {symbol: peak_plpc} — tracks all-time high unrealised P&L per position
        self._peak_plpc: dict[str, float] = {}

        console.print("[bold yellow]MomoBreakoutAgent initialised[/bold yellow]")

    def run(self, regime: str = "neutral") -> list[dict]:
        actions = []
        actions += self._manage_existing_positions()

        # In risk_on or neutral (VIX < 25), scans for bullish calls and bearish puts
        # In risk_off (VIX < 35 kill switch), scans specifically for bearish breakdown puts (short alpha)
        if len(self.active_positions) < MAX_POSITIONS and self.rm.current_vix < config.RISK.vix_kill_switch:
            actions += self._scan_breakouts(regime=regime)

        return actions

    def _scan_breakouts(self, regime: str = "neutral") -> list[dict]:
        """Scan watchlist for EMA crossover + volume surge (Calls on bullish, Puts on bearish)."""
        actions = []
        account = self.client.get_account()
        equity = account["equity"]

        for symbol in self.watchlist:
            if symbol in self.active_positions:
                continue

            size_multiplier = 1.0

            try:
                # ── 1. Mathematical Breakout Signals (Zero LLM calls) ────────
                ema_signal = self.md.get_ema_signal(symbol)
                vol_surge = self.md.get_volume_surge(symbol)

                # Both mathematical conditions must be true: crossover + volume surge
                if not (ema_signal.get("crossover") and vol_surge.get("is_surging")):
                    continue

                # Determine direction: Bullish (Call) or Bearish (Put)
                is_bullish = (
                    ema_signal.get("crossover_bullish")
                    or (ema_signal.get("crossover") and ema_signal.get("signal") == "bullish")
                )
                is_bearish = (
                    ema_signal.get("crossover_bearish")
                    or (ema_signal.get("crossover") and ema_signal.get("signal") == "bearish")
                )

                if is_bullish:
                    # Bullish calls only in calm/neutral markets (VIX < 25)
                    if self.rm.current_vix >= 25:
                        log.info(f"[{symbol}] Bullish breakout skipped: VIX={self.rm.current_vix:.1f} >= 25")
                        continue
                    direction = "bullish"
                    option_type = "call"
                    strategy_name = "momentum_call"
                elif is_bearish:
                    # Bearish breakdowns (puts) allowed across all regimes including risk_off
                    direction = "bearish"
                    option_type = "put"
                    strategy_name = "momentum_put"
                else:
                    continue

                console.print(
                    f"[yellow][SIGNAL] Momentum {direction.upper()} detected: {symbol} | "
                    f"EMA={ema_signal.get('signal')} | surge={vol_surge.get('surge_ratio', 1.0):.1f}x[/yellow]"
                )

                # ── 2. Contract & Option Pricing Feasibility ──────────────────
                price = self.md.get_price(symbol)
                hist_vol = self.md.estimate_historical_vol(symbol)

                expiry_min = (date.today() + timedelta(days=28)).isoformat()
                expiry_max = (date.today() + timedelta(days=45)).isoformat()

                contracts = self.client.get_option_contracts(symbol, expiry_min, expiry_max, option_type)
                if not contracts:
                    log.info(f"[{symbol}] No {option_type} contracts found in DTE range")
                    continue

                target_strike = self.md.find_otm_strike(
                    symbol, [c["strike"] for c in contracts], option_type, OTM_PCT
                )
                contract = next((c for c in contracts if c["strike"] == target_strike), None)
                if not contract:
                    log.info(f"[{symbol}] Target {option_type} strike ${target_strike} not available")
                    continue

                est_premium = price * 0.03
                if est_premium <= 0.01:
                    log.info(f"[{symbol}] Premium too low (${est_premium:.2f}), skipping")
                    continue

                # ── 3. Preliminary Kelly Sizing Check ────────────────────────
                if self.kelly is not None:
                    prelim_contracts = self.kelly.get_contract_count(
                        strategy="momo",
                        equity=equity,
                        premium_per_contract=est_premium * 100,
                        override_max_pct=MAX_PREMIUM_PER_TRADE_PCT,
                        size_multiplier=1.0,
                        greedy_multiplier=1.0,
                    )
                else:
                    prelim_contracts = max(1, int((equity * MAX_PREMIUM_PER_TRADE_PCT) / (est_premium * 100)))

                if prelim_contracts <= 0:
                    log.info(f"[{symbol}] Kelly preliminary sizing = 0 contracts, skipping")
                    continue

                # ── 4. Opportunity Scorer — greedy multiplier ─────────────────
                greedy_multiplier = 1.0
                if self.opportunity_scorer is not None:
                    opp_ctx = {
                        "ivr": min(hist_vol * 200, 100),
                        "vix": self.rm.current_vix,
                        "ema_signal": ema_signal.get("signal", "neutral"),
                        "ema_crossover": ema_signal.get("crossover", False),
                        "volume_surge_ratio": vol_surge.get("surge_ratio", 1.0),
                        "strategy": "momo",
                    }
                    greedy_multiplier = self.opportunity_scorer.score(
                        opp_ctx,
                        open_positions=list(self.active_positions.keys()),
                        session_pnl=self.rm.daily_pnl,
                    )

                # ── 5. LLM Council Vote Gate (Only runs after ALL math passes) ──
                size_multiplier = 1.0
                consensus = None
                if self.council is not None:
                    ctx = SignalEnhancer.build_momo_context(
                        symbol=symbol,
                        ema_signal=ema_signal,
                        vol_surge=vol_surge,
                        vix=self.rm.current_vix,
                        price=price,
                        hist_vol=hist_vol,
                        direction=direction,
                    )
                    consensus = self.council.vote(symbol, ctx, strategy=strategy_name)
                    console.print(consensus.summary())
                    if consensus.conviction_tier == "veto":
                        console.print(
                            f"[yellow][COUNCIL VETO] {symbol} momo {direction}: "
                            f"score={consensus.net_score:+.3f} | tier=VETO[/yellow]"
                        )
                        log.info(
                            f"[{symbol}] Council vetoed momo {direction} entry: "
                            f"score={consensus.net_score:+.3f} | tier=veto"
                        )
                        continue
                    size_multiplier = consensus.size_multiplier
                    console.print(
                        f"[green][COUNCIL {consensus.conviction_tier.upper()}] {symbol} momo {direction}: "
                        f"size_mult={size_multiplier:.2f}[/green]"
                    )

                # ── 6. Final Kelly Sizing & Execution ─────────────────────────
                votes_list = consensus.votes if (self.council and consensus) else []
                action = self._buy_option(
                    symbol=symbol,
                    equity=equity,
                    contract=contract,
                    target_strike=target_strike,
                    est_premium=est_premium,
                    option_type=option_type,
                    size_multiplier=size_multiplier,
                    greedy_multiplier=greedy_multiplier,
                    votes=votes_list,
                )
                if action:
                    actions.append(action)

            except RiskViolation as e:
                console.print(f"[yellow][RISK BLOCK] [{symbol} momo]: {e}[/yellow]")
            except Exception as e:
                log.error(f"Momo scan error [{symbol}]: {e}")

        return actions

    def _buy_option(
        self,
        symbol: str,
        equity: float,
        contract: dict,
        target_strike: float,
        est_premium: float,
        option_type: str = "call",
        size_multiplier: float = 1.0,
        greedy_multiplier: float = 1.0,
        votes: Optional[list] = None,
    ) -> Optional[dict]:
        """Buy OTM option (Call or Put) using Kelly-sized position and mid-price limit order."""
        # ── Kelly Criterion sizing ────────────────────────────────────
        if self.kelly is not None:
            n_contracts = self.kelly.get_contract_count(
                strategy="momo",
                equity=equity,
                premium_per_contract=est_premium * 100,
                override_max_pct=MAX_PREMIUM_PER_TRADE_PCT,
                size_multiplier=size_multiplier,
                greedy_multiplier=greedy_multiplier,
            )
        else:
            n_contracts = max(1, int((equity * MAX_PREMIUM_PER_TRADE_PCT) / (est_premium * 100)))

        if n_contracts <= 0:
            return None

        order_value = est_premium * 100 * n_contracts

        # Long call has positive delta (~+0.30); Long put has negative delta (~-0.30)
        delta_impact = (0.30 if option_type == "call" else -0.30) * 100 * n_contracts

        self.rm.approve_order(
            symbol=contract["symbol"],
            order_value=order_value,
            delta_impact=delta_impact,
            is_option=True,
        )

        # ── Smart limit order execution ────────────────────────────
        if self.executor is not None:
            result = self.executor.execute_option_order(contract["symbol"], n_contracts, "buy")
        else:
            result = self.client.place_option_market_order(contract["symbol"], n_contracts, "buy")
        # ─────────────────────────────────────────────────────────

        conviction_tier = "bypass"
        if self.council is not None:
            tier_by_mult = {1.0: "strong", 0.70: "moderate", 0.40: "pilot", 0.0: "veto"}
            conviction_tier = tier_by_mult.get(round(size_multiplier, 2), "strong")

        action_name = f"buy_{option_type}"
        action = {
            "agent": "MomoBreakout",
            "action": action_name,
            "option_type": option_type,
            "direction": "bullish" if option_type == "call" else "bearish",
            "symbol": symbol,
            "contract": contract["symbol"],
            "strike": target_strike,
            "expiration": contract["expiration"],
            "qty": n_contracts,
            "entry_premium_est": est_premium,
            "order_id": result.get("id"),
            "conviction_tier": conviction_tier,
            "size_multiplier": size_multiplier,
            "greedy_multiplier": greedy_multiplier,
        }
        self.active_positions[symbol] = {
            **action,
            "entry_premium": est_premium,
        }
        self._peak_plpc[symbol] = 0.0  # initialise trailing stop tracker

        # ── Journal & Credibility: log entry ───────────────────────
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
            if self.council is not None and getattr(self.council, "_credibility_tracker", None):
                target_votes = votes if votes is not None else getattr(self.council, "_last_votes", [])
                if target_votes:
                    self.council._credibility_tracker.record_votes(trade_id, target_votes)
        # ─────────────────────────────────────────────────────────

        console.print(
            f"[green][FILLED] MomoBreakout BOUGHT {option_type.upper()}: {symbol} "
            f"${target_strike:.0f} exp={contract['expiration']} x{n_contracts} "
            f"| est premium=${est_premium:.2f} | tier={conviction_tier.upper()} "
            f"| greed={greedy_multiplier:.2f} | Kelly-sized[/green]"
        )
        return action

    def _buy_call(self, *args, **kwargs) -> Optional[dict]:
        """Backward compatibility alias for _buy_option with option_type='call'."""
        kwargs["option_type"] = "call"
        return self._buy_option(*args, **kwargs)

    def _buy_put(self, *args, **kwargs) -> Optional[dict]:
        """Backward compatibility alias for _buy_option with option_type='put'."""
        kwargs["option_type"] = "put"
        return self._buy_option(*args, **kwargs)

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

                # ── Journal & Credibility: log exit ────────────────────
                if self.journal is not None and "journal_id" in meta:
                    exit_price = meta["entry_premium"] * (1 + plpc)
                    self.journal.log_exit(meta["journal_id"], exit_price, reason)
                    if self.council is not None and getattr(self.council, "_credibility_tracker", None):
                        self.council._credibility_tracker.update_credibility(meta["journal_id"], plpc > 0)
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
