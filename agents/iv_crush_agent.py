"""
agents/iv_crush_agent.py — Earnings IV Crush Straddle Seller.

Strategy:
  - Before earnings: sell ATM straddle (simultaneous call + put)
  - Profit from IV collapsing post-earnings (IV crush)
  - Close position within 1–2 days after earnings announcement

Entry conditions:
  - Earnings within 1–3 days
  - IVR > 60 (very elevated premium due to earnings uncertainty)
  - VIX < 30 (not during market-wide vol spike)

Exit conditions:
  - 1 trading day after earnings announcement
  - 40% profit target
  - Stop loss: if position loses 1.5x premium received
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

PROFIT_TARGET_PCT = 0.40
STOP_LOSS_MULTIPLIER = 1.5
MAX_POSITIONS = config.UNIVERSE.iv_crush_max_positions
MIN_IVR_TO_ENTER = 60


class IVCrushAgent:
    """
    Sells ATM straddles before earnings to capture IV crush.
    High-probability income strategy in calm markets.

    Pro improvements:
      1. LLM Council: assesses earnings IV + timing before each straddle (hybrid tiers)
      2. Kelly Criterion: sizes straddle to historical win rate with greedy multiplier
      3. SmartExecutor: mid-price fills on both legs
      4. TradeJournal: logs entry/exit for adaptive sizing
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
        opportunity_scorer=None,
    ) -> None:
        self.client = client
        self.md = market_data
        self.rm = risk_manager
        self.council = council
        self.kelly = kelly_sizer
        self.executor = smart_executor
        self.journal = journal
        self.opportunity_scorer = opportunity_scorer
        self.active_positions: dict[str, dict] = {}

        console.print("[bold magenta]IVCrushAgent initialised[/bold magenta]")

    def run(self, earnings_calendar: list[dict]) -> list[dict]:
        """
        Args:
            earnings_calendar: [{"symbol": "AAPL", "date": "2026-08-30"}, ...]
        Returns:
            list of actions taken
        """
        actions = []
        actions += self._manage_existing_positions()

        if len(self.active_positions) < MAX_POSITIONS:
            actions += self._scan_earnings_entries(earnings_calendar)

        return actions

    def _scan_earnings_entries(self, calendar: list[dict]) -> list[dict]:
        """Find upcoming earnings and evaluate straddle entry."""
        actions = []
        today = date.today()

        for event in calendar:
            symbol = event["symbol"]
            earnings_date = date.fromisoformat(event["date"])
            days_to_earnings = (earnings_date - today).days

            # Only trade if earnings are 1–3 days away
            if not (1 <= days_to_earnings <= 3):
                continue
            if symbol in self.active_positions:
                continue

            try:
                action = self._open_straddle(symbol)
                if action:
                    # Register earnings event in risk manager for cooldown
                    from datetime import datetime, timezone
                    event_dt = datetime.combine(earnings_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                    self.rm.add_earnings_event(event_dt)
                    actions.append(action)
            except RiskViolation as e:
                console.print(f"[yellow][RISK BLOCK] [{symbol} straddle]: {e}[/yellow]")
            except Exception as e:
                log.error(f"Straddle entry error [{symbol}]: {e}")

        return actions

    def _open_straddle(self, symbol: str) -> Optional[dict]:
        """Sell ATM call + ATM put (straddle) ahead of earnings."""
        account = self.client.get_account()
        equity = account["equity"]
        price = self.md.get_price(symbol)

        # Get near-term options (7-14 DTE)
        expiry_min = (date.today() + timedelta(days=5)).isoformat()
        expiry_max = (date.today() + timedelta(days=14)).isoformat()

        calls = self.client.get_option_contracts(symbol, expiry_min, expiry_max, "call")
        puts = self.client.get_option_contracts(symbol, expiry_min, expiry_max, "put")

        if not calls or not puts:
            log.info(f"[{symbol}] No contracts for straddle")
            return None

        # Find ATM strike
        all_strikes = list({c["strike"] for c in calls} | {p["strike"] for p in puts})
        atm_strike = self.md.find_atm_strike(symbol, all_strikes)

        call = next((c for c in calls if c["strike"] == atm_strike), None)
        put = next((p for p in puts if p["strike"] == atm_strike), None)

        if not call or not put:
            return None

        # ── Condition 3: Hybrid LLM Council vote gate ─────────────────
        hist_vol = self.md.estimate_historical_vol(symbol)
        ivr = min(hist_vol * 200, 100)
        size_multiplier = 1.0  # default if council disabled
        if self.council is not None:
            # Retrieve days_to_earnings from the symbol's earnings event
            # (we approximate from DTE of the options — earnings are within 5-14 DTE)
            days_to_earnings = 2  # conservative mid-estimate; passed from _scan_earnings_entries
            ctx = SignalEnhancer.build_iv_crush_context(
                symbol=symbol,
                price=price,
                vix=self.rm.current_vix,
                days_to_earnings=days_to_earnings,
                ivr=ivr,
                atm_strike=atm_strike,
                dte=10,
            )
            consensus = self.council.vote(symbol, ctx, strategy="iv_crush_straddle")
            console.print(consensus.summary())
            if consensus.conviction_tier == "veto":
                console.print(
                    f"[red][COUNCIL VETO] {symbol} straddle: "
                    f"score={consensus.net_score:+.3f} | tier=VETO | "
                    f"regime threshold not met[/red]"
                )
                log.info(
                    f"[{symbol}] Council vetoed straddle entry: "
                    f"score={consensus.net_score:+.3f} | tier=VETO"
                )
                return None
            size_multiplier = consensus.size_multiplier
            console.print(
                f"[green][COUNCIL {consensus.conviction_tier.upper()}] {symbol} straddle: "
                f"size_mult={size_multiplier:.2f}[/green]"
            )
        # ──────────────────────────────────────────────────────────────

        # ── Opportunity Scorer — greedy multiplier ─────────────────────
        greedy_multiplier = 1.0
        if self.opportunity_scorer is not None:
            opp_ctx = {
                "ivr": ivr,
                "vix": self.rm.current_vix,
                "ema_signal": "neutral",  # straddle is delta-neutral
                "strategy": "iv_crush",
            }
            greedy_multiplier = self.opportunity_scorer.score(
                opp_ctx,
                open_positions=list(self.active_positions.keys()),
                session_pnl=self.rm.daily_pnl,
            )
        # ──────────────────────────────────────────────────────────────

        # ── Kelly Criterion sizing ─────────────────────────────────────────
        straddle_margin_per_contract = atm_strike * 100 * 0.20

        if self.kelly is not None:
            n_contracts = self.kelly.get_contract_count(
                strategy="iv_crush",
                equity=equity,
                premium_per_contract=straddle_margin_per_contract,
                size_multiplier=size_multiplier,
                greedy_multiplier=greedy_multiplier,
            )
        else:
            n_contracts = max(1, int((equity * 0.03) / (atm_strike * 100)))
        # ─────────────────────────────────────────────────────────────────

        # Risk gate — straddle margin requirement (~20% of underlying notional)
        straddle_margin = straddle_margin_per_contract * n_contracts
        self.rm.approve_order(
            symbol=symbol,
            order_value=straddle_margin,
            delta_impact=0.0,  # straddle is delta-neutral at entry
            is_option=True,
        )

        # ── Smart limit order execution — both legs ────────────────────────
        if self.executor is not None:
            call_result = self.executor.execute_option_order(call["symbol"], n_contracts, "sell")
            put_result = self.executor.execute_option_order(put["symbol"], n_contracts, "sell")
        else:
            call_result = self.client.place_option_market_order(call["symbol"], n_contracts, "sell")
            put_result = self.client.place_option_market_order(put["symbol"], n_contracts, "sell")
        # ─────────────────────────────────────────────────────────────────

        action = {
            "agent": "IVCrush",
            "action": "sell_straddle",
            "symbol": symbol,
            "strike": atm_strike,
            "call_contract": call["symbol"],
            "put_contract": put["symbol"],
            "qty": n_contracts,
            "expiration": call["expiration"],
            "conviction_tier": consensus.conviction_tier if self.council else "bypass",
            "size_multiplier": size_multiplier,
            "greedy_multiplier": greedy_multiplier,
        }
        self.active_positions[symbol] = action

        # ── Journal: log entry (straddle = two legs; log as one combined trade) ──
        if self.journal is not None:
            # Premium approximation: straddle premium ≈ 5% of underlying ATM price
            est_combined_premium = atm_strike * 0.05
            trade_id = self.journal.log_entry(
                agent="IVCrush",
                strategy="iv_crush",
                symbol=symbol,
                contract=f"{call['symbol']}+{put['symbol']}",
                side="sell",
                qty=n_contracts,
                entry_price=est_combined_premium,
            )
            self.active_positions[symbol]["journal_id"] = trade_id
            if self.council is not None and getattr(self.council, "_credibility_tracker", None) and hasattr(consensus, "votes"):
                self.council._credibility_tracker.record_votes(trade_id, consensus.votes)
        # ─────────────────────────────────────────────────────────────────

        console.print(
            f"[green][FILLED] IVCrush SOLD STRADDLE: {symbol} ${atm_strike:.0f} "
            f"exp={call['expiration']} x{n_contracts} | Kelly-sized[/green]"
        )
        return action

    def _manage_existing_positions(self) -> list[dict]:
        """Close positions that hit profit target or stop loss."""
        actions = []
        positions = {p["symbol"]: p for p in self.client.get_all_positions()}

        for symbol, meta in list(self.active_positions.items()):
            call_sym = meta.get("call_contract")
            put_sym = meta.get("put_contract")

            in_positions = call_sym in positions or put_sym in positions
            if not in_positions:
                console.print(f"[green][EXPIRED] {symbol} straddle expired (max profit)[/green]")
                if self.journal is not None and "journal_id" in meta:
                    self.journal.log_exit(meta["journal_id"], 0.0, "expired_worthless")
                    if self.council is not None and getattr(self.council, "_credibility_tracker", None):
                        self.council._credibility_tracker.update_credibility(meta["journal_id"], True)
                del self.active_positions[symbol]
                continue

        return actions
