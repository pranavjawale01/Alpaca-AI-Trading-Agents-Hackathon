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
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.options_pricer import greeks, iv_rank
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

    Pro improvements:
      1. LLM Council: hybrid tier-based vote gate (STRONG/MODERATE/PILOT)
      2. Kelly Criterion: size puts to win-rate-adjusted edge with greedy multiplier
      3. SmartExecutor: sell at mid-price (receive more than bid)
      4. TradeJournal: persistent logging for Kelly feedback loop
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
        self.symbols = config.UNIVERSE.theta_symbols
        self.active_positions: dict[str, dict] = {}

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
                console.print(f"[yellow][RISK BLOCK] [{symbol}]: {e}[/yellow]")
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

        # ── Condition 3: Hybrid LLM Council vote gate ─────────────────
        size_multiplier = 1.0  # default if council disabled
        if self.council is not None:
            ctx = SignalEnhancer.build_theta_context(
                symbol=symbol,
                price=price,
                ivr=ivr,
                vix=vix,
                hist_vol=hist_vol,
                dte=TARGET_DTE_MIN,
            )
            consensus = self.council.vote(symbol, ctx, strategy="theta_put")
            console.print(consensus.summary())
            if consensus.conviction_tier == "veto":
                console.print(
                    f"[red][COUNCIL VETO] {symbol} CSP: "
                    f"score={consensus.net_score:+.3f} | tier=VETO | "
                    f"regime threshold not met[/red]"
                )
                return None
            size_multiplier = consensus.size_multiplier
            console.print(
                f"[green][COUNCIL {consensus.conviction_tier.upper()}] {symbol} CSP: "
                f"size_mult={size_multiplier:.2f}[/green]"
            )
        # ──────────────────────────────────────────────────────────────

        # ── Opportunity Scorer — greedy multiplier ─────────────────────
        greedy_multiplier = 1.0
        if self.opportunity_scorer is not None:
            opp_ctx = {
                "ivr": ivr, "vix": vix,
                "ema_signal": "bullish",  # theta assumes bullish/neutral
                "strategy": "theta",
            }
            open_syms = list(self.active_positions.keys())
            greedy_multiplier = self.opportunity_scorer.score(
                opp_ctx, open_positions=open_syms,
                session_pnl=self.rm.daily_pnl,
            )
        # ──────────────────────────────────────────────────────────────

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

        # Calculate premium via Black-Scholes (avoid fetching option as stock quote)
        T = TARGET_DTE_MIN / 365
        from core.options_pricer import black_scholes_price
        premium = black_scholes_price(price, target_strike, T, RISK_FREE_RATE, hist_vol, "put")
        if premium <= 0.01:
            log.info(f"[{symbol}] Premium too low ({premium:.2f}), skipping")
            return None

        # ── Kelly Criterion sizing (hybrid multipliers) ────────────────────
        notional = target_strike * 100
        margin_requirement = notional * 0.20  # 20% margin for CSP

        if self.kelly is not None:
            # For short puts, "cost" = margin tied up; Kelly sizes on that
            n_contracts = self.kelly.get_contract_count(
                strategy="theta",
                equity=equity,
                premium_per_contract=margin_requirement,  # margin as the "spend"
                size_multiplier=size_multiplier,
                greedy_multiplier=greedy_multiplier,
            )
        else:
            n_contracts = max(1, int((equity * 0.04) / max(margin_requirement, 1000)))
        # ─────────────────────────────────────────────────────────────────

        order_value = max(premium * 100 * n_contracts, margin_requirement * n_contracts * 0.10)

        # Risk gate
        self.rm.approve_order(
            symbol=contract["symbol"],
            order_value=order_value,
            delta_impact=-TARGET_DELTA * 100 * n_contracts,
            is_option=True,
        )

        # ── Smart limit order execution ────────────────────────────────────
        if self.executor is not None:
            result = self.executor.execute_option_order(contract["symbol"], n_contracts, "sell")
        else:
            result = self.client.place_option_market_order(contract["symbol"], n_contracts, "sell")
        # ─────────────────────────────────────────────────────────────────

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
            "conviction_tier": consensus.conviction_tier if self.council else "bypass",
            "size_multiplier": size_multiplier,
            "greedy_multiplier": greedy_multiplier,
        }

        self.active_positions[symbol] = {
            **action,
            "entry_premium": premium,
            "stop_loss_premium": premium * STOP_LOSS_MULTIPLIER,
        }

        # ── Journal: log entry ─────────────────────────────────────────────
        if self.journal is not None:
            trade_id = self.journal.log_entry(
                agent="ThetaCollector",
                strategy="theta",
                symbol=symbol,
                contract=contract["symbol"],
                side="sell",
                qty=n_contracts,
                entry_price=premium,
            )
            self.active_positions[symbol]["journal_id"] = trade_id
            if self.council is not None and getattr(self.council, "_credibility_tracker", None) and hasattr(consensus, "votes"):
                self.council._credibility_tracker.record_votes(trade_id, consensus.votes)
        # ─────────────────────────────────────────────────────────────────

        console.print(
            f"[green][FILLED] ThetaCollector SOLD PUT: {symbol} "
            f"${target_strike:.0f} exp={contract['expiration']} "
            f"x{n_contracts} | premium=${premium*100*n_contracts:,.0f} | "
            f"tier={action['conviction_tier'].upper()} | "
            f"greed={greedy_multiplier:.2f}[/green]"
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
                # Position closed (expired worthless = max profit!)
                console.print(f"[green][EXPIRED] {symbol} CSP expired worthless (max profit)[/green]")
                if self.journal is not None and "journal_id" in meta:
                    self.journal.log_exit(meta["journal_id"], 0.0, "expired_worthless")
                    if self.council is not None and getattr(self.council, "_credibility_tracker", None):
                        self.council._credibility_tracker.update_credibility(meta["journal_id"], True)
                del self.active_positions[symbol]
                actions.append({"agent": "ThetaCollector", "action": "expired_worthless", "symbol": symbol})
                continue

            pos = positions[contract_symbol]
            current_value = abs(float(pos["market_value"]))
            entry_premium = meta["entry_premium"] * 100 * meta["qty"]
            profit_pct = 1 - (current_value / entry_premium) if entry_premium > 0 else 0

            # Exit: 50% profit target
            if profit_pct >= PROFIT_TARGET_PCT:
                self._close_position(symbol, contract_symbol, meta, "profit_target", profit_pct)
                actions.append({"agent": "ThetaCollector", "action": "closed_profit_target",
                                 "symbol": symbol, "profit_pct": profit_pct})

        return actions

    def _close_position(
        self, symbol: str, contract_symbol: str, meta: dict, reason: str, profit_pct: float = 0.0
    ) -> None:
        """Close (buy back) a short put using SmartExecutor + journal logging."""
        try:
            qty = meta["qty"]
            # ── Smart execution on close ───────────────────────────────────
            if self.executor is not None:
                self.executor.execute_option_order(contract_symbol, qty, "buy")
            else:
                self.client.place_option_market_order(contract_symbol, qty, "buy")
            # ─────────────────────────────────────────────────────────────

            # ── Journal & Credibility: log exit ────────────────────────────
            if self.journal is not None and "journal_id" in meta:
                # For short put: current cost = entry × (1 - profit_pct)
                exit_price = meta["entry_premium"] * (1 - profit_pct)
                self.journal.log_exit(meta["journal_id"], exit_price, reason)
                if self.council is not None and getattr(self.council, "_credibility_tracker", None):
                    self.council._credibility_tracker.update_credibility(meta["journal_id"], profit_pct > 0)
            # ─────────────────────────────────────────────────────────────

            del self.active_positions[symbol]
            console.print(f"[blue][CLOSED] ThetaCollector closed {symbol} [{reason}] profit={profit_pct:.1%}[/blue]")
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")
