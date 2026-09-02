"""
agents/orchestrator.py — Master Orchestrator Agent.

The Orchestrator is the "brain" of the Cache Me system.

Responsibilities:
  1. Detect market regime (risk-on, risk-off, neutral) using VIX
  2. Route capital allocation between sub-agents based on regime
  3. Update the Risk Manager with latest account state
  4. Coordinate earnings calendar across all agents
  5. Run all sub-agents in the correct order each session
  6. Optionally use LLM (Featherless AI via MCP) for edge-case decisions

Regime Map:
  VIX < 18  → Risk-ON   → Theta + Momo both active, hedge off
  18 ≤ VIX < 28 → Neutral → Theta active, Momo reduced, hedge light
  VIX ≥ 28  → Risk-OFF  → Only hedge + theta defensive; momo/iv_crush off
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.mcp_bridge import MCPBridge
from core.risk_manager import RiskManager
from core.llm_council import LLMCouncil
from core.trade_journal import TradeJournal
from core.kelly_sizer import KellySizer
from core.smart_executor import SmartExecutor
from core.opportunity_scorer import OpportunityScorer
from core.model_credibility import ModelCredibilityTracker
from agents.theta_collector import ThetaCollectorAgent
from agents.iv_crush_agent import IVCrushAgent
from agents.momo_breakout import MomoBreakoutAgent
from agents.hedge_agent import HedgeAgent

console = Console()
log = logging.getLogger(__name__)

# VIX regime thresholds
VIX_RISK_ON = 18
VIX_RISK_OFF = 28


def detect_regime(vix: float) -> str:
    if vix < VIX_RISK_ON:
        return "risk_on"
    elif vix < VIX_RISK_OFF:
        return "neutral"
    else:
        return "risk_off"


class Orchestrator:
    """
    Master controller that runs all agents in sequence each market session.

    Usage:
        orch = Orchestrator()
        orch.run_session()
    """

    def __init__(self, client: Optional[AlpacaClient] = None) -> None:
        console.rule("[bold blue]Cache Me If You Can — Master Orchestrator[/bold blue]")

        # Core infrastructure
        self.client = client if client is not None else AlpacaClient()
        self.md = MarketData(self.client)
        self.rm = RiskManager()
        self.mcp = MCPBridge(self.client)

        # LLM Council — 3-model voting ensemble for signal quality filtering
        self.council = LLMCouncil()

        # ── Pro Trading Systems ──────────────────────────────────────────
        # Trade journal: persistent SQLite log for every entry/exit
        self.journal = TradeJournal()

        # Kelly sizer: reads journal stats → computes optimal position sizes
        self.kelly = KellySizer(self.journal)

        # Smart executor: mid-price limit orders → avoids paying full spread
        self.executor = SmartExecutor(self.client)

        # Opportunity scorer: greedy multiplier from stacking market criteria
        self.opportunity_scorer = OpportunityScorer()

        # Model credibility tracker: per-model accuracy weights
        self.credibility_tracker = ModelCredibilityTracker()
        self.council.set_credibility_tracker(self.credibility_tracker)
        # ────────────────────────────────────────────────────────────────

        # Sub-agents (all pro systems + council + opportunity scorer injected)
        self.theta = ThetaCollectorAgent(
            self.client, self.md, self.rm,
            council=self.council, kelly_sizer=self.kelly,
            smart_executor=self.executor, journal=self.journal,
            opportunity_scorer=self.opportunity_scorer,
        )
        self.iv_crush = IVCrushAgent(
            self.client, self.md, self.rm,
            council=self.council, kelly_sizer=self.kelly,
            smart_executor=self.executor, journal=self.journal,
            opportunity_scorer=self.opportunity_scorer,
        )
        self.momo = MomoBreakoutAgent(
            self.client, self.md, self.rm,
            council=self.council, kelly_sizer=self.kelly,
            smart_executor=self.executor, journal=self.journal,
            opportunity_scorer=self.opportunity_scorer,
        )
        self.hedge = HedgeAgent(self.client, self.md, self.rm)  # hedge never filtered

        self.session_log: list[dict] = []
        self._cached_earnings: list[dict] = []
        self._earnings_last_fetched: Optional[datetime] = None
        console.print("[bold green]All agents initialised successfully (HYBRID MODE)[/bold green]")

    # ─────────────────────────────────────────
    # Main Session Runner
    # ─────────────────────────────────────────

    def run_session(self) -> dict:
        """
        Run a full trading session.
        Called once per market open by the scheduler.

        Returns:
            Session summary dict
        """
        session_start = datetime.now(timezone.utc)
        console.rule(f"[cyan]Session Start: {session_start.strftime('%Y-%m-%d %H:%M UTC')}[/cyan]")

        # Step 1: Update risk manager state
        self._update_risk_state()

        # Step 2: Detect market regime
        vix = self.rm.current_vix
        regime = detect_regime(vix)
        console.print(f"[bold]Market Regime: [yellow]{regime.upper()}[/yellow] | VIX={vix:.1f}[/bold]")

        # Push regime to LLM council for regime-adaptive consensus thresholds
        if self.council is not None:
            self.council.set_regime(regime)

        # Step 3: Run agents based on regime
        all_actions = []

        # Hedge always runs first (protective)
        all_actions += self.hedge.run()

        if regime in ("risk_on", "neutral"):
            # Theta runs in all non-extreme regimes
            all_actions += self.theta.run()

        # Momo active across all regimes:
        # In risk_on & neutral: looks for bullish call breakouts
        # In risk_off: looks for bearish breakdown put buys (short directional alpha)
        all_actions += self.momo.run(regime=regime)

        if regime in ("risk_on", "neutral"):
            # IV crush needs earnings calendar — use LLM to help
            earnings = self._get_earnings_calendar(regime)
            all_actions += self.iv_crush.run(earnings)

        # Step 4: Log session
        session_summary = self._build_session_summary(session_start, regime, all_actions)
        self.session_log.append(session_summary)
        self._print_session_summary(session_summary)

        # Step 5: Print cumulative performance & Kelly edges (all-time from journal)
        self.journal.print_performance_table()
        self.kelly.print_strategy_edges(self.rm.equity)

        return session_summary

    # ─────────────────────────────────────────
    # State Updates
    # ─────────────────────────────────────────

    def _update_risk_state(self) -> None:
        """Refresh risk manager with latest account data."""
        try:
            account = self.client.get_account()
            self.rm.update_equity(account["equity"])

            # Estimate daily P&L
            import config
            daily_pnl = account["equity"] - config.RISK.starting_balance
            self.rm.update_daily_pnl(daily_pnl)

            # Update VIX
            vix = self.md.get_vix()
            self.rm.update_vix(vix)

            # Update portfolio delta from positions
            positions = self.client.get_all_positions()
            total_delta = self._estimate_portfolio_delta(positions)
            self.rm.update_portfolio_delta(total_delta)

            options_exposure = sum(
                abs(float(p["market_value"]))
                for p in positions
                if p.get("asset_class") == "us_option"
            )
            self.rm.update_options_exposure(options_exposure)

            log.info(
                f"Risk state updated: equity=${account['equity']:,.0f} "
                f"| daily_pnl=${daily_pnl:+,.0f} | VIX={vix:.1f} | delta={total_delta:.1f}"
            )
        except Exception as e:
            log.error(f"Risk state update failed: {e}")

    def _estimate_portfolio_delta(self, positions: list[dict]) -> float:
        """
        Estimate total portfolio delta from positions.
        Uses 1.0 delta for stock positions, 0.5 for options (rough estimate).
        Correctly accounts for option type (Call vs Put) and direction (Long vs Short):
          - Long Call  (qty > 0): +50 delta
          - Short Call (qty < 0): -50 delta
          - Long Put   (qty > 0): -50 delta
          - Short Put  (qty < 0): +50 delta
        """
        total = 0.0
        for p in positions:
            qty = float(p.get("qty", 0))
            sym = str(p.get("symbol", ""))
            if p.get("asset_class") == "us_option" or any(c in sym for c in ("C0", "P0", "C1", "P1")):
                # In OCC symbology, the 9th character from the end is 'C' or 'P' (before 8 digits of strike)
                is_put = "P" in sym[-9:] if len(sym) >= 9 else ("P" in sym)
                # Long put has negative delta; short put has positive delta
                delta_multiplier = -50.0 if is_put else 50.0
                total += qty * delta_multiplier
            else:
                total += qty  # stocks: 1 delta each
        return total

    # ─────────────────────────────────────────
    # Earnings Calendar (LLM-Assisted)
    # ─────────────────────────────────────────

    def _get_earnings_calendar(self, regime: str) -> list[dict]:
        """
        Use LLM (via MCP) to get earnings events for the watchlist.
        Caches results for 6 hours to avoid redundant LLM queries.
        Falls back to empty list if MCP is unavailable.
        """
        now = datetime.now(timezone.utc)
        if (
            self._earnings_last_fetched is not None
            and (now - self._earnings_last_fetched).total_seconds() < 21600  # 6 hours
        ):
            return self._cached_earnings

        watchlist = [
            "AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT", "GOOGL", "AMD"
        ]
        try:
            response = self.mcp.query(
                f"Which of these companies have earnings announcements in the next 3 days? "
                f"Watchlist: {watchlist}. "
                f"Return JSON array: [{{'symbol': 'AAPL', 'date': 'YYYY-MM-DD'}}]. "
                f"If none, return []."
            )
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                calendar = json.loads(response[start:end])
                log.info(f"Earnings calendar from LLM: {calendar}")
                self._cached_earnings = calendar
                self._earnings_last_fetched = now
                return calendar
        except Exception as e:
            log.warning(f"Earnings calendar fetch failed: {e}")

        return self._cached_earnings if self._cached_earnings else []

    # ─────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────

    def _build_session_summary(
        self,
        start: datetime,
        regime: str,
        actions: list[dict],
    ) -> dict:
        account = self.client.get_account()
        risk_summary = self.rm.summary()

        # Tally council statistics from all votes this session
        council_votes = []
        if self.council and self.council._available:
            # Collect from in-memory vote history (future: persist per session)
            pass  # votes are logged per-trade; aggregate counts come from actions

        vetoed = sum(1 for a in actions if a.get("action") == "council_vetoed")
        executed = len([a for a in actions if a.get("action") != "council_vetoed"])

        return {
            "timestamp": start.isoformat(),
            "regime": regime,
            "vix": self.rm.current_vix,
            "equity": account["equity"],
            "daily_pnl": risk_summary["daily_pnl"],
            "daily_pnl_pct": risk_summary["daily_pnl_pct"],
            "portfolio_delta": risk_summary["portfolio_delta"],
            "options_exposure_pct": risk_summary["options_exposure_pct"],
            "actions_taken": len(actions),
            "actions": actions,
            "council_stats": {
                "enabled": self.council.enabled if self.council else False,
                "models": self.council.models if self.council else [],
                "threshold": self.council.threshold if self.council else None,
                "trades_vetoed": vetoed,
                "trades_executed": executed,
            },
        }

    def _print_session_summary(self, summary: dict) -> None:
        """Pretty-print session summary to console."""
        console.rule("[bold blue]Session Summary[/bold blue]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="white")

        pnl_color = "green" if summary["daily_pnl"] >= 0 else "red"
        table.add_row("Regime", f"[yellow]{summary['regime'].upper()}[/yellow]")
        table.add_row("VIX", f"{summary['vix']:.1f}")
        table.add_row("Equity", f"${summary['equity']:,.2f}")
        table.add_row("Daily P&L", f"[{pnl_color}]${summary['daily_pnl']:+,.2f} ({summary['daily_pnl_pct']:+.2f}%)[/{pnl_color}]")
        table.add_row("Portfolio Delta", f"{summary['portfolio_delta']:.1f}")
        table.add_row("Options Exposure", f"{summary['options_exposure_pct']:.1f}%")
        table.add_row("Actions Taken", str(summary["actions_taken"]))

        # Council stats
        cs = summary.get("council_stats", {})
        if cs.get("enabled"):
            n_models = len(cs.get("models", []))
            table.add_row("─" * 20, "─" * 20)
            table.add_row(
                "Council Models",
                f"[bold]{n_models} models[/bold] | threshold={cs.get('threshold', '?')}"
            )
            table.add_row("Trades Executed", f"[green]{cs.get('trades_executed', '?')}[/green]")
            table.add_row("Trades Vetoed", f"[yellow]{cs.get('trades_vetoed', 0)}[/yellow]")
        else:
            table.add_row("LLM Council", "[dim]disabled (rules-only)[/dim]")

        console.print(table)
