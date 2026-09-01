"""
core/kelly_sizer.py — Kelly Criterion Position Sizer.

Computes mathematically optimal position sizes based on historical win rate
and reward:risk ratio per strategy, then applies a conservative fraction.

Kelly Formula:
    f* = (b·p - q) / b
    where:
        b = avg_win / avg_loss  (reward-to-risk ratio)
        p = historical win rate
        q = 1 - p

We use Quarter-Kelly (f* × 0.25) — the industry standard for live trading:
  - Full Kelly maximises long-run growth but causes catastrophic drawdowns
  - Half-Kelly is safer but still aggressive
  - Quarter-Kelly is conservative, smooth equity curve, preferred by most quants

Without sufficient trade history (< MIN_TRADES), falls back to a conservative
default percentage so the agent can still trade from Day 1.

Usage:
    sizer = KellySizer(journal)
    dollar_risk = sizer.get_position_size("theta", equity=100_000)
    n_contracts = max(1, int(dollar_risk / (premium * 100)))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from rich.console import Console

import config

if TYPE_CHECKING:
    from core.trade_journal import TradeJournal

console = Console()
log = logging.getLogger(__name__)

# Minimum number of closed trades before Kelly is reliable
_MIN_KELLY_TRADES = config.EXECUTION.kelly_min_trades

# Hardcoded sane defaults when no history (conservative)
_DEFAULT_PCT_BY_STRATEGY = {
    "theta":    0.010,   # 1.0% equity — theta is our most reliable strategy
    "momo":     0.006,   # 0.6% equity — higher risk, lower default
    "iv_crush": 0.008,   # 0.8% equity — moderate risk
    "default":  0.005,   # 0.5% equity — safe fallback
}


class KellySizer:
    """
    Per-strategy Kelly Criterion position sizer.

    Reads win/loss history from TradeJournal and computes the optimal
    dollar risk per trade for each strategy independently.
    """

    def __init__(
        self,
        journal: "TradeJournal",
        kelly_fraction: Optional[float] = None,
    ) -> None:
        self.journal = journal
        self.kelly_fraction = kelly_fraction or config.EXECUTION.kelly_fraction
        console.print(
            f"[cyan]KellySizer initialised | "
            f"fraction={self.kelly_fraction:.2f} | "
            f"min_trades={_MIN_KELLY_TRADES}[/cyan]"
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def get_position_size(
        self,
        strategy: str,
        equity: float,
        override_max_pct: Optional[float] = None,
        size_multiplier: float = 1.0,
        greedy_multiplier: float = 1.0,
    ) -> float:
        """
        Compute optimal dollar position size for a strategy.

        Args:
            strategy: "theta" | "momo" | "iv_crush" | "default"
            equity: Current portfolio equity in USD
            override_max_pct: Hard cap on position size (fraction of equity).
                              Falls back to config.RISK.max_position_pct if None.
            size_multiplier: Council conviction tier multiplier (0.40–1.00).
                             From HybridConsensusResult.
            greedy_multiplier: Opportunity score multiplier (1.0–2.0).
                               From OpportunityScorer.

        Returns:
            Dollar amount to risk on this trade (USD).
            Already capped by the hard risk limit.
        """
        max_pct = override_max_pct or config.RISK.max_position_pct
        hard_cap_dollars = equity * max_pct

        # Compute effective Kelly fraction with hybrid multipliers
        effective_kelly = self.kelly_fraction * size_multiplier * greedy_multiplier
        max_kelly = config.HYBRID.max_kelly_fraction
        effective_kelly = min(effective_kelly, max_kelly)

        stats = self.journal.get_strategy_stats(strategy)
        n = stats.get("n_trades", 0)

        if n < _MIN_KELLY_TRADES:
            # Not enough history — use conservative defaults, still apply multipliers
            default_pct = _DEFAULT_PCT_BY_STRATEGY.get(strategy, _DEFAULT_PCT_BY_STRATEGY["default"])
            adjusted_pct = default_pct * size_multiplier * greedy_multiplier
            size = equity * adjusted_pct
            log.info(
                f"[KellySizer][{strategy}] Only {n} trades < {_MIN_KELLY_TRADES} minimum. "
                f"Using default {default_pct*100:.1f}% × {size_multiplier:.2f} × {greedy_multiplier:.2f} → ${size:,.0f}"
            )
            return min(size, hard_cap_dollars)

        p = stats["win_rate"]
        q = 1.0 - p
        avg_win = stats["avg_win"]        # average profit on winning trades (fraction)
        avg_loss = stats["avg_loss"]       # average loss on losing trades (fraction, positive)

        if avg_loss <= 0 or p <= 0:
            log.warning(f"[KellySizer][{strategy}] Degenerate stats (p={p}, avg_loss={avg_loss}) — using default")
            default_pct = _DEFAULT_PCT_BY_STRATEGY.get(strategy, _DEFAULT_PCT_BY_STRATEGY["default"])
            return min(equity * default_pct, hard_cap_dollars)

        # b = reward:risk ratio
        b = avg_win / avg_loss

        # Raw Kelly fraction: f* = (b·p - q) / b
        raw_kelly = (b * p - q) / b

        if raw_kelly <= 0:
            # Negative Kelly = negative edge → don't trade this strategy
            log.warning(
                f"[KellySizer][{strategy}] Negative Kelly ({raw_kelly:.3f}) — "
                f"win_rate={p:.1%}, b={b:.2f}. Using minimum size."
            )
            return equity * 0.003  # minimum 0.3% — keep skin in the game

        # Apply fractional Kelly with hybrid multipliers
        fractional_kelly_pct = raw_kelly * effective_kelly

        size = equity * fractional_kelly_pct
        size = min(size, hard_cap_dollars)  # hard risk cap

        console.print(
            f"[cyan][Kelly][{strategy}] "
            f"p={p:.1%} | b={b:.2f}:1 | f*={raw_kelly:.3f} | "
            f"eff_kelly={effective_kelly:.3f} (size={size_multiplier:.2f}×greed={greedy_multiplier:.2f}) "
            f"→ ${size:,.0f}[/cyan]"
        )
        return size

    def get_contract_count(
        self,
        strategy: str,
        equity: float,
        premium_per_contract: float,
        override_max_pct: Optional[float] = None,
        size_multiplier: float = 1.0,
        greedy_multiplier: float = 1.0,
    ) -> int:
        """
        Convenience: returns number of option contracts (rounded down).

        Args:
            premium_per_contract: Cost/credit per contract in USD (e.g. premium × 100).
            size_multiplier: Council conviction tier multiplier (0.40–1.00).
            greedy_multiplier: Opportunity score multiplier (1.0–2.0).

        Returns:
            Integer number of contracts (minimum 1).
        """
        if premium_per_contract <= 0:
            return 1
        dollar_size = self.get_position_size(
            strategy, equity, override_max_pct,
            size_multiplier=size_multiplier,
            greedy_multiplier=greedy_multiplier,
        )
        n = max(1, int(dollar_size / premium_per_contract))
        log.info(
            f"[KellySizer][{strategy}] ${dollar_size:,.0f} / ${premium_per_contract:,.0f} "
            f"per contract → {n} contracts"
        )
        return n

    # ──────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────

    def print_strategy_edges(self, equity: float) -> None:
        """Print a Rich table of all strategy Kelly sizes for debugging."""
        from rich.table import Table
        table = Table(title="Kelly Sizer — Strategy Edges", header_style="bold cyan")
        table.add_column("Strategy", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("b (R:R)", justify="right")
        table.add_column("Raw Kelly", justify="right")
        table.add_column("¼-Kelly $", justify="right", style="green")

        for strategy in ["theta", "momo", "iv_crush"]:
            stats = self.journal.get_strategy_stats(strategy)
            n = stats.get("n_trades", 0)
            p = stats.get("win_rate", 0)
            q = 1 - p
            avg_win = stats.get("avg_win", 0)
            avg_loss = stats.get("avg_loss", 1)
            b = avg_win / max(avg_loss, 0.001)
            raw_k = max(0, (b * p - q) / b) if b > 0 and p > 0 else 0
            size = self.get_position_size(strategy, equity)
            table.add_row(
                strategy,
                str(n),
                f"{p:.1%}",
                f"{b:.2f}",
                f"{raw_k:.3f}",
                f"${size:,.0f}",
            )
        console.print(table)
