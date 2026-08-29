"""
core/trade_journal.py — SQLite-backed Trade Performance Journal.

Every trade entry and exit is logged to a local SQLite database.
This gives the system persistent memory across sessions, enabling:
  1. Kelly Criterion position sizing (requires historical win/loss data)
  2. Dynamic capital allocation (orchestrator reads per-strategy stats)
  3. Post-session performance analysis

Schema:
    trades
      id           INTEGER PRIMARY KEY
      opened_at    TEXT     (ISO-8601 UTC)
      closed_at    TEXT     (ISO-8601 UTC, NULL while open)
      agent        TEXT     ('ThetaCollector' | 'MomoBreakout' | 'IVCrush')
      strategy     TEXT     ('theta' | 'momo' | 'iv_crush')
      symbol       TEXT     (underlying, e.g. 'SPY')
      contract     TEXT     (OCC symbol, e.g. 'SPY250919P00480000')
      side         TEXT     ('buy' | 'sell')
      qty          INTEGER
      entry_price  REAL     (price per share/unit, not per contract)
      exit_price   REAL     (NULL while open)
      pnl          REAL     (USD, NULL while open)
      pnl_pct      REAL     (fraction, NULL while open; positive = win)
      win          INTEGER  (1 = win, 0 = loss, NULL while open)
      exit_reason  TEXT     ('profit_target' | 'stop_loss' | 'trailing_stop'
                             | 'time_stop' | 'expired_worthless')
      council_score REAL    (LLM council net_score at entry, optional)

Usage:
    journal = TradeJournal()
    trade_id = journal.log_entry("MomoBreakout", "momo", "NVDA", contract, ...)
    journal.log_exit(trade_id, exit_price=2.45, reason="profit_target")
    stats = journal.get_strategy_stats("momo")
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table

import config

console = Console()
log = logging.getLogger(__name__)

_DB_PATH = config.LOG_DB_PATH
_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at     TEXT    NOT NULL,
    closed_at     TEXT,
    agent         TEXT    NOT NULL,
    strategy      TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    contract      TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    qty           INTEGER NOT NULL,
    entry_price   REAL    NOT NULL,
    exit_price    REAL,
    pnl           REAL,
    pnl_pct       REAL,
    win           INTEGER,
    exit_reason   TEXT,
    council_score REAL
);
"""


class TradeJournal:
    """
    Persistent trade log for performance tracking and Kelly sizing.

    Thread-safe for concurrent agent reads; writes are serialised
    through SQLite's WAL journal mode.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or _DB_PATH
        self._init_db()
        console.print(
            f"[cyan]TradeJournal initialised | db={self.db_path}[/cyan]"
        )

    # ──────────────────────────────────────────────
    # DB initialisation
    # ──────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create database file and schema if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL;")

    @contextmanager
    def _connect(self):
        """Context manager: open connection, commit on success, rollback on error."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ──────────────────────────────────────────────
    # Write Operations
    # ──────────────────────────────────────────────

    def log_entry(
        self,
        agent: str,
        strategy: str,
        symbol: str,
        contract: str,
        side: str,
        qty: int,
        entry_price: float,
        council_score: Optional[float] = None,
    ) -> int:
        """
        Record a new trade entry. Returns the trade ID for later exit logging.

        Args:
            agent:         Agent name (e.g. 'ThetaCollector')
            strategy:      Strategy key: 'theta' | 'momo' | 'iv_crush'
            symbol:        Underlying ticker (e.g. 'SPY')
            contract:      OCC options symbol
            side:          'buy' | 'sell'
            qty:           Number of contracts
            entry_price:   Premium per share at entry (multiply by 100 for per-contract)
            council_score: LLM council net score at signal time (optional, for analysis)

        Returns:
            trade_id (int) — store this and pass to log_exit() on close
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades
                  (opened_at, agent, strategy, symbol, contract,
                   side, qty, entry_price, council_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now, agent, strategy, symbol, contract,
                 side, qty, entry_price, council_score),
            )
            trade_id = cur.lastrowid

        log.info(
            f"[Journal] ENTRY logged: #{trade_id} | {agent} | {symbol} {contract} "
            f"| {side} x{qty} @ {entry_price:.4f}"
        )
        return trade_id

    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        reason: str,
    ) -> dict:
        """
        Record a trade exit. Computes P&L automatically.

        Args:
            trade_id:   ID returned by log_entry()
            exit_price: Premium per share at exit
            reason:     Exit reason string

        Returns:
            Dict with {pnl, pnl_pct, win}
        """
        now = datetime.now(timezone.utc).isoformat()

        # Fetch entry to compute P&L
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()

            if not row:
                log.error(f"[Journal] Trade #{trade_id} not found for exit logging")
                return {}

            entry_price = row["entry_price"]
            qty = row["qty"]
            side = row["side"]

            # P&L logic:
            # BUY side: profit = (exit - entry) × qty × 100
            # SELL side: profit = (entry - exit) × qty × 100  (sold high, buy back low)
            if side == "buy":
                pnl = (exit_price - entry_price) * qty * 100
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            else:  # sell (short options)
                pnl = (entry_price - exit_price) * qty * 100
                pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0

            win = 1 if pnl > 0 else 0

            conn.execute(
                """
                UPDATE trades
                   SET closed_at = ?, exit_price = ?,
                       pnl = ?, pnl_pct = ?, win = ?, exit_reason = ?
                 WHERE id = ?
                """,
                (now, exit_price, pnl, pnl_pct, win, reason, trade_id),
            )

        result = {"pnl": pnl, "pnl_pct": pnl_pct, "win": bool(win)}
        outcome = "WIN" if win else "LOSS"
        log.info(
            f"[Journal] EXIT logged: #{trade_id} | {reason} | "
            f"P&L={pnl:+.2f} ({pnl_pct:+.1%}) [{outcome}]"
        )
        return result

    # ──────────────────────────────────────────────
    # Read: Per-Strategy Statistics
    # ──────────────────────────────────────────────

    def get_strategy_stats(self, strategy: str) -> dict:
        """
        Compute win rate, avg win/loss for a strategy from closed trades.

        Returns:
            {
                "strategy":  str,
                "n_trades":  int,    # total closed trades
                "win_rate":  float,  # fraction 0.0–1.0
                "avg_win":   float,  # avg pnl_pct on winning trades (positive)
                "avg_loss":  float,  # avg |pnl_pct| on losing trades (positive)
                "total_pnl": float,  # USD sum of all closed P&L
                "sharpe":    float,  # simplified Sharpe-like metric (0 if < 5 trades)
            }
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pnl_pct, pnl, win
                  FROM trades
                 WHERE strategy = ? AND closed_at IS NOT NULL
                """,
                (strategy,),
            ).fetchall()

        if not rows:
            return {
                "strategy": strategy, "n_trades": 0,
                "win_rate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "total_pnl": 0.0, "sharpe": 0.0,
            }

        pnl_pcts = [r["pnl_pct"] for r in rows]
        pnls = [r["pnl"] for r in rows]
        wins = [r["pnl_pct"] for r in rows if r["win"] == 1]
        losses = [abs(r["pnl_pct"]) for r in rows if r["win"] == 0]

        import statistics
        n = len(rows)
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.001  # avoid div/0

        # Simplified Sharpe: mean / stdev of pnl_pct (annualise later if needed)
        if n >= 5 and statistics.stdev(pnl_pcts) > 0:
            sharpe = statistics.mean(pnl_pcts) / statistics.stdev(pnl_pcts)
        else:
            sharpe = 0.0

        return {
            "strategy": strategy,
            "n_trades": n,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": sum(pnls),
            "sharpe": sharpe,
        }

    def get_all_strategy_stats(self) -> dict[str, dict]:
        """Return stats for all strategies in one call."""
        return {
            s: self.get_strategy_stats(s)
            for s in ["theta", "momo", "iv_crush"]
        }

    # ──────────────────────────────────────────────
    # Read: Daily Summary
    # ──────────────────────────────────────────────

    def get_daily_summary(self) -> dict:
        """
        Summary of trades closed today (UTC date).
        """
        today = datetime.now(timezone.utc).date().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT strategy, pnl, win
                  FROM trades
                 WHERE closed_at LIKE ? AND closed_at IS NOT NULL
                """,
                (f"{today}%",),
            ).fetchall()

        total_pnl = sum(r["pnl"] for r in rows)
        wins = sum(1 for r in rows if r["win"] == 1)
        losses = sum(1 for r in rows if r["win"] == 0)

        return {
            "date": today,
            "trades_closed": len(rows),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(rows) if rows else 0.0,
            "total_pnl": total_pnl,
        }

    # ──────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────

    def print_performance_table(self) -> None:
        """Print a Rich console table of all strategy statistics."""
        table = Table(
            title="Strategy Performance (All-Time)",
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Strategy", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Win", justify="right", style="green")
        table.add_column("Avg Loss", justify="right", style="red")
        table.add_column("Sharpe", justify="right")
        table.add_column("Total P&L", justify="right")

        all_stats = self.get_all_strategy_stats()
        for strategy, s in all_stats.items():
            n = s["n_trades"]
            pnl_color = "green" if s["total_pnl"] >= 0 else "red"
            table.add_row(
                strategy.upper(),
                str(n),
                f"{s['win_rate']:.1%}" if n > 0 else "—",
                f"{s['avg_win']:.1%}" if n > 0 else "—",
                f"{s['avg_loss']:.1%}" if n > 0 else "—",
                f"{s['sharpe']:.2f}" if n >= 5 else "—",
                f"[{pnl_color}]${s['total_pnl']:+,.0f}[/{pnl_color}]" if n > 0 else "—",
            )
        console.print(table)
