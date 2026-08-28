"""
main.py — Entry point for Cache Me trading system.

Modes:
  python main.py run        → Run a single trading session now
  python main.py schedule   → Start scheduled daily sessions (market hours)
  python main.py status     → Print current account status
  python main.py backtest   → (placeholder) Run backtesting

Usage:
  python main.py run
  python main.py status
"""

from __future__ import annotations

import sys
import logging
import time
from datetime import datetime, timezone

import schedule as sched
from rich.console import Console
from rich.logging import RichHandler

import config
from agents.orchestrator import Orchestrator

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)
console = Console()


def run_session() -> None:
    """Run a single trading session."""
    console.rule("[bold green]🚀 Cache Me — Starting Trading Session[/bold green]")
    orch = Orchestrator()
    summary = orch.run_session()
    console.print(f"\n[bold]Session complete. P&L: [{'green' if summary['daily_pnl'] >= 0 else 'red'}]${summary['daily_pnl']:+,.2f}[/][/bold]")


def show_status() -> None:
    """Print current account status without trading."""
    from core.alpaca_client import AlpacaClient
    from core.risk_manager import RiskManager

    console.rule("[bold cyan]Cache Me — Account Status[/bold cyan]")
    client = AlpacaClient()
    account = client.get_account()
    positions = client.get_all_positions()

    console.print(f"\n[bold]Account ID:[/bold] {account['id']}")
    console.print(f"[bold]Equity:[/bold]     ${account['equity']:,.2f}")
    console.print(f"[bold]Cash:[/bold]       ${account['cash']:,.2f}")
    console.print(f"[bold]Buying Power:[/bold] ${account['buying_power']:,.2f}")

    if positions:
        from rich.table import Table
        table = Table(title="Open Positions", header_style="bold cyan")
        table.add_column("Symbol")
        table.add_column("Qty")
        table.add_column("Avg Price")
        table.add_column("Market Value")
        table.add_column("Unrealized P&L")

        for p in positions:
            color = "green" if float(p["unrealized_pl"]) >= 0 else "red"
            table.add_row(
                p["symbol"],
                str(p["qty"]),
                f"${p['avg_entry_price']:.2f}",
                f"${p['market_value']:.2f}",
                f"[{color}]${p['unrealized_pl']:+.2f}[/{color}]",
            )
        console.print(table)
    else:
        console.print("\n[dim]No open positions.[/dim]")


def run_scheduled() -> None:
    """
    Schedule daily trading sessions at market open (ET → UTC).
    Market opens at 09:30 ET = 14:30 UTC.
    Pre-scan at 09:00 ET = 14:00 UTC.
    """
    console.rule("[bold blue]Cache Me — Scheduled Mode[/bold blue]")
    console.print("Sessions scheduled: 14:00 UTC (pre-scan) and 14:30 UTC (trading)")

    sched.every().day.at("14:00").do(lambda: log.info("Pre-market scan starting..."))
    sched.every().day.at("14:30").do(run_session)

    while True:
        sched.run_pending()
        time.sleep(30)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    if mode == "run":
        run_session()
    elif mode == "status":
        show_status()
    elif mode == "schedule":
        run_scheduled()
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        console.print("Usage: python main.py [run|status|schedule]")
        sys.exit(1)


if __name__ == "__main__":
    main()
