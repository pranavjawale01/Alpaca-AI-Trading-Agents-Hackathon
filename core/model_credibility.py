"""
core/model_credibility.py — Per-Model LLM Accuracy & Dynamic Credibility Weighting.

Tracks the voting performance of each LLM in the council over time and dynamically
adjusts voting weights based on trade outcomes (profitable vs. unprofitable).

Mechanics:
  1. Record votes per trade (`model_votes` table).
  2. When a trade closes, evaluate each model's recommendation (`update_credibility`):
     - Buy/Sell on profitable trade: correct (+0.05 weight, +1 correct_votes)
     - Buy/Sell on unprofitable trade: incorrect (-0.05 weight)
     - Hold votes: skipped from outcome evaluation
     - Clamp weights to [0.5, 1.5]
     - Decay toward 1.0: weight = weight * 0.98 + 1.0 * 0.02
  3. Provide weighted influence to council decisions (`get_weights`).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

import config

console = Console()
log = logging.getLogger(__name__)

_DEFAULT_DB_PATH: str = getattr(config, "LOG_DB_PATH", "logs/trading.db")

_VOTES_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    model_name TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_CREDIBILITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_credibility (
    model_name TEXT PRIMARY KEY,
    weight REAL NOT NULL DEFAULT 1.0,
    correct_votes INTEGER DEFAULT 0,
    total_votes INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class ModelCredibilityTracker:
    """
    Tracks per-model LLM voting accuracy and computes credibility weights.

    Stores vote history and dynamic weights in a local SQLite database
    (shares logs/trading.db with TradeJournal).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialise tracker with SQLite database path.

        Args:
            db_path: Path to SQLite database file. Defaults to config.LOG_DB_PATH.
        """
        self.db_path: str = db_path if db_path is not None else _DEFAULT_DB_PATH
        self._init_db()
        console.print(
            f"[cyan]ModelCredibilityTracker initialised | db={self.db_path}[/cyan]"
        )

    def _init_db(self) -> None:
        """Create database file, directories, and tables if they do not exist."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = self._connect()
        try:
            with conn:
                conn.execute(_VOTES_SCHEMA)
                conn.execute(_CREDIBILITY_SCHEMA)
                conn.execute("PRAGMA journal_mode=WAL;")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """
        Helper to connect to the SQLite database.

        Returns:
            sqlite3.Connection with Row factory configured.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def record_votes(
        self,
        trade_id: int,
        votes: list[dict[str, Any] | Any],
    ) -> None:
        """
        Record council model votes associated with a trade.

        Args:
            trade_id: Unique identifier of the associated trade.
            votes: List of vote dicts (or ModelVote objects) containing
                   'model' (or 'model_name'), 'action', and 'confidence'.
        """
        if not votes:
            return

        records: list[tuple[int, str, str, float]] = []
        for v in votes:
            if isinstance(v, dict):
                model_name = str(v.get("model") or v.get("model_name") or "unknown")
                action = str(v.get("action", "")).lower()
                confidence = float(v.get("confidence", 0.0))
            else:
                model_name = str(getattr(v, "model", getattr(v, "model_name", "unknown")))
                action = str(getattr(v, "action", "")).lower()
                confidence = float(getattr(v, "confidence", 0.0))
            records.append((trade_id, model_name, action, confidence))

        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO model_votes (trade_id, model_name, action, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    records,
                )
        finally:
            conn.close()

        log.info("Recorded %d model votes for trade %s", len(records), trade_id)

    def update_credibility(self, trade_id: int, trade_profitable: bool) -> None:
        """
        Update credibility weights for models that voted on a trade.

        Rules:
          - Only models voting 'buy' or 'sell' (not 'hold') are evaluated.
          - If trade was profitable: correct_votes += 1, weight += 0.05.
          - If trade was not profitable: weight -= 0.05.
          - Weight clamped to [0.5, 1.5].
          - Decay toward 1.0 applied: weight = weight * 0.98 + 1.0 * 0.02.

        Args:
            trade_id: Identifier of the completed trade.
            trade_profitable: True if trade generated positive P&L, False otherwise.
        """
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT model_name, action, confidence FROM model_votes WHERE trade_id = ?",
                    (trade_id,),
                )
                vote_rows = cursor.fetchall()

                if not vote_rows:
                    log.warning("No model votes found for trade_id=%s", trade_id)
                    return

                for row in vote_rows:
                    model_name = str(row["model_name"])
                    action = str(row["action"]).lower()

                    # Only 'buy' and 'sell' votes are evaluated for trade outcome accuracy
                    if action not in ("buy", "sell"):
                        continue

                    # Fetch current credibility record
                    cred_cursor = conn.execute(
                        "SELECT weight, correct_votes, total_votes FROM model_credibility WHERE model_name = ?",
                        (model_name,),
                    )
                    cred_row = cred_cursor.fetchone()

                    if cred_row:
                        current_weight = float(cred_row["weight"])
                        correct_votes = int(cred_row["correct_votes"])
                        total_votes = int(cred_row["total_votes"])
                    else:
                        current_weight = 1.0
                        correct_votes = 0
                        total_votes = 0

                    total_votes += 1
                    if trade_profitable:
                        correct_votes += 1
                        new_weight = current_weight + 0.05
                    else:
                        new_weight = current_weight - 0.05

                    # Clamp to [0.5, 1.5]
                    new_weight = max(0.5, min(1.5, new_weight))

                    # Apply decay toward 1.0
                    new_weight = new_weight * 0.98 + 1.0 * 0.02

                    # Update credibility database
                    conn.execute(
                        """
                        INSERT INTO model_credibility (model_name, weight, correct_votes, total_votes, updated_at)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(model_name) DO UPDATE SET
                            weight = excluded.weight,
                            correct_votes = excluded.correct_votes,
                            total_votes = excluded.total_votes,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (model_name, new_weight, correct_votes, total_votes),
                    )
                    log.info(
                        "Updated credibility for %s: weight=%.4f, correct=%d/%d (profitable=%s)",
                        model_name,
                        new_weight,
                        correct_votes,
                        total_votes,
                        trade_profitable,
                    )
        finally:
            conn.close()

    def get_weights(self) -> dict[str, float]:
        """
        Return mapping of model names to credibility weights.

        Models not tracked in the database return default weight 1.0 when queried.

        Returns:
            Dict mapping model_name to credibility weight.
        """
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT model_name, weight FROM model_credibility"
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        class _DefaultCredibilityDict(dict):
            def __missing__(self, key: str) -> float:
                return 1.0

        weights = _DefaultCredibilityDict(
            {str(row["model_name"]): float(row["weight"]) for row in rows}
        )
        return weights

    def get_weight(self, model_name: str) -> float:
        """
        Return credibility weight for a single model (defaults to 1.0 if not tracked).

        Args:
            model_name: Name of the LLM model.

        Returns:
            Float weight in [0.5, 1.5].
        """
        return self.get_weights().get(model_name, 1.0)

    def get_model_stats(self) -> list[dict[str, Any]]:
        """
        Return performance summary statistics for all tracked models.

        Returns:
            List of dicts containing model statistics.
        """
        conn = self._connect()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    SELECT model_name, weight, correct_votes, total_votes, updated_at
                    FROM model_credibility
                    ORDER BY weight DESC
                    """
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        stats = []
        for r in rows:
            total = int(r["total_votes"])
            correct = int(r["correct_votes"])
            acc = (correct / total * 100.0) if total > 0 else 0.0
            stats.append(
                {
                    "model_name": str(r["model_name"]),
                    "weight": float(r["weight"]),
                    "correct_votes": correct,
                    "total_votes": total,
                    "accuracy_pct": acc,
                    "updated_at": str(r["updated_at"]),
                }
            )
        return stats

    def display_summary(self) -> None:
        """Print a formatted Rich table of model credibility scores."""
        stats = self.get_model_stats()
        if not stats:
            console.print("[dim]No model credibility data recorded yet.[/dim]")
            return

        table = Table(title="LLM Council Credibility Leaderboard")
        table.add_column("Model", style="cyan")
        table.add_column("Weight", justify="right", style="bold")
        table.add_column("Correct", justify="right", style="green")
        table.add_column("Total", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Last Updated", style="dim")

        for s in stats:
            color = (
                "green"
                if s["weight"] >= 1.05
                else ("red" if s["weight"] <= 0.95 else "yellow")
            )
            table.add_row(
                s["model_name"],
                f"[{color}]{s['weight']:.4f}[/{color}]",
                str(s["correct_votes"]),
                str(s["total_votes"]),
                f"{s['accuracy_pct']:.1f}%",
                s["updated_at"],
            )

        console.print(table)
