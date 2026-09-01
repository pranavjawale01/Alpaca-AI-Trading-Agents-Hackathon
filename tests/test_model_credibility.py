"""
tests/test_model_credibility.py — Unit tests for ModelCredibilityTracker.

Tests vote recording, credibility updates, weight decay/clamping,
and performance reporting offline using SQLite in temporary directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from core.model_credibility import ModelCredibilityTracker


@dataclass
class DummyModelVote:
    model: str
    action: str
    confidence: float
    reasoning: str = ""


@pytest.fixture
def tracker(tmp_path: Path) -> ModelCredibilityTracker:
    """Fixture providing a fresh ModelCredibilityTracker with an isolated temp DB."""
    db_file = str(tmp_path / "test_trading.db")
    return ModelCredibilityTracker(db_path=db_file)


class TestModelCredibilityTrackerInit:

    def test_init_creates_tables(self, tracker: ModelCredibilityTracker):
        """Verify model_votes and model_credibility tables exist after initialisation."""
        with tracker._connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row["name"] for row in cursor.fetchall()]

        assert "model_votes" in tables
        assert "model_credibility" in tables

    def test_get_weights_empty_db(self, tracker: ModelCredibilityTracker):
        """Untracked models should default to 1.0 weight."""
        weights = tracker.get_weights()
        assert weights["non_existent_model"] == 1.0
        assert tracker.get_weight("non_existent_model") == 1.0


class TestVoteRecording:

    def test_record_votes_from_dicts(self, tracker: ModelCredibilityTracker):
        """Test recording votes passed as standard dictionaries."""
        votes = [
            {"model": "llama-3.1-8b", "action": "buy", "confidence": 0.85},
            {"model": "mistral-7b", "action": "sell", "confidence": 0.70},
            {"model": "deepseek-r1", "action": "hold", "confidence": 0.50},
        ]
        tracker.record_votes(trade_id=101, votes=votes)

        with tracker._connect() as conn:
            cursor = conn.execute(
                "SELECT model_name, action, confidence FROM model_votes WHERE trade_id = 101 ORDER BY model_name"
            )
            rows = cursor.fetchall()

        assert len(rows) == 3
        data = {r["model_name"]: (r["action"], r["confidence"]) for r in rows}
        assert data["llama-3.1-8b"] == ("buy", 0.85)
        assert data["mistral-7b"] == ("sell", 0.70)
        assert data["deepseek-r1"] == ("hold", 0.50)

    def test_record_votes_from_objects(self, tracker: ModelCredibilityTracker):
        """Test recording votes passed as object dataclasses."""
        votes = [
            DummyModelVote(model="gpt-4o", action="buy", confidence=0.92),
            DummyModelVote(model="claude-3-5", action="sell", confidence=0.88),
        ]
        tracker.record_votes(trade_id=202, votes=votes)

        with tracker._connect() as conn:
            cursor = conn.execute(
                "SELECT model_name, action, confidence FROM model_votes WHERE trade_id = 202"
            )
            rows = cursor.fetchall()

        assert len(rows) == 2


class TestCredibilityUpdates:

    def test_profitable_trade_boosts_weight(self, tracker: ModelCredibilityTracker):
        """A buy/sell vote on a profitable trade increases weight and correct_votes."""
        votes = [
            {"model": "model-bull", "action": "buy", "confidence": 0.90},
        ]
        tracker.record_votes(trade_id=1, votes=votes)
        tracker.update_credibility(trade_id=1, trade_profitable=True)

        # Base 1.0 + 0.05 = 1.05 -> decayed: 1.05 * 0.98 + 1.0 * 0.02 = 1.049
        expected_weight = (1.05 * 0.98) + (1.0 * 0.02)
        weight = tracker.get_weight("model-bull")
        assert pytest.approx(weight, abs=1e-4) == expected_weight

        stats = tracker.get_model_stats()
        assert len(stats) == 1
        assert stats[0]["model_name"] == "model-bull"
        assert stats[0]["correct_votes"] == 1
        assert stats[0]["total_votes"] == 1
        assert stats[0]["accuracy_pct"] == 100.0

    def test_unprofitable_trade_reduces_weight(self, tracker: ModelCredibilityTracker):
        """A buy/sell vote on an unprofitable trade reduces weight."""
        votes = [
            {"model": "model-bear", "action": "sell", "confidence": 0.80},
        ]
        tracker.record_votes(trade_id=2, votes=votes)
        tracker.update_credibility(trade_id=2, trade_profitable=False)

        # Base 1.0 - 0.05 = 0.95 -> decayed: 0.95 * 0.98 + 1.0 * 0.02 = 0.951
        expected_weight = (0.95 * 0.98) + (1.0 * 0.02)
        weight = tracker.get_weight("model-bear")
        assert pytest.approx(weight, abs=1e-4) == expected_weight

        stats = tracker.get_model_stats()
        assert stats[0]["correct_votes"] == 0
        assert stats[0]["total_votes"] == 1
        assert stats[0]["accuracy_pct"] == 0.0

    def test_hold_votes_not_evaluated(self, tracker: ModelCredibilityTracker):
        """Hold votes are skipped during outcome evaluation."""
        votes = [
            {"model": "model-neutral", "action": "hold", "confidence": 0.50},
        ]
        tracker.record_votes(trade_id=3, votes=votes)
        tracker.update_credibility(trade_id=3, trade_profitable=True)

        stats = tracker.get_model_stats()
        assert len(stats) == 0
        assert tracker.get_weight("model-neutral") == 1.0

    def test_weight_clamping_and_successive_updates(self, tracker: ModelCredibilityTracker):
        """Verify weights remain within [0.5, 1.5] even with extreme streaks."""
        # 30 consecutive profitable trades
        for tid in range(10, 40):
            tracker.record_votes(
                trade_id=tid,
                votes=[{"model": "model-winner", "action": "buy", "confidence": 0.9}],
            )
            tracker.update_credibility(trade_id=tid, trade_profitable=True)

        win_weight = tracker.get_weight("model-winner")
        assert win_weight <= 1.5
        assert win_weight > 1.0

        # 30 consecutive unprofitable trades
        for tid in range(50, 80):
            tracker.record_votes(
                trade_id=tid,
                votes=[{"model": "model-loser", "action": "buy", "confidence": 0.9}],
            )
            tracker.update_credibility(trade_id=tid, trade_profitable=False)

        loss_weight = tracker.get_weight("model-loser")
        assert loss_weight >= 0.5
        assert loss_weight < 1.0


class TestDisplaySummary:

    def test_display_summary_runs_cleanly(self, tracker: ModelCredibilityTracker):
        """Verify display_summary executes without raising exceptions."""
        tracker.display_summary()  # empty state

        tracker.record_votes(
            trade_id=1,
            votes=[
                {"model": "model-a", "action": "buy", "confidence": 0.8},
                {"model": "model-b", "action": "sell", "confidence": 0.7},
            ],
        )
        tracker.update_credibility(trade_id=1, trade_profitable=True)
        tracker.display_summary()  # populated state
