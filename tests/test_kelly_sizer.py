"""
tests/test_kelly_sizer.py — Unit tests for Kelly Criterion position sizer.

Tests the Kelly math, fallback logic, hard cap, and edge cases
all offline using a mock TradeJournal.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.kelly_sizer import KellySizer


# ── Helpers ──────────────────────────────────────────────────────

def _make_journal(
    n_trades: int = 0,
    win_rate: float = 0.0,
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
) -> MagicMock:
    """Create a mock TradeJournal that returns fixed stats."""
    j = MagicMock()
    j.get_strategy_stats.return_value = {
        "strategy": "theta",
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": 0.0,
        "sharpe": 0.0,
    }
    return j


def _make_sizer(
    n_trades: int = 0,
    win_rate: float = 0.0,
    avg_win: float = 0.0,
    avg_loss: float = 0.0,
    kelly_fraction: float = 0.25,
) -> KellySizer:
    j = _make_journal(n_trades, win_rate, avg_win, avg_loss)
    sizer = KellySizer.__new__(KellySizer)
    sizer.journal = j
    sizer.kelly_fraction = kelly_fraction
    return sizer


# ════════════════════════════════════════════════════════════════
# 1. Fallback (insufficient history)
# ════════════════════════════════════════════════════════════════

class TestFallback:

    def test_zero_trades_uses_default_theta(self):
        """0 trades → default 1.0% of equity for theta."""
        sizer = _make_sizer(n_trades=0)
        size = sizer.get_position_size("theta", equity=100_000)
        # Default for theta = 1.0% = $1,000
        assert abs(size - 1_000) < 1

    def test_zero_trades_uses_default_momo(self):
        """0 trades → default 0.6% of equity for momo."""
        sizer = _make_sizer(n_trades=0)
        size = sizer.get_position_size("momo", equity=100_000)
        assert abs(size - 600) < 1

    def test_below_min_trades_uses_default(self):
        """9 trades (< 10 minimum) → still uses default."""
        sizer = _make_sizer(n_trades=9, win_rate=0.80, avg_win=0.50, avg_loss=0.25)
        size = sizer.get_position_size("theta", equity=100_000)
        assert abs(size - 1_000) < 1

    def test_exactly_min_trades_uses_kelly(self):
        """Exactly 10 trades → Kelly is now trusted."""
        sizer = _make_sizer(n_trades=10, win_rate=0.70, avg_win=0.50, avg_loss=0.30)
        size = sizer.get_position_size("theta", equity=100_000)
        # Kelly: b=0.50/0.30≈1.667, p=0.70, q=0.30
        # f* = (1.667*0.70 - 0.30)/1.667 = (1.167-0.30)/1.667 ≈ 0.520
        # ¼-Kelly: 0.520*0.25 ≈ 0.130 → $13,000
        # But capped at max_position_pct (default 5% = $5,000)
        assert size <= 100_000 * 0.05  # hard cap
        assert size > 1_000  # should be more than default


# ════════════════════════════════════════════════════════════════
# 2. Kelly Math
# ════════════════════════════════════════════════════════════════

class TestKellyMath:

    def test_positive_kelly_scales_with_edge(self):
        """Higher win rate → larger Kelly fraction → larger position."""
        low_edge = _make_sizer(n_trades=20, win_rate=0.52, avg_win=0.20, avg_loss=0.20)
        high_edge = _make_sizer(n_trades=20, win_rate=0.75, avg_win=0.30, avg_loss=0.20)

        size_low = low_edge.get_position_size("theta", equity=100_000)
        size_high = high_edge.get_position_size("theta", equity=100_000)

        assert size_high > size_low

    def test_breakeven_edge_gives_minimum_size(self):
        """Win rate at breakeven (Kelly = 0) → minimum position, not zero."""
        # p = 0.5, b = 1.0 → f* = (1.0*0.5 - 0.5)/1.0 = 0.0
        sizer = _make_sizer(n_trades=20, win_rate=0.50, avg_win=0.20, avg_loss=0.20)
        size = sizer.get_position_size("theta", equity=100_000)
        # Raw Kelly = 0 → should fall to the minimum 0.3%
        assert abs(size - 300) < 1  # 0.3% of 100_000

    def test_negative_edge_gives_minimum_size(self):
        """Losing strategy → minimum position (agent won't stop, just risk minimum)."""
        sizer = _make_sizer(n_trades=20, win_rate=0.30, avg_win=0.20, avg_loss=0.40)
        # b = 0.5, p = 0.3 → f* = (0.5*0.3 - 0.7)/0.5 = (0.15-0.70)/0.5 = negative
        size = sizer.get_position_size("theta", equity=100_000)
        assert size == pytest.approx(300, abs=1)  # 0.3% minimum

    def test_kelly_formula_correctness(self):
        """
        Verify exact Kelly formula with known inputs.
        p=0.60, avg_win=0.50, avg_loss=0.25
        b = 0.50/0.25 = 2.0
        f* = (2.0*0.60 - 0.40)/2.0 = (1.20-0.40)/2.0 = 0.40
        ¼-Kelly = 0.40 * 0.25 = 0.10 → 10% of equity = $10,000
        Capped by max_position_pct = 5% = $5,000
        """
        sizer = _make_sizer(n_trades=50, win_rate=0.60, avg_win=0.50, avg_loss=0.25)
        equity = 100_000
        size = sizer.get_position_size("theta", equity=equity, override_max_pct=0.20)
        # With 20% cap override, should get full 10% = $10,000
        assert abs(size - 10_000) < 100

    def test_half_kelly_vs_quarter_kelly(self):
        """Half-Kelly should produce exactly 2× the position of quarter-Kelly."""
        quarter = _make_sizer(n_trades=20, win_rate=0.65, avg_win=0.40, avg_loss=0.30, kelly_fraction=0.25)
        half = _make_sizer(n_trades=20, win_rate=0.65, avg_win=0.40, avg_loss=0.30, kelly_fraction=0.50)

        q_size = quarter.get_position_size("theta", equity=100_000, override_max_pct=1.0)
        h_size = half.get_position_size("theta", equity=100_000, override_max_pct=1.0)

        assert abs(h_size / q_size - 2.0) < 0.01


# ════════════════════════════════════════════════════════════════
# 3. Hard Cap
# ════════════════════════════════════════════════════════════════

class TestHardCap:

    def test_kelly_capped_by_max_position_pct(self):
        """Even very high Kelly fraction must not exceed max_position_pct."""
        # Very high edge: p=0.90, b=3.0 → f* ≈ 0.86, ¼-Kelly = 0.22 = 22%
        sizer = _make_sizer(n_trades=100, win_rate=0.90, avg_win=0.60, avg_loss=0.20)
        size = sizer.get_position_size("theta", equity=100_000, override_max_pct=0.05)
        assert size <= 100_000 * 0.05 + 1  # must not exceed 5% cap

    def test_override_max_pct_is_respected(self):
        """override_max_pct should set a different cap."""
        sizer = _make_sizer(n_trades=100, win_rate=0.80, avg_win=0.40, avg_loss=0.20)
        size_5pct = sizer.get_position_size("theta", equity=100_000, override_max_pct=0.05)
        size_10pct = sizer.get_position_size("theta", equity=100_000, override_max_pct=0.10)
        assert size_10pct >= size_5pct  # relaxed cap allows more


# ════════════════════════════════════════════════════════════════
# 4. Contract Count Helper
# ════════════════════════════════════════════════════════════════

class TestContractCount:

    def test_contract_count_min_one(self):
        """Even with tiny position size, return at least 1 contract."""
        sizer = _make_sizer(n_trades=0)  # uses default
        n = sizer.get_contract_count("theta", equity=1000, premium_per_contract=5000)
        assert n >= 1

    def test_contract_count_scales_with_size(self):
        """More equity → more contracts."""
        sizer = _make_sizer(n_trades=50, win_rate=0.65, avg_win=0.40, avg_loss=0.25)
        n_small = sizer.get_contract_count("theta", equity=10_000, premium_per_contract=500)
        n_large = sizer.get_contract_count("theta", equity=100_000, premium_per_contract=500)
        assert n_large >= n_small

    def test_contract_count_zero_premium_returns_one(self):
        """Zero or negative premium → fallback to 1 contract."""
        sizer = _make_sizer(n_trades=50, win_rate=0.65, avg_win=0.40, avg_loss=0.25)
        n = sizer.get_contract_count("theta", equity=100_000, premium_per_contract=0)
        assert n == 1
