"""
tests/test_risk_manager.py — Unit tests for the Risk Manager.

Run: pytest tests/ -v
"""

import pytest
from datetime import datetime, timezone, timedelta

from core.risk_manager import RiskManager, RiskViolation


@pytest.fixture
def rm():
    """Fresh RiskManager with $100,000 equity."""
    return RiskManager(equity=100_000)


# ── VIX Kill Switch ────────────────────────────────────────────

def test_vix_kill_switch_blocks_trade(rm):
    rm.update_vix(36.0)  # above threshold
    with pytest.raises(RiskViolation, match="VIX kill switch"):
        rm.approve_order("SPY", order_value=1000)


def test_vix_below_threshold_passes(rm):
    rm.update_vix(20.0)
    # Should not raise
    rm.approve_order("SPY", order_value=1000)


# ── Daily Loss Limit ───────────────────────────────────────────

def test_daily_loss_limit_blocks_at_threshold(rm):
    rm.update_daily_pnl(-2001)  # just over -$2,000 (2% of $100k)
    with pytest.raises(RiskViolation, match="Daily loss limit"):
        rm.approve_order("QQQ", order_value=1000)


def test_daily_loss_at_limit_exactly_blocks(rm):
    rm.update_daily_pnl(-2000)  # exactly at limit
    with pytest.raises(RiskViolation, match="Daily loss limit"):
        rm.approve_order("QQQ", order_value=1000)


def test_daily_loss_below_limit_passes(rm):
    rm.update_daily_pnl(-1999)  # just below limit
    rm.approve_order("QQQ", order_value=1000)  # should pass


# ── Position Size ──────────────────────────────────────────────

def test_position_size_exceeds_5pct_blocks(rm):
    with pytest.raises(RiskViolation, match="Position too large"):
        rm.approve_order("AAPL", order_value=6000)  # 6% > 5%


def test_position_size_at_limit_blocks(rm):
    with pytest.raises(RiskViolation, match="Position too large"):
        rm.approve_order("AAPL", order_value=5001)


def test_position_size_within_limit_passes(rm):
    rm.approve_order("AAPL", order_value=5000)  # exactly 5%


# ── Options Exposure ───────────────────────────────────────────

def test_options_exposure_exceeds_30pct_blocks(rm):
    # Pre-set exposure at $27k; adding $4k order = $31k > 30% limit ($30k)
    # $4k is within the 5% position size limit ($5k), so only options gate fires
    rm.update_options_exposure(27_000)
    with pytest.raises(RiskViolation, match="Options exposure"):
        rm.approve_order("SPY_PUT", order_value=4_000, is_option=True)



def test_options_exposure_within_limit_passes(rm):
    rm.update_options_exposure(25_000)
    rm.approve_order("SPY_PUT", order_value=4000, is_option=True)  # stays at 29%


# ── Delta Limits ───────────────────────────────────────────────

def test_delta_too_positive_blocks(rm):
    rm.update_portfolio_delta(45.0)
    with pytest.raises(RiskViolation, match="Delta out of bounds"):
        rm.approve_order("AAPL", order_value=1000, delta_impact=10.0)  # would be +55


def test_delta_too_negative_blocks(rm):
    rm.update_portfolio_delta(-45.0)
    with pytest.raises(RiskViolation, match="Delta out of bounds"):
        rm.approve_order("AAPL", order_value=1000, delta_impact=-10.0)  # would be -55


def test_delta_within_range_passes(rm):
    rm.update_portfolio_delta(0.0)
    rm.approve_order("AAPL", order_value=1000, delta_impact=30.0)  # +30 delta, fine


# ── Earnings Cooldown ──────────────────────────────────────────

def test_earnings_cooldown_blocks_near_event(rm):
    now = datetime.now(timezone.utc)
    # Earnings in 30 minutes — within 2h cooldown
    rm.add_earnings_event(now + timedelta(minutes=30))
    with pytest.raises(RiskViolation, match="Earnings cooldown"):
        rm.approve_order("AAPL", order_value=1000)


def test_earnings_cooldown_passes_far_from_event(rm):
    now = datetime.now(timezone.utc)
    # Earnings in 5 hours — outside cooldown window
    rm.add_earnings_event(now + timedelta(hours=5))
    rm.approve_order("AAPL", order_value=1000)  # should pass


def test_earnings_cooldown_skipped_for_hedge(rm):
    now = datetime.now(timezone.utc)
    rm.add_earnings_event(now + timedelta(minutes=30))
    # With skip_earnings_check=True, should not raise
    rm.approve_order("SPY_PUT", order_value=1000, skip_earnings_check=True)


# ── Summary ────────────────────────────────────────────────────

def test_summary_returns_expected_keys(rm):
    summary = rm.summary()
    assert "equity" in summary
    assert "daily_pnl" in summary
    assert "portfolio_delta" in summary
    assert "options_exposure" in summary
    assert "vix" in summary


def test_summary_pnl_pct_calculation(rm):
    rm.update_daily_pnl(1000)
    summary = rm.summary()
    assert abs(summary["daily_pnl_pct"] - 1.0) < 0.01  # 1% gain
