"""
tests/test_options_pricer.py — Unit tests for Black-Scholes and Options Greeks.
"""

import pytest
import math
from core.options_pricer import black_scholes_price, greeks, iv_from_price, iv_rank


def test_black_scholes_call_price_positive():
    # Spot=100, Strike=100, T=30/365, r=0.05, sigma=0.20
    price = black_scholes_price(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, option_type="call")
    assert price > 0
    # Approx ATM 30-day option price should be around 2-3 USD
    assert 1.5 < price < 4.0


def test_black_scholes_put_price_positive():
    price = black_scholes_price(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, option_type="put")
    assert price > 0
    assert 1.5 < price < 4.0


def test_put_call_parity():
    # C - P = S - K * exp(-r*T)
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert pytest.approx(lhs, abs=1e-4) == rhs


def test_expired_options():
    # ITM call at expiry
    assert black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="call") == 10.0
    # OTM call at expiry
    assert black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="call") == 0.0
    # ITM put at expiry
    assert black_scholes_price(S=90, K=100, T=0, r=0.05, sigma=0.20, option_type="put") == 10.0
    # OTM put at expiry
    assert black_scholes_price(S=110, K=100, T=0, r=0.05, sigma=0.20, option_type="put") == 0.0


def test_greeks_call():
    g = greeks(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, option_type="call")
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.theta < 0.0  # Time decay hurts long options
    assert g.vega > 0.0


def test_greeks_put():
    g = greeks(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, option_type="put")
    assert -1.0 < g.delta < 0.0
    assert g.gamma > 0.0
    assert g.vega > 0.0


def test_iv_solver():
    S, K, T, r, target_iv = 100.0, 100.0, 30 / 365, 0.05, 0.25
    market_price = black_scholes_price(S, K, T, r, target_iv, "call")
    computed_iv = iv_from_price(market_price, S, K, T, r, "call")
    assert pytest.approx(computed_iv, abs=1e-4) == target_iv


def test_iv_rank():
    assert pytest.approx(iv_rank(current_iv=0.30, iv_52w_low=0.10, iv_52w_high=0.50)) == 50.0
    assert pytest.approx(iv_rank(current_iv=0.10, iv_52w_low=0.10, iv_52w_high=0.50)) == 0.0
    assert pytest.approx(iv_rank(current_iv=0.50, iv_52w_low=0.10, iv_52w_high=0.50)) == 100.0
