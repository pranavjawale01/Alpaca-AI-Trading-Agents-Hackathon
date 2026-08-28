"""
core/options_pricer.py — Black-Scholes options pricing and Greeks.

Provides:
  - black_scholes_price(): theoretical option price
  - greeks(): delta, gamma, theta, vega, rho
  - iv_from_price(): implied volatility via bisection
  - iv_rank(): IVR calculation for premium-selling signals
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from scipy.stats import norm


# ──────────────────────────────────────────────
# Core Black-Scholes
# ──────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 for Black-Scholes."""
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    """Compute d2 for Black-Scholes."""
    return d1 - sigma * math.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes theoretical option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years (e.g. 30 days = 30/365)
        r: Risk-free rate (e.g. 0.05 for 5%)
        sigma: Implied volatility (e.g. 0.20 for 20%)
        option_type: 'call' or 'put'

    Returns:
        Theoretical option price
    """
    if T <= 0:
        # At expiration, intrinsic value only
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ──────────────────────────────────────────────
# Greeks
# ──────────────────────────────────────────────

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float   # per day (not per year)
    vega: float    # per 1% vol move
    rho: float


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """
    Compute all first-order Greeks for an option.

    Returns:
        Greeks dataclass with delta, gamma, theta (per day), vega, rho
    """
    if T <= 0:
        return Greeks(
            delta=1.0 if (option_type == "call" and S > K) else 0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
        )

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    nd1 = norm.pdf(d1)
    sqrt_T = math.sqrt(T)

    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for calls and puts)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Theta (annualised → convert to per-day)
    if option_type == "call":
        theta_annual = (
            -(S * nd1 * sigma) / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
    else:
        theta_annual = (
            -(S * nd1 * sigma) / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )
    theta = theta_annual / 365

    # Vega (per 1% move in vol)
    vega = S * nd1 * sqrt_T * 0.01

    # Rho (per 1% move in rates)
    if option_type == "call":
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) * 0.01

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


# ──────────────────────────────────────────────
# Implied Volatility (Bisection Method)
# ──────────────────────────────────────────────

def iv_from_price(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 500,
) -> float:
    """
    Compute implied volatility from a market price using bisection.

    Returns:
        Implied volatility as a decimal (e.g. 0.25 = 25%)
        Returns -1.0 if IV cannot be found.
    """
    low, high = 1e-6, 5.0  # vol bounds: near 0% to 500%

    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid

    return -1.0  # failed to converge


# ──────────────────────────────────────────────
# IV Rank (IVR)
# ──────────────────────────────────────────────

def iv_rank(current_iv: float, iv_52w_low: float, iv_52w_high: float) -> float:
    """
    Compute IV Rank (IVR) as a percentage.

    IVR = (current_iv - 52w_low) / (52w_high - 52w_low) * 100

    IVR > 50 → premium-selling environment (sell options)
    IVR < 30 → premium-buying environment (buy options)

    Returns:
        IVR in range [0, 100]
    """
    if iv_52w_high == iv_52w_low:
        return 50.0
    return max(0.0, min(100.0, (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100))
