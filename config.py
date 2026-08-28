"""
config.py — Central configuration for Cache Me trading system.

Loads from environment variables (.env file).
All agent modules import from here — never hardcode credentials elsewhere.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# Alpaca API Credentials
# ──────────────────────────────────────────────
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL: str = "https://data.alpaca.markets"

# ──────────────────────────────────────────────
# Featherless AI (LLM Reasoning)
# ──────────────────────────────────────────────
FEATHERLESS_API_KEY: str = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL: str = os.getenv(
    "FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"
)
FEATHERLESS_MODEL: str = os.getenv(
    "FEATHERLESS_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
)


# ──────────────────────────────────────────────
# Risk Parameters
# ──────────────────────────────────────────────
@dataclass
class RiskConfig:
    """All risk gate thresholds. Modify here or via env vars."""

    # Position sizing
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
    max_options_exposure_pct: float = float(
        os.getenv("MAX_OPTIONS_EXPOSURE_PCT", "0.30")
    )

    # Loss limits
    daily_loss_limit_pct: float = float(
        os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02")
    )

    # Market regime
    vix_kill_switch: float = float(os.getenv("VIX_KILL_SWITCH", "35"))

    # Portfolio Greeks limits
    max_portfolio_delta: float = 50.0
    min_portfolio_delta: float = -50.0

    # Earnings event cooldown (minutes)
    earnings_cooldown_minutes: int = 120

    # Account starting balance (hackathon requirement: $100,000)
    starting_balance: float = float(
        os.getenv("ACCOUNT_STARTING_BALANCE", "100000")
    )


RISK = RiskConfig()


# ──────────────────────────────────────────────
# Trading Universe
# ──────────────────────────────────────────────
@dataclass
class UniverseConfig:
    """Symbols each agent watches."""

    # Theta Collector — liquid, high-IV ETFs for CSP selling
    theta_symbols: list = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "GLD"]
    )

    # Hedge Agent — portfolio protection
    hedge_symbol: str = "SPY"

    # Momo Breakout — scanned dynamically, seeded with these
    momo_watchlist: list = field(
        default_factory=lambda: ["NVDA", "TSLA", "AAPL", "META", "AMZN"]
    )

    # IV Crush — populated dynamically from earnings calendar
    iv_crush_max_positions: int = 3


UNIVERSE = UniverseConfig()


# ──────────────────────────────────────────────
# Market Hours (ET)
# ──────────────────────────────────────────────
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
PRE_MARKET_SCAN_ET = "09:00"   # agent scans + decision window


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DB_PATH: str = "logs/trading.db"   # SQLite log database
