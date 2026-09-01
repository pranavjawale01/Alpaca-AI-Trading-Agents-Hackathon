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
# LLM / AI Reasoning (NVIDIA, Hugging Face, Groq, OpenRouter, Featherless, OpenAI)
# Free options:
#   - NVIDIA NIM (https://build.nvidia.com) -> free 1,000 credits, ultra-fast
#   - Hugging Face (https://huggingface.co/settings/tokens) -> free serverless inference API
#   - Groq (https://console.groq.com) -> free tier 30 RPM / 14,400 RPD
#   - OpenRouter (https://openrouter.ai) -> free :free model endpoints
# ──────────────────────────────────────────────
FEATHERLESS_API_KEY: str = (
    os.getenv("NVIDIA_API_KEY")
    or os.getenv("HUGGINGFACE_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("HF_API_KEY")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("FEATHERLESS_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)

_default_base_url = "https://api.featherless.ai/v1"
_default_model = "meta-llama/Llama-3.1-8B-Instruct"

if os.getenv("NVIDIA_API_KEY") or FEATHERLESS_API_KEY.startswith("nvapi-"):
    _default_base_url = "https://integrate.api.nvidia.com/v1"
    _default_model = "meta/llama-3.1-8b-instruct"
elif os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or FEATHERLESS_API_KEY.startswith("hf_"):
    _default_base_url = "https://router.huggingface.co/hf-inference/v1"
    _default_model = "meta-llama/Llama-3.1-8B-Instruct"
elif os.getenv("GROQ_API_KEY") or FEATHERLESS_API_KEY.startswith("gsk_"):
    _default_base_url = "https://api.groq.com/openai/v1"
    _default_model = "llama-3.1-8b-instant"
elif os.getenv("OPENROUTER_API_KEY") or FEATHERLESS_API_KEY.startswith("sk-or-"):
    _default_base_url = "https://openrouter.ai/api/v1"
    _default_model = "meta-llama/llama-3.1-8b-instruct:free"

FEATHERLESS_BASE_URL: str = os.getenv("FEATHERLESS_BASE_URL", _default_base_url)
FEATHERLESS_MODEL: str = os.getenv("FEATHERLESS_MODEL", _default_model)


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

    # Theta Collector — liquid, high-IV ETFs and stocks for CSP selling
    theta_symbols: list = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "F"]
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
# LLM Council (3-Model Voting Ensemble)
# ──────────────────────────────────────────────
@dataclass
class CouncilConfig:
    """
    Configuration for the 3-model LLM voting council.

    All three models are queried in parallel via the Featherless AI API
    (OpenAI-compatible). Set COUNCIL_ENABLED=false to bypass voting
    and fall back to pure rules-based signals (useful for backtesting).
    """

    # The three models that form the council.
    # Automatically chooses free models tailored to the active provider.
    models: list = field(
        default_factory=lambda: (
            [
                os.getenv("COUNCIL_MODEL_1", "meta/llama-3.1-8b-instruct"),
                os.getenv("COUNCIL_MODEL_2", "mistralai/mistral-7b-instruct-v0.3"),
                os.getenv("COUNCIL_MODEL_3", "deepseek-ai/deepseek-r1"),
            ]
            if ("nvidia.com" in FEATHERLESS_BASE_URL or FEATHERLESS_API_KEY.startswith("nvapi-"))
            else [
                os.getenv("COUNCIL_MODEL_1", "meta-llama/Llama-3.1-8B-Instruct"),
                os.getenv("COUNCIL_MODEL_2", "mistralai/Mistral-7B-Instruct-v0.2"),
                os.getenv("COUNCIL_MODEL_3", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
            ]
            if ("huggingface.co" in FEATHERLESS_BASE_URL or FEATHERLESS_API_KEY.startswith("hf_"))
            else [
                os.getenv("COUNCIL_MODEL_1", "llama-3.1-8b-instant"),
                os.getenv("COUNCIL_MODEL_2", "llama-3.3-70b-versatile"),
                os.getenv("COUNCIL_MODEL_3", "deepseek-r1-distill-llama-70b"),
            ]
            if ("groq.com" in FEATHERLESS_BASE_URL or FEATHERLESS_API_KEY.startswith("gsk_"))
            else [
                os.getenv("COUNCIL_MODEL_1", "meta-llama/llama-3.1-8b-instruct:free"),
                os.getenv("COUNCIL_MODEL_2", "mistralai/mistral-7b-instruct:free"),
                os.getenv("COUNCIL_MODEL_3", "qwen/qwen-2.5-7b-instruct:free"),
            ]
            if ("openrouter.ai" in FEATHERLESS_BASE_URL or FEATHERLESS_API_KEY.startswith("sk-or-"))
            else [
                os.getenv("COUNCIL_MODEL_1", "meta-llama/Llama-3.1-8B-Instruct"),
                os.getenv("COUNCIL_MODEL_2", "mistralai/Mistral-7B-Instruct-v0.3"),
                os.getenv("COUNCIL_MODEL_3", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            ]
        )
    )

    # Minimum |weighted score| required to act on a signal.
    # Range: 0.0 (always act) to 1.0 (unanimous perfect confidence required).
    # 0.60 = two confident models override one uncertain one.
    consensus_threshold: float = float(
        os.getenv("COUNCIL_THRESHOLD", "0.60")
    )

    # Set to False to disable the council and run pure rules-based signals.
    # Useful for backtesting or when API is unavailable.
    enabled: bool = os.getenv("COUNCIL_ENABLED", "true").lower() == "true"

    # Per-model query timeout in seconds (models queried in parallel).
    timeout_seconds: float = float(os.getenv("COUNCIL_TIMEOUT", "25.0"))


COUNCIL = CouncilConfig()


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
LOG_DB_PATH: str = os.getenv("LOG_DB_PATH", "logs/trading.db")  # SQLite log database


# ──────────────────────────────────────────────
# Execution & Pro Trading Parameters
# ──────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    """
    Controls professional-grade order execution and position management.
    All parameters are overridable via environment variables.
    """

    # ── Smart Limit Order Execution ──────────────────────────────
    # Use limit orders at mid-price instead of market orders for options.
    # Saves bid-ask spread on every trade (typically 3–15% of premium).
    use_limit_orders: bool = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"

    # Seconds to wait for a limit order fill before stepping price.
    limit_order_timeout_seconds: float = float(
        os.getenv("LIMIT_ORDER_TIMEOUT", "30.0")
    )

    # Number of times to step the limit price toward aggressive before
    # giving up and converting to a market order.
    limit_price_aggression_steps: int = int(
        os.getenv("LIMIT_AGGRESSION_STEPS", "3")
    )

    # ── Trailing Stop Loss (Momo calls) ──────────────────────────
    # Exit a long call when it pulls back this fraction from its peak P&L.
    # E.g. 0.25 = if trade was up 100% and falls back to 75%, close it.
    trailing_stop_pct: float = float(
        os.getenv("TRAILING_STOP_PCT", "0.25")
    )

    # ── Kelly Criterion Position Sizing ──────────────────────────
    # Fraction of full Kelly to use (industry standard: 0.25 = quarter-Kelly).
    # Higher = more aggressive growth but more volatile equity curve.
    kelly_fraction: float = float(
        os.getenv("KELLY_FRACTION", "0.25")
    )

    # Minimum closed trades required before Kelly is trusted.
    # Below this, conservative default sizes are used.
    kelly_min_trades: int = int(
        os.getenv("KELLY_MIN_TRADES", "10")
    )


EXECUTION = ExecutionConfig()


# ──────────────────────────────────────────────
# Hybrid Greedy-Voting Council
# ──────────────────────────────────────────────
@dataclass
class HybridConfig:
    """
    Configuration for the hybrid greedy-voting council upgrade.

    Controls regime-adaptive consensus thresholds, pilot position sizing,
    opportunity multiplier caps, and the Streamlit auto-pilot session interval.
    """

    # Enable/disable greedy opportunity scoring entirely
    greedy_enabled: bool = os.getenv("GREEDY_ENABLED", "true").lower() == "true"

    # Max Kelly scale-up from opportunity scorer
    # At 2.0: quarter-Kelly (0.25) can scale to half-Kelly (0.50) on prime setups
    max_greedy_multiplier: float = float(
        os.getenv("MAX_GREEDY_MULTIPLIER", "2.0")
    )

    # Pilot position size multiplier (weak consensus trades)
    pilot_size_multiplier: float = float(
        os.getenv("PILOT_SIZE_MULTIPLIER", "0.40")
    )

    # Auto-pilot: minutes between trading sessions during market hours
    session_interval_minutes: int = int(
        os.getenv("SESSION_INTERVAL_MINUTES", "10")
    )

    # Max effective Kelly fraction after all multipliers
    max_kelly_fraction: float = float(
        os.getenv("MAX_KELLY_FRACTION", "0.50")
    )


HYBRID = HybridConfig()
