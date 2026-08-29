# Cache Me If You Can — Autonomous Options Alpha Agents
> **Alpaca AI Trading Agents Hackathon 2026** | **Team:** Cache Me If You Can | **Platform:** lablab.ai x Alpaca

[![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Alpaca API](https://img.shields.io/badge/Alpaca-Paper%20Trading%20API-FFD100?logo=alpaca&logoColor=black)](https://alpaca.markets)
[![MCP Protocol](https://img.shields.io/badge/Model%20Context%20Protocol-Alpaca%20MCP-00D4B2)](https://docs.alpaca.markets/us/docs/alpaca-mcp-server)
[![Tests](https://img.shields.io/badge/Tests-62%2F62%20Passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**Cache Me If You Can** is an autonomous multi-agent AI options trading system built on Alpaca's developer platform. It combines real-time market regime detection, mathematical options pricing (Black-Scholes and Greeks), a **3-model LLM voting council**, **Kelly Criterion position sizing**, **mid-price smart execution**, and rigorous 6-gate risk guardrails to generate consistent, risk-adjusted P&L on a dedicated **$100,000** paper account.

---

## What Makes This a Professional-Grade Agent

Most retail algorithmic traders lose money not because their signals are wrong, but because of four silent killers: **bad sizing, overpaying on spreads, cutting winners too early, and never learning from the past**. This system eliminates all four:

| Pro Feature | Problem Solved | Implementation |
|:---|:---|:---|
| **3-Model LLM Council** | False signals — 1 model can be confidently wrong | `meta-llama` + `Mistral` + `DeepSeek` vote in parallel; only high-consensus signals trade |
| **Kelly Criterion Sizing** | Fixed % sizing ignores edge — over-bets losers, under-bets winners | Quarter-Kelly per strategy, fed by historical win rate from trade journal |
| **Mid-Price Limit Orders** | Market orders on options pay full bid-ask spread (10–30% of premium) | `SmartExecutor` targets mid, steps price, market fallback — saves significant slippage |
| **Trailing Stop Loss** | Fixed stop cuts winners too early after they peak | Tracks high-water mark; exits at 25% pullback from peak, not 50% from cost |
| **Trade Journal (SQLite)** | Agents never learn from past trades | Every entry/exit logged; Kelly reads stats next session to improve sizing |

---

## System Architecture

```
                         +------------------------+
                         |   Market Data Stream   |
                         | (Alpaca Data API / VIX)|
                         +-----------+------------+
                                     |
                                     v
                       +---------------------------+
                       |     Master Orchestrator   |
                       |  (VIX Regime Detection)   |
                       +-------------+-------------+
                                     |
         +----------------+----------+---------+----------------+
         |                |                    |                |
         v                v                    v                v
 +---------------+ +---------------+  +-----------------+ +-----------+
 │ ThetaCollector│ │ IVCrushAgent  │  │  MomoBreakout   │ │HedgeAgent │
 │  (CSP Income) │ │ (Straddles)   │  │   (OTM Calls)   │ │(SPY Puts) │
 +-------+-------+ +-------+-------+  +--------+--------+ +-----+-----+
         |                 |                   |
         +---+  LLM Council (3 Models)  +------+
             |  Vote Gate per Signal    |
             +-----------+--------------+
                         |
         +---------------v---------------+
         |        Pro Trading Layer       |
         | KellySizer → SmartExecutor     |
         | TradeJournal → Trailing Stop   |
         +---------------+---------------+
                         |
                         v
         +-----------------------------+
         |     Risk Manager Engine     |
         |   (6 Hard Circuit Breakers) |
         +--------------+--------------+
                        | [Approved Orders]
                        v
         +-----------------------------+
         |   Alpaca Trading API        |
         |   (Paper Account)           |
         +-----------------------------+
```

---

## Multi-Agent Strategy Portfolio

| Agent | Strategy & Mechanics | Target Delta | DTE Range | Exit Conditions |
| :--- | :--- | :---: | :---: | :--- |
| **Theta Collector** | Sells cash-secured puts on liquid ETFs (`SPY`, `QQQ`, `IWM`, `GLD`, `PLTR`, `SOFI`) to collect time decay premium. Kelly-sized contracts, mid-price limit sell. | ~0.20 | 28-45 Days | 50% max profit OR 21 DTE time stop |
| **IV Crush Agent** | Sells ATM straddles 1-3 days prior to earnings to capture volatility collapse. Kelly-sized, limit fills on both legs. | 0.00 | 7-14 Days | 40% profit OR expired worthless |
| **Momo Breakout** | Buys cheap OTM calls on EMA 20/50 crossover + 2× volume surge. Kelly-sized, **trailing stop** from peak P&L. | ~0.30 | 28-45 Days | +100% gain OR trailing stop (25% pullback from peak) |
| **Hedge Agent** | Buys protective SPY puts when portfolio delta > +30 or VIX > 22. Always runs before other agents. | -0.20 | 28-45 Days | Closed when conditions normalise |

---

## 3-Model LLM Council (Voting Ensemble)

Every trade signal generated by rules-based conditions passes through a **three-model voting panel** before any order is submitted:

```
Signal Detected (EMA crossover, IVR threshold, etc.)
         │
         ▼
┌────────────────────────────────────────────────────┐
│              LLM Council Vote                       │
│                                                    │
│  Model 1: meta-llama/Llama-3.1-8B-Instruct         │
│  Model 2: mistralai/Mistral-7B-Instruct-v0.3       │
│  Model 3: deepseek-ai/DeepSeek-R1-Distill-Llama-8B │
│                                                    │
│  All 3 queried in parallel (ThreadPoolExecutor)    │
│                                                    │
│  Vote formula:                                     │
│  net_score = Σ(confidence_i × vote_i) / n_models  │
│  buy=+1, hold=0, sell=-1                           │
│                                                    │
│  Consensus if |net_score| ≥ 0.60                   │
└────────────────────────────────────────────────────┘
         │ consensus.agreed == True?
         ├── YES → Trade executes
         └── NO  → COUNCIL VETO (logged, skipped)
```

**Timeout safety:** A model that takes > 10s casts a neutral `hold` at 0 confidence — it doesn't block or trigger trades.  
**Bypass mode:** Set `COUNCIL_ENABLED=false` in `.env` to revert to pure rules-based trading instantly.

---

## Kelly Criterion Position Sizing

Position sizes are computed mathematically per strategy, not guessed as a fixed percentage:

```
Kelly fraction:  f* = (b·p - q) / b

where:
  b = avg_win / avg_loss  (reward-to-risk from trade history)
  p = historical win rate  (from TradeJournal SQLite)
  q = 1 - p

Actual size used:  f* × 0.25  (quarter-Kelly — industry standard)
Hard cap:          config.RISK.max_position_pct (5% of equity)
```

- **No trade history** (< 10 closed trades): conservative defaults (0.5–1.0% equity)
- **As the journal grows**: Kelly converges to the mathematically optimal size
- **Negative Kelly** (losing strategy): minimum 0.3% "skin-in-the-game" allocation

---

## 6-Gate Risk Management Framework

Every trade must pass through the stateful **Risk Manager** (`core/risk_manager.py`):

1. **VIX Kill Switch:** Halts all new open orders whenever VIX ≥ 35.0.
2. **Daily Loss Limit:** Freezes new trades if daily drawdown reaches -2% (-$2,000).
3. **Single Position Cap:** Maximum notional/margin per trade ≤ 5% of equity (≤$5,000).
4. **Total Options Exposure:** Aggregated options positions limited to ≤ 30% of equity (≤$30,000).
5. **Delta Neutrality Bounds:** Portfolio net delta enforced within [-50.0, +50.0].
6. **Binary Event Cooldown:** 2-hour trading blackout before/after high-impact events.

---

## Alpaca Platform & Partner Integration

- **Alpaca Trading API (`alpaca-py 0.44.0`):** Account queries, stock/options order submission, position monitoring. Supports both `MarketOrderRequest` and `LimitOrderRequest` for options.
- **Model Context Protocol (MCP) Bridge (`core/mcp_bridge.py`):** Connects AI agents with Alpaca tools using structured function calling for earnings analysis.
- **Featherless AI LLM Inference:** Serverless open-source model inference for all 3 council models and MCP reasoning — no GPU required.
- **Real-Time Market Data:** Alpaca Historical and Latest Data Clients for OHLCV bars, quotes, and option contract chains.
- **Alpaca CLI Runner (`cli/run_agent.sh`):** Shell script for automated sessions, structured JSON output, and cron jobs.

---

## Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon.git
cd Alpaca-AI-Trading-Agents-Hackathon
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
cp .env.example .env   # Linux/Mac
# or: copy .env.example .env   (Windows)
```

Edit `.env` with your credentials:
```env
# Required
ALPACA_API_KEY=your_alpaca_paper_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FEATHERLESS_API_KEY=your_featherless_key

# LLM Council (uses same Featherless key — no extra cost)
COUNCIL_ENABLED=true
COUNCIL_THRESHOLD=0.60

# Pro Trading (all optional — safe defaults apply)
USE_LIMIT_ORDERS=true
TRAILING_STOP_PCT=0.25
KELLY_FRACTION=0.25
```

### 3. Usage & Modes

#### Check Account Status & Open Positions
```bash
python -X utf8 main.py status
```

#### Execute an Autonomous Trading Session
```bash
python -X utf8 main.py run
```

#### Start Scheduled Daemon (Runs automatically at market open)
```bash
python -X utf8 main.py schedule
```

#### Launch Real-Time Streamlit Dashboard
```bash
streamlit run dashboard/monitor.py
```
> Open your browser at **http://localhost:8501** to monitor live equity, P&L gauge, open option positions, and risk metrics.

#### Run via Alpaca CLI Runner
```bash
bash cli/run_agent.sh status
bash cli/run_agent.sh run
```

---

## Automated Testing

```bash
python -m pytest tests/ -v
```

```text
============================= test session starts =============================
collected 62 items

tests/test_kelly_sizer.py::TestFallback::test_zero_trades_uses_default_theta  PASSED
tests/test_kelly_sizer.py::TestKellyMath::test_kelly_formula_correctness      PASSED
tests/test_kelly_sizer.py::TestHardCap::test_kelly_capped_by_max_position_pct PASSED
...
tests/test_llm_council.py::TestVotingMath::test_tally_unanimous_buy           PASSED
tests/test_llm_council.py::TestJsonParsing::test_parse_json_embedded_in_text  PASSED
tests/test_llm_council.py::TestAutoApprove::test_no_api_key_auto_approves     PASSED
...
tests/test_options_pricer.py::test_black_scholes_call_price_positive          PASSED
tests/test_options_pricer.py::test_put_call_parity                            PASSED
tests/test_options_pricer.py::test_greeks_call                                PASSED
...
tests/test_risk_manager.py::test_vix_kill_switch_blocks_trade                 PASSED
tests/test_risk_manager.py::test_daily_loss_limit_blocks_at_threshold         PASSED
tests/test_risk_manager.py::test_delta_within_range_passes                    PASSED
...
============================= 62 passed in 5.70s ==============================
```

---

## Project Structure

```
Alpaca-AI-Trading-Agents-Hackathon/
├── agents/
│   ├── orchestrator.py        # Master controller: regime detection & capital routing
│   ├── theta_collector.py     # Cash-Secured Put (CSP) income agent + Kelly/Journal
│   ├── iv_crush_agent.py      # Pre-earnings ATM straddle agent + Kelly/Journal
│   ├── momo_breakout.py       # EMA cross + volume breakout agent + trailing stop
│   └── hedge_agent.py         # Dynamic SPY protective put portfolio hedge
├── core/
│   ├── alpaca_client.py       # Alpaca Trading & Market Data API wrapper
│   ├── mcp_bridge.py          # Alpaca MCP Server + Featherless LLM bridge
│   ├── risk_manager.py        # 6-gate risk management engine
│   ├── options_pricer.py      # Black-Scholes pricing, Greeks & IV solver
│   ├── market_data.py         # VIX tracking, EMA crossover & volume surge
│   ├── llm_council.py         # 3-model parallel voting ensemble
│   ├── signal_enhancer.py     # Market context builder for LLM prompts
│   ├── kelly_sizer.py         # Kelly Criterion position sizing
│   ├── trade_journal.py       # SQLite trade performance journal
│   └── smart_executor.py      # Mid-price limit order execution
├── dashboard/
│   └── monitor.py             # Interactive Streamlit live trading dashboard
├── cli/
│   └── run_agent.sh           # Alpaca CLI session runner & cron automation
├── tests/
│   ├── test_risk_manager.py   # Unit tests for all 6 risk gates
│   ├── test_options_pricer.py # Unit tests for Black-Scholes, Greeks & IV
│   ├── test_llm_council.py    # Unit tests for LLM voting math
│   └── test_kelly_sizer.py    # Unit tests for Kelly Criterion
├── logs/
│   └── trading.db             # SQLite trade journal (auto-created on first run)
├── config.py                  # Global parameters, universe & execution config
├── main.py                    # Multi-mode CLI entry point
├── requirements.txt           # Project dependencies
└── .env.example               # Environment variable template
```

---

## Session Output (Example)

```
───────── Cache Me If You Can — Master Orchestrator ─────────
OK AlpacaClient initialised (paper trading)
TradeJournal initialised | db=logs/trading.db
KellySizer initialised | fraction=0.25 | min_trades=10
SmartExecutor initialised | limit_orders=True | timeout=30s | steps=3
LLMCouncil init | models=3 | threshold=0.60

Market Regime: RISK_ON | VIX=17.4

[SIGNAL] Breakout detected: NVDA | EMA=bullish | surge=2.8x
Council: BUY | score=+0.763 | agreed=True
  model-a (Llama): buy  @ 0.89 — "Strong EMA crossover with institutional volume"
  model-b (Mistral): buy @ 0.82 — "Momentum confirmed, VIX benign"
  model-c (DeepSeek): buy @ 0.71 — "Pattern valid, entry justified"
[Kelly][momo] p=68% | b=2.31:1 | f*=0.418 | ¼-Kelly=0.105 → $1,045
[Executor] NVDA241220C00890000: bid=$2.40 ask=$2.70 spread=$0.30 → mid=$2.55
[FILLED] Limit fill @ $2.55 (step 0) — saved $75
[FILLED] MomoBreakout BOUGHT CALL: NVDA $890 exp=2024-12-20 x4 | Kelly-sized

───────────── Session Summary ─────────────
 Metric              Value
 Regime              RISK_ON
 VIX                 17.4
 Equity              $100,423.50
 Daily P&L           +$423.50 (+0.42%)
 Council Models      3 models | threshold=0.60
 Trades Executed     3
 Trades Vetoed       1

Strategy Performance (All-Time)
 Strategy   Trades  Win Rate  Avg Win  Avg Loss  Sharpe  Total P&L
 THETA          14    78.6%    52.3%     21.1%    1.84   +$3,240
 MOMO            6    66.7%    89.4%     42.3%    1.12   +$1,180
 IV_CRUSH        4    75.0%    38.2%     18.5%    1.45   +$860
```

---

## Hackathon Details

- **Event:** Alpaca AI Trading Agents Hackathon (lablab.ai x Alpaca)
- **Dates:** August 28 – September 4, 2026
- **Team Name:** Cache Me If You Can
- **Dedicated Account ID:** `9e62f22b-a0cb-49c3-99d0-10a1fcd2c9a3`
- **Initial Paper Balance:** $100,000.00 USD

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
