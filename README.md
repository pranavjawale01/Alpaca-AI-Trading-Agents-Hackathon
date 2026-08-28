# ⚡ Cache Me — Autonomous Options Alpha Agents
> **Alpaca AI Trading Agents Hackathon 2026** | **Team:** Cache Me | **Platform:** lablab.ai × Alpaca

[![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Alpaca API](https://img.shields.io/badge/Alpaca-Paper%20Trading%20API-FFD100?logo=alpaca&logoColor=black)](https://alpaca.markets)
[![MCP Protocol](https://img.shields.io/badge/Model%20Context%20Protocol-Alpaca%20MCP-00D4B2)](https://docs.alpaca.markets/us/docs/alpaca-mcp-server)
[![Tests](https://img.shields.io/badge/Tests-26%2F26%20Passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

**Cache Me** is an autonomous multi-agent AI options trading system built on Alpaca's developer platform. It combines real-time market regime detection, mathematical options pricing (Black-Scholes & Greeks), LLM reasoning via Alpaca's Model Context Protocol (MCP) server & Featherless AI, and rigorous risk guardrails to generate consistent, risk-adjusted P&L on a dedicated **\$100,000** paper account.

---

## 🏗️ System Architecture

```
                                 ┌────────────────────────┐
                                 │   Market Data Stream   │
                                 │ (Alpaca Data API / VIX)│
                                 └───────────┬────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     Master Orchestrator   │
                               │  (VIX Regime Detection)   │
                               └─────────────┬─────────────┘
                                             │
             ┌────────────────┬──────────────┴───────────────┬────────────────┐
             │                │                              │                │
             ▼                ▼                              ▼                ▼
     ┌───────────────┐ ┌───────────────┐            ┌─────────────────┐ ┌───────────┐
     │ ThetaCollector│ │ IVCrushAgent  │            │  MomoBreakout   │ │HedgeAgent │
     │  (CSP Income) │ │ (Straddles)   │            │   (OTM Calls)   │ │(SPY Puts) │
     └───────┬───────┘ └───────┬───────┘            └────────┬────────┘ └─────┬─────┘
             │                 │                             │                │
             └─────────────────┴──────────────┬──────────────┴────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │     Risk Manager Engine     │
                               │   (6 Hard Circuit Breakers) │
                               └──────────────┬──────────────┘
                                              │ [Approved Orders]
                                              ▼
                               ┌─────────────────────────────┐
                               │   Execution & Monitoring    │
                               │  ┌────────────────────────┐ │
                               │  │   Alpaca Trading API   │ │
                               │  │   Alpaca MCP Server    │ │
                               │  │   Alpaca CLI Runner    │ │
                               │  │   Streamlit Dashboard  │ │
                               │  └────────────────────────┘ │
                               └─────────────────────────────┘
```

---

## 🎯 Multi-Agent Strategy Portfolio

| Agent | Strategy & Mechanics | Target Delta | DTE Range | Exit Conditions |
| :--- | :--- | :---: | :---: | :--- |
| **Theta Collector** | Sells cash-secured puts on liquid ETFs and high-volume stocks (`SPY`, `QQQ`, `IWM`, `GLD`, `PLTR`, `SOFI`) to collect time decay premium. | ~0.20 | 28–45 Days | 50% max profit OR 21 DTE time stop |
| **IV Crush Agent** | Sells ATM straddles 1–3 days prior to company earnings announcements to capture rapid volatility collapse. | 0.00 | 7–14 Days | 40% profit target OR 1 day post-earnings |
| **Momo Breakout** | Buys cheap OTM calls on explosive momentum breakouts confirmed by 20/50 EMA bullish cross + 2x volume surge. | ~0.30 | 28–45 Days | +100% gain target OR -50% premium stop loss |
| **Hedge Agent** | Automatically purchases protective SPY puts when aggregate portfolio delta exceeds +30 or VIX spikes above 22. | -0.20 | 28–45 Days | Closed when market regime normalizes |

---

## 🛡️ 6-Gate Risk Management Framework

Every single trade must pass through the stateful **Risk Manager** ([`core/risk_manager.py`](core/risk_manager.py)) prior to execution:

1. **☠️ VIX Kill Switch:** Halts all new open orders whenever VIX $\ge 35.0$.
2. **🛑 Daily Loss Limit:** Freezes new trades if daily portfolio drawdown reaches $-2\%$ ($-\$2,000$).
3. **🎯 Single Position Cap:** Maximum notional/margin exposure per trade is capped at $\le 5\%$ of equity ($\le \$5,000$).
4. **📊 Total Options Exposure:** Aggregated options positions are limited to $\le 30\%$ of total equity ($\le \$30,000$).
5. **⚖️ Delta Neutrality Bounds:** Portfolio net delta is continuously enforced within $[-50.0, +50.0]$.
6. **⏳ Binary Event Cooldown:** Enforces a 2-hour trading blackout before and after high-impact market events.

---

## 🔌 Alpaca Platform & Partner Integration

- **⚡ Alpaca Trading API (`alpaca-py 0.44.0`):** Full-featured integration for account queries, stock/options order submission, and position monitoring.
- **🔌 Model Context Protocol (MCP) Bridge ([`core/mcp_bridge.py`](core/mcp_bridge.py)):** Connects AI agents directly with Alpaca tools using structured function calling.
- **⌨️ Alpaca CLI Automation ([`cli/run_agent.sh`](cli/run_agent.sh)):** Lightweight CLI shell script for automated sessions, structured JSON output, and cron jobs.
- **🧠 Featherless AI LLM Inference:** Serverless open-source model inference (`meta-llama/Llama-3.1-8B-Instruct`) for earnings analysis and unstructured market reasoning.
- **📈 Real-Time Market Data:** Alpaca Historical & Latest Data Clients for multi-timeframe OHLCV bars, quotes, and option contract chains.

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon.git
cd Alpaca-AI-Trading-Agents-Hackathon
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
```env
ALPACA_API_KEY=your_alpaca_paper_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FEATHERLESS_API_KEY=your_featherless_key
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
> Open your browser at **[http://localhost:8501](http://localhost:8501)** to monitor live equity, P&L gauge, open option positions, and risk metrics.

#### Run via Alpaca CLI Runner
```bash
bash cli/run_agent.sh status
bash cli/run_agent.sh run
```

---

## 🧪 Automated Testing

The project includes unit tests for mathematical options pricing, Greeks, and all risk gates:

```bash
python -m pytest tests/ -v
```

```text
============================= test session starts =============================
collected 26 items

tests/test_options_pricer.py::test_black_scholes_call_price_positive PASSED
tests/test_options_pricer.py::test_black_scholes_put_price_positive PASSED
tests/test_options_pricer.py::test_put_call_parity PASSED
tests/test_options_pricer.py::test_expired_options PASSED
tests/test_options_pricer.py::test_greeks_call PASSED
tests/test_options_pricer.py::test_greeks_put PASSED
tests/test_options_pricer.py::test_iv_solver PASSED
tests/test_options_pricer.py::test_iv_rank PASSED
tests/test_risk_manager.py::test_vix_kill_switch_blocks_trade PASSED
tests/test_risk_manager.py::test_daily_loss_limit_blocks_at_threshold PASSED
tests/test_risk_manager.py::test_position_size_exceeds_5pct_blocks PASSED
tests/test_risk_manager.py::test_options_exposure_exceeds_30pct_blocks PASSED
tests/test_risk_manager.py::test_delta_within_range_passes PASSED
...
============================= 26 passed in 3.11s ==============================
```

---

## 📂 Project Structure

```
Alpaca-AI-Trading-Agents-Hackathon/
├── agents/
│   ├── orchestrator.py        # Master controller: regime detection & capital routing
│   ├── theta_collector.py     # Cash-Secured Put (CSP) income generation agent
│   ├── iv_crush_agent.py      # Pre-earnings ATM straddle volatility agent
│   ├── momo_breakout.py       # EMA cross + volume breakout call buyer
│   └── hedge_agent.py         # Dynamic SPY protective put portfolio hedge
├── core/
│   ├── alpaca_client.py       # Alpaca Trading & Market Data API wrapper
│   ├── mcp_bridge.py          # Alpaca MCP Server + Featherless LLM bridge
│   ├── risk_manager.py        # 6-gate risk management engine
│   ├── options_pricer.py      # Black-Scholes pricing, Greeks & IV solver
│   └── market_data.py         # VIX tracking, EMA crossover, & volume surge
├── dashboard/
│   └── monitor.py             # Interactive Streamlit live trading dashboard
├── cli/
│   └── run_agent.sh           # Alpaca CLI session runner & cron automation
├── docs/
│   └── writeup.md             # One-page hackathon architecture write-up
├── tests/
│   ├── test_risk_manager.py   # Unit tests for all 6 risk gates
│   └── test_options_pricer.py # Unit tests for Black-Scholes, Greeks & IV
├── config.py                  # Global parameters & universe configuration
├── main.py                    # Multi-mode CLI entry point
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🏆 Hackathon Details

- **Event:** Alpaca AI Trading Agents Hackathon (lablab.ai × Alpaca)
- **Dates:** August 28 – September 4, 2026
- **Team Name:** Cache Me
- **Dedicated Account ID:** `9e62f22b-a0cb-49c3-99d0-10a1fcd2c9a3`
- **Initial Paper Balance:** \$100,000.00 USD

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
