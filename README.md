# Cache Me — Options Alpha Agent
> **Alpaca AI Trading Agents Hackathon 2026** | Team: Cache Me | lablab.ai × Alpaca

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-yellow)](https://alpaca.markets)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🤖 What is Cache Me?

**Cache Me** is an autonomous multi-agent AI trading system that generates P&L by selling and buying options on Alpaca's paper trading platform. The system "caches" market context across every decision, enabling smarter, regime-aware trading.

### Core Architecture

```
Orchestrator (VIX Regime Detection)
    ├── ThetaCollector  → Sells cash-secured puts on SPY/QQQ/IWM/GLD
    ├── IVCrushAgent    → Sells straddles before earnings (IV crush)
    ├── MomoBreakout    → Buys OTM calls on momentum breakouts
    └── HedgeAgent      → Buys SPY puts for portfolio protection
```

All agents route through a **Risk Manager** with 6 hard gates before any order fires.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon
cd Alpaca-AI-Trading-Agents-Hackathon
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Fill in your Alpaca Paper API key and Featherless AI key
```

### 3. Create Alpaca Paper Account
- Sign up at [alpaca.markets](https://alpaca.markets)
- Create a **Paper Trading** account
- Set starting balance to **\$100,000**
- Copy your API key and secret to `.env`

### 4. Run

```bash
# Single trading session
python main.py run

# Account status
python main.py status

# Scheduled daily sessions (runs at market open)
python main.py schedule

# Live dashboard
streamlit run dashboard/monitor.py

# Using Alpaca CLI
bash cli/run_agent.sh run
bash cli/run_agent.sh status
```

---

## 📦 Project Structure

```
├── agents/
│   ├── orchestrator.py      # Master agent: regime detection + routing
│   ├── theta_collector.py   # CSP income: sell puts on SPY/QQQ/IWM/GLD
│   ├── iv_crush_agent.py    # Sell straddles before earnings
│   ├── momo_breakout.py     # Buy OTM calls on momentum breakouts
│   └── hedge_agent.py       # SPY put hedge for portfolio protection
├── core/
│   ├── alpaca_client.py     # Alpaca Python SDK wrapper
│   ├── mcp_bridge.py        # Alpaca MCP Server + Featherless AI integration
│   ├── risk_manager.py      # 6-gate risk enforcement system
│   ├── options_pricer.py    # Black-Scholes + Greeks + IV solver
│   └── market_data.py       # VIX, prices, EMA signals, volume surge
├── dashboard/
│   └── monitor.py           # Streamlit real-time P&L dashboard
├── cli/
│   └── run_agent.sh         # Alpaca CLI runner + cron scheduler
├── docs/
│   └── writeup.md           # Judge write-up: AI logic, risk, infrastructure
├── tests/
│   └── test_risk_manager.py # pytest suite for all risk gates
├── config.py                # Central config (env-based)
├── main.py                  # Entry point
└── requirements.txt
```

---

## 🎯 Trading Strategies

### 1. Theta Collector (Primary Income)
Sells **cash-secured puts** on liquid ETFs (SPY, QQQ, IWM, GLD).
- **DTE**: 30–45 days to expiration
- **Strike**: ~10% OTM (Delta ≈ 0.20)
- **Entry**: IVR > 30, VIX < 30
- **Exit**: 50% profit target OR 21 DTE

### 2. IV Crush Agent
Sells **ATM straddles** before earnings announcements.
- Profits from implied volatility collapsing after the event
- Closes within 1–2 days post-earnings
- Max 3 simultaneous positions

### 3. Momo Breakout
Buys **cheap OTM calls** on momentum breakout stocks.
- Signal: 20-EMA crosses 50-EMA + volume surge ≥ 2x average
- Only active in risk-on regime (VIX < 25)
- Max 1% of equity per trade

### 4. Hedge Agent
Buys **SPY protective puts** when portfolio delta gets too long or VIX rises.
- Activates when: portfolio delta > 30 OR VIX > 22
- Cost capped at 0.5% of equity per cycle

---

## 🛡️ Risk Gates

Every order passes through 6 hard gates before execution:

| Gate | Threshold |
|---|---|
| VIX Kill Switch | Halt all new orders if VIX ≥ 35 |
| Daily Loss Limit | Halt if daily P&L ≤ –\$2,000 (–2%) |
| Max Position Size | ≤ \$5,000 per position (5% of equity) |
| Max Options Exposure | ≤ \$30,000 (30% of equity) |
| Portfolio Delta | –50 to +50 at all times |
| Earnings Cooldown | No new trades within 2h of earnings |

---

## 🔌 Alpaca Infrastructure

| Component | Usage |
|---|---|
| **Trading API** | Order placement, position management, account queries |
| **MCP Server** | Featherless AI LLM ↔ Alpaca API for natural-language decisions |
| **Alpaca CLI** | Session runner, cron scheduling, structured JSON output |
| **Paper Trading** | All strategies tested with \$100,000 virtual capital |
| **Market Data API** | Real-time quotes, OHLCV bars, options chains |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Dashboard

```bash
streamlit run dashboard/monitor.py
```

Open [http://localhost:8501](http://localhost:8501) to view the live P&L dashboard.

---

## ⚙️ Configuration

Key settings in `config.py` (override via `.env`):

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Paper trading API key |
| `ALPACA_SECRET_KEY` | — | Paper trading secret |
| `FEATHERLESS_API_KEY` | — | Featherless AI key |
| `MAX_POSITION_PCT` | 0.05 | Max 5% per position |
| `MAX_OPTIONS_EXPOSURE_PCT` | 0.30 | Max 30% in options |
| `DAILY_LOSS_LIMIT_PCT` | 0.02 | Halt at –2% daily loss |
| `VIX_KILL_SWITCH` | 35 | Halt above VIX 35 |

---

## 🏆 Hackathon Submission

- **Team**: Cache Me
- **Event**: Alpaca AI Trading Agents Hackathon (lablab.ai × Alpaca)
- **Dates**: Aug 28 – Sep 4, 2026
- **Strategy**: Multi-agent options system (theta + IV crush + momentum)

---

## 📜 License

MIT
