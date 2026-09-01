# Cache Me If You Can — Project Write-Up
**Alpaca AI Trading Agents Hackathon 2026** | **Platform:** lablab.ai x Alpaca

---

## 1. Executive Summary & AI Architecture

**Cache Me If You Can** is an autonomous multi-agent options trading system built on Alpaca's developer platform. It combines real-time market regime detection, mathematical options pricing (Black-Scholes and Greeks), a **Hybrid Greedy-Voting LLM Council**, **Dynamic Kelly Criterion Position Sizing**, **mid-price smart limit order execution**, and **6 stateful risk circuit breakers** to generate consistent, risk-adjusted alpha on a dedicated **$100,000** paper account.

The system is engineered for complete standalone deployment on **Streamlit Community Cloud**, featuring an in-process background auto-pilot scheduler and client-side keep-alive heartbeats to run trading sessions autonomously without requiring an external server.

---

## 2. Multi-Agent Strategy Portfolio & Hybrid Council

### Master Orchestrator & Dynamic Regime Routing
- **VIX < 18 (Risk-ON)**: All strategies active (Theta, Momo Breakout, IV Crush, Hedge).
- **18 <= VIX < 28 (Neutral)**: Theta income + Momo Breakout + IV Crush + Hedge active.
- **VIX >= 28 (Risk-OFF)**: Pure defensive hedging only.

### 4 Specialized Autonomous Sub-Agents
1. **ThetaCollector** — Sells 28-45 DTE cash-secured puts on liquid ETFs and high-volume stocks (`SPY`, `QQQ`, `IWM`, `GLD`, `PLTR`, `SOFI`) at ~20 delta when IVR > 30. Scaled by opportunity multipliers; mid-price limit fills. Closes at 50% profit target or 21 DTE time stop.
2. **IVCrushAgent** — Sells ATM straddles 1-3 days before company earnings announcements, capturing volatility collapse post-announcement.
3. **MomoBreakout** — Buys cheap 5% OTM calls on 20/50 EMA crossover with 2x volume surge. Features a dynamic **trailing stop** (25% pullback from peak P&L) to lock in gains on winners. Active in both `RISK_ON` and `NEUTRAL` regimes.
4. **HedgeAgent** — Automatically purchases protective SPY puts when aggregate portfolio delta exceeds +30 or VIX spikes above 22.

### Hybrid Greedy-Voting LLM Council
Before any trade signal is submitted, it is evaluated concurrently by 3 distinct LLMs:
- **Model 1**: `meta-llama/Llama-3.1-8B-Instruct` (Fast primary reasoner)
- **Model 2**: `mistralai/Mistral-7B-Instruct-v0.3` (Contrarian validation)
- **Model 3**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (Chain-of-thought analysis)

#### Core Hybrid Upgrades:
1. **Model Credibility Weighting (`core/model_credibility.py`)**:
   - Each model's vote is scaled by its historical accuracy weight `w_i ∈ [0.5, 1.5]`.
   - `net_score = Σ(w_i × confidence_i × vote_i) / Σ(w_i)`.
   - Accurate models gain voting leverage; incorrect models lose voting power over time with exponential decay toward baseline `1.0`.
2. **Regime-Adaptive Conviction Tiers (`core/llm_council.py`)**:
   - Replaces binary pass/fail with sized execution:
     - **STRONG Tier**: `1.00x` full Kelly allocation.
     - **MODERATE Tier**: `0.70x` Kelly allocation.
     - **PILOT Tier**: `0.40x` Kelly allocation (for exploratory / emerging setups).
     - **VETO Tier**: `0.00x` (order skipped and logged).
3. **Greedy Opportunity Scorer (`core/opportunity_scorer.py`)**:
   - Multiplies Kelly allocation by `1.0x` to `2.0x` by stacking 6 independent market factors:
     - `+0.20` for elevated IVR (> 50)
     - `+0.20` for EMA trend confirmation
     - `+0.15` for sweet-spot VIX (15-22 range)
     - `+0.15` for volume surge (> 1.8x)
     - `+0.15` for diversification (symbol not currently in open positions)
     - `+0.15` for positive session P&L (scaling into winning streaks)

---

## 3. Professional Execution & Capital Sizing

1. **Dynamic Kelly Criterion Position Sizing (`core/kelly_sizer.py`)**:
   - Computes mathematical edge: `f* = (b·p - q) / b`.
   - Effective fraction: `min(Quarter-Kelly × Size Multiplier × Greedy Multiplier, 0.50)`.
2. **Smart Mid-Price Limit Order Execution (`core/smart_executor.py`)**:
   - Eliminates options bid-ask spread slippage by submitting at mid-price `(bid + ask) / 2`, stepping price toward market if unfilled, with seamless fallback to market orders.
3. **Persistent SQLite Trade Journal (`core/trade_journal.py`)**:
   - Records all trade entries, exits, P&L, Sharpe, and council scores in `logs/trading.db` for session-over-session learning.

---

## 4. 6-Gate Risk Management Framework

Every order is strictly vetted by `core/risk_manager.py` before hitting Alpaca's API:

| Gate | Threshold | Purpose |
|---|---|---|
| **VIX Kill Switch** | VIX >= 35.0 | Freezes all opening orders during market crises |
| **Daily Loss Limit** | P&L <= -$2,000 (-2.0%) | Preserves capital during severe drawdowns |
| **Single Position Cap** | <= $5,000 (5.0% of equity) | Enforces strict diversification |
| **Options Exposure Cap** | <= $30,000 (30.0% of equity) | Controls total portfolio leverage |
| **Portfolio Delta Bounds** | [-50.0, +50.0] | Keeps portfolio directionally neutral |
| **Binary Event Cooldown** | +/- 2 hours around events | Prevents unhedged earnings exposure |

---

## 5. Streamlit Cloud Auto-Pilot & Autonomous UI

1. **In-Process Autonomous Scheduler**:
   - Launches a background daemon thread upon startup in `dashboard/monitor.py` that checks market hours (13:30 to 21:00 UTC, Mon-Fri) and runs automated sessions at configurable intervals.
2. **Persistent Keep-Alive**:
   - Non-rendering JavaScript heartbeat component pings the application URL every 4.5 minutes to keep Streamlit Community Cloud active.
3. **Live Terminal Tabs**:
   - **Portfolio Overview**: Real-time equity gauge, cash balance, active positions, and risk breakers.
   - **Live AI Decisions**: Live scan watchlist showing conviction tiers, greed multipliers, and council verdicts.
   - **Live Kelly Sizing & Execution**: Dynamic budget allocation matrix and SQLite trade journal.
   - **Market Charts & Payoffs**: Interactive Plotly candlestick charts with indicators + isolated Options Expiration Simulator.
   - **Trading Engine**: Live auto-pilot status, manual execution triggers, and session activity logs.

---

## 6. Alpaca Platform & Partner Integration

| Component | Usage & Implementation |
|---|---|
| **Alpaca Trading API (`alpaca-py 0.44.0`)** | Full account management, options market and limit order execution, position tracking |
| **Alpaca Market Data API** | Historical daily/intraday bars, live quotes on IEX feed, options contract chains |
| **Model Context Protocol (MCP) Bridge** | Integrates Featherless AI reasoning with Alpaca tools via structured function calling |
| **Alpaca CLI Runner (`cli/run_agent.sh`)** | Automated shell execution, cron jobs, and JSON logging |
| **Streamlit Real-Time Terminal** | Full-featured web UI (`dashboard/monitor.py`) with Plotly charts, live Greeks, and interactive controls |
| **Paper Trading Environment** | Dedicated $100,000 account tested with live market conditions |

---

## 7. Paper Trading Account

- **Account ID**: `9e62f22b-a0cb-49c3-99d0-10a1fcd2c9a3`
- **Initial Balance**: $100,000.00 USD
- **Environment**: Alpaca Paper Trading

