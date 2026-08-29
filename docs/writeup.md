# Cache Me If You Can — Project Write-Up
**Alpaca AI Trading Agents Hackathon 2026** | **Platform:** lablab.ai x Alpaca

---

## 1. Executive Summary & AI Architecture

**Cache Me If You Can** is an autonomous multi-agent options trading system built on Alpaca's developer platform. It combines real-time market regime detection, mathematical options pricing (Black-Scholes and Greeks), a **3-model LLM voting council**, **Kelly Criterion dynamic sizing**, **mid-price smart limit order execution**, and **6 stateful risk circuit breakers** to generate consistent, risk-adjusted alpha on a dedicated **$100,000** paper account.

---

## 2. Multi-Agent Strategy Portfolio & AI Council

### Master Orchestrator & Regime Routing
- **VIX < 18 (Risk-ON)**: All strategies active (Theta, Momo Breakout, IV Crush, Hedge).
- **18 <= VIX < 28 (Neutral)**: Theta income + IV Crush + Hedge active.
- **VIX >= 28 (Risk-OFF)**: Pure defensive hedging only.

### 4 Specialized Autonomous Sub-Agents
1. **ThetaCollector** — Sells 28-45 DTE cash-secured puts on liquid ETFs and high-volume stocks (`SPY`, `QQQ`, `IWM`, `GLD`, `PLTR`, `SOFI`) at ~20 delta when IVR > 30. Kelly-sized contracts, mid-price limit fills. Closes at 50% profit target or 21 DTE time stop.
2. **IVCrushAgent** — Sells ATM straddles 1-3 days before company earnings announcements, capturing volatility collapse post-announcement.
3. **MomoBreakout** — Buys cheap 5% OTM calls on 20/50 EMA crossover with 2x volume surge. Features a dynamic **trailing stop** (25% pullback from peak P&L) to lock in gains on winners.
4. **HedgeAgent** — Automatically purchases protective SPY puts when aggregate portfolio delta exceeds +30 or VIX spikes above 22.

### 3-Model LLM Council (Voting Ensemble)
Before any trade signal is submitted, it is evaluated concurrently by 3 distinct LLMs:
- **Model 1**: `meta-llama/Llama-3.1-8B-Instruct` (Fast primary reasoner)
- **Model 2**: `mistralai/Mistral-7B-Instruct-v0.3` (Contrarian validation)
- **Model 3**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (Chain-of-thought analysis)

Votes are weighted by confidence: `net_score = Σ(confidence_i × vote_i) / n_models`. Only signals reaching consensus (|net_score| >= 0.60) proceed to execution, drastically eliminating false breakouts and whipsaws.

---

## 3. Professional Execution & Capital Sizing

1. **Kelly Criterion Position Sizing (`core/kelly_sizer.py`)**:
   - Replaces static percentage bets with mathematical optimal sizing: `f* = (b·p - q) / b`.
   - Uses quarter-Kelly (`f* × 0.25`) per strategy based on real historical win rates from the trade journal.
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

## 5. Alpaca Platform & Partner Integration

| Component | Usage & Implementation |
|---|---|
| **Alpaca Trading API (`alpaca-py 0.44.0`)** | Full account management, options market and limit order execution, position tracking |
| **Alpaca Market Data API** | Historical daily/intraday bars, live quotes on IEX feed, options contract chains |
| **Model Context Protocol (MCP) Bridge** | Integrates Featherless AI reasoning with Alpaca tools via structured function calling |
| **Alpaca CLI Runner (`cli/run_agent.sh`)** | Automated shell execution, cron jobs, and JSON logging |
| **Streamlit Real-Time Terminal** | Full-featured web UI (`dashboard/monitor.py`) with Plotly charts, live Greeks, and interactive controls |
| **Paper Trading Environment** | Dedicated $100,000 account tested with live market conditions |

---

## 6. Paper Trading Account

- **Account ID**: `9e62f22b-a0cb-49c3-99d0-10a1fcd2c9a3`
- **Initial Balance**: $100,000.00 USD
- **Environment**: Alpaca Paper Trading
