# Cache Me — Project Write-Up
**Alpaca AI Trading Agents Hackathon 2026**

---

## AI Logic

Cache Me is a **multi-agent autonomous trading system** where a master Orchestrator routes capital between four specialised sub-agents based on real-time market regime detection.

**Regime Detection**: The Orchestrator reads VIX at session start and classifies the market:
- VIX < 18 → Risk-ON (all agents active)
- 18 ≤ VIX < 28 → Neutral (theta + hedge only)
- VIX ≥ 28 → Risk-OFF (hedge only, defensive)

**LLM Integration (Featherless AI via MCP)**: The Orchestrator queries a Llama-3 model through Alpaca's MCP Server to identify upcoming earnings events from our watchlist and make edge-case positioning decisions in natural language.

**Agent Specialisation**:
1. **ThetaCollector** — Sells 30-45 DTE cash-secured puts on SPY/QQQ/IWM/GLD at ~20 delta when IVR > 30. Closes at 50% profit target or 21 DTE.
2. **IVCrushAgent** — Sells ATM straddles 1–3 days before earnings announcements, closing 1 day after to capture IV collapse.
3. **MomoBreakout** — Buys 5% OTM calls when 20-EMA crosses 50-EMA with volume surge ≥ 2x average. Active only in risk-on regime.
4. **HedgeAgent** — Buys 4% OTM SPY puts when portfolio delta > 30 or VIX > 22, capped at 0.5% of equity.

---

## Risk Gates

Every order is evaluated against 6 hard gates **before execution**. A `RiskViolation` exception cancels the order and logs the reason.

| Gate | Threshold | Purpose |
|---|---|---|
| VIX Kill Switch | VIX ≥ 35 | Prevents trading in crisis conditions |
| Daily Loss Limit | P&L ≤ –\$2,000 (–2%) | Preserves capital if strategies misfire |
| Max Position Size | ≤ \$5,000 (5% of equity) | Prevents concentration risk |
| Max Options Exposure | ≤ \$30,000 (30% of equity) | Limits leverage via options |
| Portfolio Delta | –50 to +50 | Keeps portfolio directionally neutral |
| Earnings Cooldown | ±2h around events | Avoids binary event exposure |

---

## Alpaca Infrastructure

| Component | How We Use It |
|---|---|
| **Trading API** (`alpaca-py`) | Order submission, position queries, account monitoring |
| **MCP Server** | Featherless AI (Llama-3) calls Alpaca tools in natural language for earnings calendar + ad-hoc decisions |
| **Alpaca CLI** | `cli/run_agent.sh` wraps all sessions; used in cron jobs for scheduled daily execution with structured JSON logging |
| **Paper Trading Env** | All strategies developed and tested with \$100,000 virtual capital |
| **Market Data API** | Real-time quotes for options mid-price, OHLCV bars for EMA/volume signals |

---

## Paper Account

- **Starting Balance**: \$100,000
- **Account ID**: *(see submission form)*
- **Environment**: Alpaca Paper Trading
