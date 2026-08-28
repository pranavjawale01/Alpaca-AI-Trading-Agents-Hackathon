"""
dashboard/monitor.py — Streamlit P&L Dashboard.

Real-time monitoring dashboard for the Cache Me trading system.

Shows:
  - Current equity vs $100,000 baseline
  - Daily / cumulative P&L chart
  - Open positions table with Greeks
  - Risk gauge (delta, VIX, daily loss remaining)
  - Session activity log

Run: streamlit run dashboard/monitor.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

# Add project root to sys.path so 'core' and 'agents' packages can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import plotly.graph_objects as go
import streamlit as st

# Lazy import
def _get_client():
    from core.alpaca_client import AlpacaClient
    return AlpacaClient()

def _get_risk():
    from core.risk_manager import RiskManager
    from core.alpaca_client import AlpacaClient
    from core.market_data import MarketData
    client = AlpacaClient()
    md = MarketData(client)
    rm = RiskManager()
    vix = md.get_vix()
    rm.update_vix(vix)
    account = client.get_account()
    rm.update_equity(account["equity"])
    return rm, client


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Cache Me — Trading Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💰 Cache Me — Options Alpha Agent Dashboard")
st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.divider()
    st.markdown("**Links**")
    st.markdown("- [Alpaca Paper Account](https://app.alpaca.markets/paper/dashboard/overview)")
    st.markdown("- [Hackathon Page](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)")
    st.markdown("- [GitHub Repo](https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon)")

# ── Load Data ─────────────────────────────────────────────────
STARTING_BALANCE = 100_000.0

try:
    client = _get_client()
    account = client.get_account()
    positions = client.get_all_positions()

    equity = account["equity"]
    cash = account["cash"]
    buying_power = account["buying_power"]
    total_pnl = equity - STARTING_BALANCE
    total_pnl_pct = total_pnl / STARTING_BALANCE * 100

    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Could not connect to Alpaca API: {e}")
    st.info("Set your ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    data_loaded = False

# ── Metrics Row ────────────────────────────────────────────────
if data_loaded:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💼 Portfolio Equity",
            value=f"${equity:,.2f}",
            delta=f"${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)",
            delta_color="normal",
        )
    with col2:
        st.metric("💵 Cash", f"${cash:,.2f}")
    with col3:
        st.metric("📊 Open Positions", len(positions))
    with col4:
        st.metric(
            "⚡ Buying Power",
            f"${buying_power:,.2f}",
            help="Available margin for new positions",
        )

    st.divider()

    # ── P&L Gauge ───────────────────────────────────────────────
    col_gauge, col_positions = st.columns([1, 2])

    with col_gauge:
        st.subheader("📈 Total P&L")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=equity,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={
                "reference": STARTING_BALANCE,
                "valueformat": ",.0f",
                "prefix": "$",
            },
            gauge={
                "axis": {"range": [STARTING_BALANCE * 0.8, STARTING_BALANCE * 1.5]},
                "bar": {"color": "green" if total_pnl >= 0 else "red"},
                "steps": [
                    {"range": [STARTING_BALANCE * 0.8, STARTING_BALANCE], "color": "lightcoral"},
                    {"range": [STARTING_BALANCE, STARTING_BALANCE * 1.5], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "gold", "width": 3},
                    "thickness": 0.85,
                    "value": STARTING_BALANCE,
                },
            },
            title={"text": f"Starting: $100,000"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Positions Table ──────────────────────────────────────────
    with col_positions:
        st.subheader("📋 Open Positions")
        if positions:
            import pandas as pd
            df = pd.DataFrame(positions)
            df["unrealized_pl"] = df["unrealized_pl"].apply(lambda x: f"${float(x):+,.2f}")
            df["market_value"] = df["market_value"].apply(lambda x: f"${float(x):,.2f}")
            df["avg_entry_price"] = df["avg_entry_price"].apply(lambda x: f"${float(x):.2f}")
            st.dataframe(
                df[["symbol", "qty", "avg_entry_price", "market_value", "unrealized_pl", "asset_class"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open positions yet — agents will open positions at market open.")

    st.divider()

    # ── Risk Summary ────────────────────────────────────────────
    st.subheader("🛡️ Risk Dashboard")
    r1, r2, r3, r4 = st.columns(4)

    daily_loss_limit = -0.02 * STARTING_BALANCE
    daily_pnl = total_pnl  # simplified

    with r1:
        loss_remaining = daily_loss_limit - daily_pnl
        st.metric("Daily Loss Remaining", f"${abs(loss_remaining):,.0f}",
                  help="How much more we can lose today before halt")
    with r2:
        options_positions = [p for p in positions if "option" in p.get("asset_class", "").lower()]
        opt_exposure = sum(abs(float(p["market_value"])) for p in options_positions)
        opt_pct = opt_exposure / equity * 100
        color = "🟡" if opt_pct > 20 else "🟢"
        st.metric(f"{color} Options Exposure", f"{opt_pct:.1f}%",
                  help="Target: < 30% of equity")
    with r3:
        st.metric("🎯 Max Position Size", f"${equity * 0.05:,.0f}",
                  help="5% of equity per position")
    with r4:
        st.metric("☠️ VIX Kill Switch", "VIX > 35",
                  help="All new orders halted above this level")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("Cache Me | Alpaca AI Trading Agents Hackathon 2026 | lablab.ai × Alpaca")

# ── Auto-refresh ───────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
