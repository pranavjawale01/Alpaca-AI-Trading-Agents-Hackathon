"""
dashboard/monitor.py — Real-Time Trading & Portfolio Monitor.

Professional financial dashboard for the Cache Me If You Can trading system.
Styled with modern typography, dark-mode financial cards, live P&L gauge,
real-time open positions, and risk management limits.

Run: streamlit run dashboard/monitor.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

# Ensure project root is in sys.path
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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def _get_client():
    from core.alpaca_client import AlpacaClient
    return AlpacaClient()

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Cache Me If You Can — Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for High-End Financial Terminal UI ─────────────
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Top Bar */
    .top-bar {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        padding: 18px 24px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .terminal-title {
        font-size: 22px;
        font-weight: 700;
        color: #58a6ff;
        margin: 0;
    }
    
    .status-badge {
        background-color: #238636;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #f0f6fc;
    }
    
    .metric-change-pos {
        color: #3fb950;
        font-size: 14px;
        font-weight: 600;
    }
    
    .metric-change-neg {
        color: #f85149;
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Section containers */
    .section-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Controls")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.divider()
    st.markdown("### Resources")
    st.markdown("- [Alpaca Paper Console](https://app.alpaca.markets/paper/dashboard/overview)")
    st.markdown("- [Hackathon Portal](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)")
    st.markdown("- [GitHub Codebase](https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon)")
    st.divider()
    st.caption("Team: Cache Me If You Can")
    st.caption("Strategy: Multi-Agent Options Alpha")

# ── Top Bar Header ─────────────────────────────────────────────
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="top-bar">
    <div>
        <div class="terminal-title">Cache Me If You Can — Options Alpha Terminal</div>
        <div style="font-size: 12px; color: #8b949e; margin-top: 4px;">Real-time Alpaca Paper Trading Monitor | System Clock: {now_utc}</div>
    </div>
    <div>
        <span class="status-badge">LIVE TRADING</span>
    </div>
</div>
""", unsafe_allow_html=True)

STARTING_BALANCE = 100_000.0

try:
    client = _get_client()
    account = client.get_account()
    positions = client.get_all_positions()

    equity = float(account["equity"])
    cash = float(account["cash"])
    buying_power = float(account["buying_power"])
    account_id = str(account.get("id", ""))
    total_pnl = equity - STARTING_BALANCE
    total_pnl_pct = (total_pnl / STARTING_BALANCE) * 100

    data_loaded = True
except Exception as e:
    st.error(f"Broker connection offline: {e}")
    st.info("Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are configured in .env")
    data_loaded = False

# ── Metrics Row ────────────────────────────────────────────────
if data_loaded:
    m1, m2, m3, m4 = st.columns(4)
    
    pnl_class = "metric-change-pos" if total_pnl >= 0 else "metric-change-neg"
    pnl_sign = "+" if total_pnl >= 0 else ""

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio Equity</div>
            <div class="metric-value">${equity:,.2f}</div>
            <div class="{pnl_class}">{pnl_sign}${total_pnl:,.2f} ({pnl_sign}{total_pnl_pct:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cash Balance</div>
            <div class="metric-value">${cash:,.2f}</div>
            <div style="font-size: 13px; color: #8b949e;">Collected Premium: +${cash - STARTING_BALANCE:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active Options Contracts</div>
            <div class="metric-value">{len(positions)}</div>
            <div style="font-size: 13px; color: #8b949e;">Market Exposure Active</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Buying Power</div>
            <div class="metric-value">${buying_power:,.2f}</div>
            <div style="font-size: 13px; color: #8b949e;">Account: {account_id[:8]}...</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance Gauge & Positions Table ─────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Portfolio Performance")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=equity,
            number={"prefix": "$", "valueformat": ",.0f", "font": {"color": "#ffffff", "size": 28}},
            delta={
                "reference": STARTING_BALANCE,
                "valueformat": ",.2f",
                "prefix": "$",
                "increasing": {"color": "#3fb950"},
                "decreasing": {"color": "#f85149"},
            },
            gauge={
                "axis": {"range": [STARTING_BALANCE * 0.85, STARTING_BALANCE * 1.25], "tickcolor": "#8b949e"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#161b22",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [STARTING_BALANCE * 0.85, STARTING_BALANCE], "color": "#2d1619"},
                    {"range": [STARTING_BALANCE, STARTING_BALANCE * 1.25], "color": "#16281e"},
                ],
                "threshold": {
                    "line": {"color": "#d29922", "width": 3},
                    "thickness": 0.85,
                    "value": STARTING_BALANCE,
                },
            },
            title={"text": "Baseline: $100,000", "font": {"color": "#8b949e", "size": 14}},
        ))
        fig_gauge.update_layout(
            height=310,
            margin=dict(t=30, b=10, l=20, r=20),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        st.markdown("### Open Positions")
        if positions:
            df = pd.DataFrame(positions)
            
            # Format dataframe for professional display
            display_df = pd.DataFrame({
                "Contract": df["symbol"],
                "Qty": df["qty"].astype(float).map("{:+.1f}".format),
                "Entry Price": df["avg_entry_price"].astype(float).map("${:,.2f}".format),
                "Market Value": df["market_value"].astype(float).map("${:,.2f}".format),
                "Unrealized P&L": df["unrealized_pl"].astype(float).map("${:+,.2f}".format),
                "Asset Class": df["asset_class"].str.upper(),
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Orchestrator scans and enters orders at session triggers.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Dashboard Panel ────────────────────────────────────
    st.markdown("### Risk Manager & Circuit Breakers")
    
    r1, r2, r3, r4 = st.columns(4)

    daily_loss_limit = 2000.0  # 2% of $100k
    daily_pnl = total_pnl
    loss_remaining = max(0.0, daily_loss_limit + daily_pnl)

    options_positions = [p for p in positions if "option" in str(p.get("asset_class", "")).lower()]
    opt_exposure = sum(abs(float(p["market_value"])) for p in options_positions)
    opt_pct = (opt_exposure / equity) * 100 if equity > 0 else 0.0

    with r1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Loss Buffer</div>
            <div class="metric-value">${loss_remaining:,.2f}</div>
            <div style="font-size: 13px; color: #8b949e;">Limit: -$2,000.00 (-2.0%)</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        exposure_color = "#3fb950" if opt_pct <= 25.0 else "#d29922"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Options Exposure</div>
            <div class="metric-value" style="color: {exposure_color};">{opt_pct:.1f}%</div>
            <div style="font-size: 13px; color: #8b949e;">Hard Gate Cap: 30.0%</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Single Position Cap</div>
            <div class="metric-value">${equity * 0.05:,.0f}</div>
            <div style="font-size: 13px; color: #8b949e;">Max 5.0% of Portfolio</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">VIX Circuit Breaker</div>
            <div class="metric-value">VIX &ge; 35.0</div>
            <div style="font-size: 13px; color: #8b949e;">Auto Order Halt Active</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("Cache Me If You Can | Alpaca AI Trading Agents Hackathon 2026 | lablab.ai x Alpaca")

# ── Auto-refresh Trigger ───────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
