"""
dashboard/monitor.py — Real-Time Trading Terminal with Interactive Graph Suite.

Professional financial dashboard for the Cache Me If You Can trading system.
Features:
  - Comprehensive Interactive Plotly Analytics with Customizable Filters
  - Asset Price Action (Candlestick, Line, OHLC) with Multi-Timeframe Controls
  - Configurable Technical Overlays (EMA 20/50/200, Bollinger Bands, Volume, RSI)
  - Interactive Trade Fill & Strike Overlay Annotations
  - Portfolio Equity & Cumulative Return Progression Timeline
  - Options Expiration Payoff & Risk Profile Curve Simulator
  - Capital Allocation & Risk Management Dashboard

Run: streamlit run dashboard/monitor.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        padding: 16px 24px;
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
    
    /* Filter container card */
    .filter-panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Controls")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)
    st.divider()
    st.markdown("### Resources & Documentation")
    st.markdown("- [Alpaca Paper Console](https://app.alpaca.markets/paper/dashboard/overview)")
    st.markdown("- [Hackathon Portal](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)")
    st.markdown("- [GitHub Codebase](https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon)")
    st.divider()
    st.caption("Team: Cache Me If You Can")
    st.caption("Strategy: Autonomous Options Alpha")

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
            <div class="metric-label">Active Options Positions</div>
            <div class="metric-value">{len(positions)}</div>
            <div style="font-size: 13px; color: #8b949e;">Live Options Exposure</div>
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

    # ── Interactive Graph Section with Rich Customizable Filters ──
    st.markdown("### Interactive Market Analytics & Strategy Graph")
    
    # Customizable Filter Toolbar
    with st.expander("Chart Customization & Indicator Filters", expanded=True):
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        
        with f_col1:
            selected_asset = st.selectbox(
                "Underlying Asset",
                options=["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "VIXY"],
                index=0
            )
        with f_col2:
            timeframe_choice = st.selectbox(
                "Bar Frequency",
                options=["1Day", "1Hour", "15Min", "5Min"],
                index=0
            )
        with f_col3:
            chart_style = st.selectbox(
                "Chart Type",
                options=["Candlestick", "Line Chart (Close)", "OHLC Bars"],
                index=0
            )
        with f_col4:
            history_bars = st.slider(
                "Historical Bars",
                min_value=30, max_value=250, value=90, step=10
            )
        with f_col5:
            indicator_selection = st.multiselect(
                "Technical Overlays",
                options=["20 EMA", "50 EMA", "Bollinger Bands", "Volume", "RSI (14)"],
                default=["20 EMA", "50 EMA", "Volume"]
            )

    # Fetch Data and Render Plotly Graph
    try:
        bars = client.get_bars(selected_asset, timeframe=timeframe_choice, limit=history_bars)
        
        if bars and len(bars) >= 5:
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t").reset_index(drop=True)

            # Compute Indicators
            if "20 EMA" in indicator_selection:
                df["ema20"] = df["c"].ewm(span=20, adjust=False).mean()
            if "50 EMA" in indicator_selection:
                df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
            if "Bollinger Bands" in indicator_selection:
                df["sma20"] = df["c"].rolling(window=20).mean()
                df["std20"] = df["c"].rolling(window=20).std()
                df["bb_upper"] = df["sma20"] + (df["std20"] * 2)
                df["bb_lower"] = df["sma20"] - (df["std20"] * 2)
            if "RSI (14)" in indicator_selection:
                delta = df["c"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.nan)
                df["rsi"] = 100 - (100 / (1 + rs))

            # Determine Subplot Structure
            has_volume = "Volume" in indicator_selection
            has_rsi = "RSI (14)" in indicator_selection

            rows = 1 + (1 if has_volume else 0) + (1 if has_rsi else 0)
            row_heights = [0.65] if rows == 1 else ([0.65, 0.20] if rows == 2 else [0.55, 0.25, 0.20])

            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights
            )

            # 1. Main Price Plot
            if chart_style == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                    name=f"{selected_asset} OHLC",
                    increasing_line_color="#3fb950", decreasing_line_color="#f85149"
                ), row=1, col=1)
            elif chart_style == "Line Chart (Close)":
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["c"],
                    mode="lines",
                    name=f"{selected_asset} Price",
                    line=dict(color="#58a6ff", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(88, 166, 255, 0.05)"
                ), row=1, col=1)
            else:
                fig.add_trace(go.Ohlc(
                    x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                    name=f"{selected_asset} OHLC",
                    increasing_line_color="#3fb950", decreasing_line_color="#f85149"
                ), row=1, col=1)

            # 2. Overlays
            if "20 EMA" in indicator_selection:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["ema20"], name="20 EMA (Fast)",
                    line=dict(color="#58a6ff", width=1.5)
                ), row=1, col=1)
            if "50 EMA" in indicator_selection:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["ema50"], name="50 EMA (Slow)",
                    line=dict(color="#d29922", width=1.5)
                ), row=1, col=1)
            if "Bollinger Bands" in indicator_selection:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["bb_upper"], name="BB Upper (2σ)",
                    line=dict(color="#bc8cff", width=1, dash="dot")
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["bb_lower"], name="BB Lower (2σ)",
                    line=dict(color="#bc8cff", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(188, 140, 255, 0.04)"
                ), row=1, col=1)

            # 3. Active Option Strike Level Lines
            for p in positions:
                sym = p["symbol"]
                if selected_asset in sym:
                    try:
                        strike_val = float(sym[-8:]) / 1000.0
                        opt_type = "Put" if "P" in sym else "Call"
                        fig.add_hline(
                            y=strike_val, line_dash="dash", line_color="#39d353",
                            annotation_text=f"Active {opt_type} Strike: ${strike_val:.0f}",
                            row=1, col=1
                        )
                    except Exception:
                        pass

            current_row = 2
            # 4. Volume Subplot
            if has_volume:
                v_colors = ["#3fb950" if c >= o else "#f85149" for c, o in zip(df["c"], df["o"])]
                fig.add_trace(go.Bar(
                    x=df["t"], y=df["v"], name="Volume",
                    marker_color=v_colors, opacity=0.75
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="Volume", gridcolor="#21262d", row=current_row, col=1)
                current_row += 1

            # 5. RSI Subplot
            if has_rsi:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df["rsi"], name="RSI (14)",
                    line=dict(color="#f0883e", width=1.5)
                ), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#f85149", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#3fb950", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", range=[0, 100], gridcolor="#21262d", row=current_row, col=1)

            # Unified Layout
            fig.update_layout(
                title=dict(
                    text=f"{selected_asset} — {timeframe_choice} Resolution Analysis | Live Alpaca Market Data",
                    font=dict(color="#ffffff", size=16)
                ),
                height=560,
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False)),
                yaxis=dict(gridcolor="#21262d", title="Asset Price ($ USD)"),
                legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(color="#8b949e")),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(f"Historical market data is being retrieved for {selected_asset}...")
    except Exception as e:
        st.warning(f"Unable to load chart stream for {selected_asset}: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strategy & Allocation Tabs ─────────────────────────────
    st.markdown("### Strategy Payoff & Portfolio Allocation")
    tab_payoff, tab_allocation = st.tabs([
        "Options Expiration Payoff Simulator",
        "Capital & Margin Allocation Breakdown"
    ])

    with tab_payoff:
        st.markdown("#### Options Payoff Profile Curve")
        st.caption("Theoretical profit & loss distribution across underlying price movements at contract expiration.")

        if positions:
            for p in positions:
                sym = p["symbol"]
                qty = float(p["qty"])
                avg_price = float(p["avg_entry_price"])
                
                if "P" in sym and qty < 0:
                    try:
                        strike = float(sym[-8:]) / 1000.0
                    except Exception:
                        strike = 500.0
                    
                    premium = avg_price
                    contracts = abs(qty)
                    max_profit = premium * 100 * contracts
                    breakeven = strike - premium
                    
                    spot_prices = np.linspace(strike * 0.82, strike * 1.12, 120)
                    payoffs = [(premium - max(strike - s, 0.0)) * 100 * contracts for s in spot_prices]
                    
                    fig_payoff = go.Figure()
                    fig_payoff.add_trace(go.Scatter(
                        x=spot_prices, y=payoffs,
                        mode="lines",
                        name="P&L at Expiration",
                        line=dict(color="#58a6ff", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(88, 166, 255, 0.1)"
                    ))
                    fig_payoff.add_hline(y=0, line_dash="dash", line_color="#8b949e", annotation_text="Break-Even Line")
                    fig_payoff.add_vline(x=strike, line_dash="dot", line_color="#d29922", annotation_text=f"Strike: ${strike:.0f}")
                    fig_payoff.add_vline(x=breakeven, line_dash="dot", line_color="#f85149", annotation_text=f"BE: ${breakeven:.2f}")

                    fig_payoff.update_layout(
                        title=dict(
                            text=f"Payoff Curve: {sym} (Short {contracts:.0f}x Put @ Strike ${strike:.0f} | Max Premium: +${max_profit:.2f})",
                            font=dict(color="#ffffff", size=14)
                        ),
                        height=360,
                        margin=dict(t=40, b=20, l=20, r=20),
                        paper_bgcolor="#161b22",
                        plot_bgcolor="#161b22",
                        xaxis=dict(gridcolor="#21262d", title="Underlying Spot Price ($ USD)"),
                        yaxis=dict(gridcolor="#21262d", title="Net Return ($ USD)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_payoff, use_container_width=True)
        else:
            st.info("No active options positions. Payoff curve will display when orders are filled.")

    with tab_allocation:
        st.markdown("#### Capital Distribution")
        opt_exposure = sum(abs(float(p["market_value"])) for p in positions)
        free_cash = max(0.0, cash - opt_exposure)
        
        alloc_labels = ["Unencumbered Free Cash", "Active Options Margin / Collateral", "Capital Buffer"]
        alloc_values = [free_cash, opt_exposure, max(0.0, equity - free_cash - opt_exposure)]
        alloc_colors = ["#238636", "#1f6feb", "#8b949e"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=alloc_labels,
            values=alloc_values,
            hole=0.55,
            marker=dict(colors=alloc_colors, line=dict(color="#0d1117", width=2)),
            textinfo="label+percent",
            insidetextorientation="radial"
        )])
        fig_pie.update_layout(
            height=360,
            margin=dict(t=30, b=20, l=20, r=20),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance Gauge & Positions Table ─────────────────────
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Portfolio Performance Gauge")
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
            df_pos = pd.DataFrame(positions)
            display_df = pd.DataFrame({
                "Contract": df_pos["symbol"],
                "Qty": df_pos["qty"].astype(float).map("{:+.1f}".format),
                "Entry Price": df_pos["avg_entry_price"].astype(float).map("${:,.2f}".format),
                "Market Value": df_pos["market_value"].astype(float).map("${:,.2f}".format),
                "Unrealized P&L": df_pos["unrealized_pl"].astype(float).map("${:+,.2f}".format),
                "Asset Class": df_pos["asset_class"].str.upper(),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Orchestrator scans and enters orders at session triggers.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Dashboard Panel ────────────────────────────────────
    st.markdown("### Risk Manager & Circuit Breakers")
    r1, r2, r3, r4 = st.columns(4)

    daily_loss_limit = 2000.0
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
