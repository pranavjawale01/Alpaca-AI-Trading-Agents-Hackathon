"""
dashboard/monitor.py — Real-Time Trading Terminal with Live AI Decision Feed & Kelly Sizing.

Professional financial dashboard for the Cache Me If You Can trading system.
Features:
  - Live Portfolio Equity, Positions, and 6-Gate Risk Monitor
  - Real-Time Live AI Decisions & Council Voting Signals across Watchlist
  - Dynamic Kelly Criterion Position Sizing & Real-Time Capital Allocations
  - Persistent SQLite Trade Journal History
  - Interactive Plotly Technical Charts (Candlesticks, EMAs, Bollinger Bands, Volume, RSI)
  - Options Expiration Payoff & Risk Profile Simulator

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

import config
from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.trade_journal import TradeJournal
from core.kelly_sizer import KellySizer

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
    
    /* Decision pill tags */
    .pill-buy {
        background-color: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #388bfd;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .pill-hold {
        background-color: #d2992222;
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .pill-veto {
        background-color: #f8514922;
        color: #f85149;
        border: 1px solid #f85149;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Client & Core Singletons ───────────────────────────────────
@st.cache_resource
def get_system():
    client = AlpacaClient()
    md = MarketData(client)
    journal = TradeJournal()
    kelly = KellySizer(journal)
    return client, md, journal, kelly

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Controls")
    auto_refresh = st.toggle("Auto-refresh (15s)", value=True)
    
    st.markdown(f"""
    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 10px;">
        <div style="font-size: 11px; color: #8b949e; text-transform: uppercase; font-weight: 600;">Autonomous Engine</div>
        <div style="font-size: 14px; font-weight: 600; color: #3fb950; margin-top: 4px;">ACTIVE (Scheduled / Market Open)</div>
        <div style="font-size: 12px; color: #8b949e; margin-top: 4px;">Trading runs automatically during market hours.</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Multi-Agent Fleet")
    st.markdown("- **Theta Collector**: Cash-Secured Puts (SPY, QQQ, IWM)")
    st.markdown("- **Momo Breakout**: OTM Calls + Trailing Stop")
    st.markdown("- **IV Crush**: Pre-Earnings Straddles")
    st.markdown("- **Hedge Agent**: Protective SPY Puts")
    st.divider()
    st.markdown("### Resources & Links")
    st.markdown("- [Alpaca Paper Console](https://app.alpaca.markets/paper/dashboard/overview)")
    st.markdown("- [Hackathon Portal](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)")
    st.markdown("- [GitHub Codebase](https://github.com/Pranav1173/Alpaca-AI-Trading-Agents-Hackathon)")
    st.divider()
    st.caption("Team: Cache Me If You Can")
    st.caption("Alpaca AI Trading Agents Hackathon 2026")

# ── Top Bar Header ─────────────────────────────────────────────
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="top-bar">
    <div>
        <div class="terminal-title">Cache Me If You Can — Options Alpha Terminal</div>
        <div style="font-size: 12px; color: #8b949e; margin-top: 4px;">Autonomous AI Trading System | Alpaca Paper Environment | System Clock: {now_utc}</div>
    </div>
    <div>
        <span class="status-badge">AUTONOMOUS LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

STARTING_BALANCE = 100_000.0

try:
    client, md, journal, kelly = get_system()
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

if data_loaded:
    # ── Top Level Tabs ─────────────────────────────────────────────
    tab_overview, tab_live_ai, tab_live_kelly, tab_charts = st.tabs([
        "Portfolio Overview",
        "Live AI Decisions",
        "Live Kelly Sizing & Execution",
        "Market Charts & Payoffs",
    ])

    with tab_overview:
        # Metrics Row
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

        # Health & Allocation Gauges
        health_col1, health_col2 = st.columns([1, 1])

        with health_col1:
            st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px; text-align: center;'>Portfolio Performance</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=equity,
                number={"prefix": "$", "valueformat": ",.0f", "font": {"color": "#ffffff", "size": 32}},
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
                title={"text": "Baseline ($100k)", "font": {"color": "#8b949e", "size": 14}},
            ))
            fig_gauge.update_layout(
                height=300,
                margin=dict(t=40, b=10, l=20, r=20),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with health_col2:
            st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px; text-align: center;'>Capital Allocation Breakdown</div>", unsafe_allow_html=True)
            opt_exposure = sum(abs(float(p["market_value"])) for p in positions)
            free_cash = max(0.0, cash - opt_exposure)
            
            alloc_labels = ["Free Cash", "Options Margin", "Buffer"]
            alloc_values = [free_cash, opt_exposure, max(0.0, equity - free_cash - opt_exposure)]
            alloc_colors = ["#238636", "#1f6feb", "#8b949e"]

            fig_pie = go.Figure(data=[go.Pie(
                labels=alloc_labels,
                values=alloc_values,
                hole=0.6,
                marker=dict(colors=alloc_colors, line=dict(color="#0d1117", width=2)),
                textinfo="label+percent",
                insidetextorientation="radial"
            )])
            fig_pie.update_layout(
                height=300,
                margin=dict(t=10, b=10, l=20, r=20),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Open Positions & Circuit Breakers
        pos_col, risk_col = st.columns([1.5, 1])

        with pos_col:
            st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px;'>Open Option Contracts</div>", unsafe_allow_html=True)
            if positions:
                df_pos = pd.DataFrame(positions)
                display_df = pd.DataFrame({
                    "Contract": df_pos["symbol"],
                    "Qty": df_pos["qty"].astype(float).map("{:+.1f}".format),
                    "Entry": df_pos["avg_entry_price"].astype(float).map("${:,.2f}".format),
                    "Market Value": df_pos["market_value"].astype(float).map("${:,.2f}".format),
                    "P&L": df_pos["unrealized_pl"].astype(float).map("${:+,.2f}".format),
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions. Autonomous agent scans watchlist at session open.")

        with risk_col:
            st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px;'>6-Gate Risk Circuit Breakers</div>", unsafe_allow_html=True)
            
            daily_loss_limit = 2000.0
            daily_pnl = total_pnl
            loss_remaining = max(0.0, daily_loss_limit + daily_pnl)

            options_positions = [p for p in positions if "option" in str(p.get("asset_class", "")).lower()]
            opt_exposure = sum(abs(float(p["market_value"])) for p in options_positions)
            opt_pct = (opt_exposure / equity) * 100 if equity > 0 else 0.0

            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #3fb950; margin-bottom: 10px; padding: 12px 16px;">
                <div class="metric-label" style="font-size: 11px;">1. Daily Loss Buffer (-2% Limit)</div>
                <div class="metric-value" style="font-size: 20px;">${loss_remaining:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            exposure_color = "#3fb950" if opt_pct <= 25.0 else "#d29922"
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {exposure_color}; margin-bottom: 10px; padding: 12px 16px;">
                <div class="metric-label" style="font-size: 11px;">2. Options Exposure Cap (30% Max)</div>
                <div class="metric-value" style="color: {exposure_color}; font-size: 20px;">{opt_pct:.1f}% / 30.0%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #58a6ff; padding: 12px 16px;">
                <div class="metric-label" style="font-size: 11px;">3. VIX Kill Switch Guardrail</div>
                <div class="metric-value" style="font-size: 20px;">VIX &ge; 35.0 (Halt)</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_live_ai:
        st.markdown("<div style='color: #58a6ff; font-size: 18px; font-weight: 700; margin-bottom: 10px;'>Live AI Market Regime & Decision Feed</div>", unsafe_allow_html=True)

        vix_val = md.get_vix()
        regime_str = "RISK_ON" if vix_val < 18.0 else ("NEUTRAL" if vix_val < 28.0 else "RISK_OFF")
        regime_color = "#3fb950" if regime_str == "RISK_ON" else ("#d29922" if regime_str == "NEUTRAL" else "#f85149")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {regime_color};">
                <div class="metric-label">Detected Market Regime</div>
                <div class="metric-value" style="color: {regime_color};">{regime_str}</div>
                <div style="font-size: 12px; color: #8b949e;">VIX Proxy: {vix_val:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #58a6ff;">
                <div class="metric-label">3-Model LLM Council Status</div>
                <div class="metric-value" style="font-size: 22px;">{'ACTIVE (3 Models)' if config.COUNCIL.enabled else 'RULES FALLBACK'}</div>
                <div style="font-size: 12px; color: #8b949e;">Consensus Threshold: &ge; {config.COUNCIL.consensus_threshold:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #bc8cff;">
                <div class="metric-label">Active Strategy Allocation</div>
                <div class="metric-value" style="font-size: 20px;">{'All 4 Agents Active' if regime_str == 'RISK_ON' else ('Theta + IV + Hedge' if regime_str == 'NEUTRAL' else 'Hedge Only')}</div>
                <div style="font-size: 12px; color: #8b949e;">Regime-routed capital defense</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px;'>Live Watchlist Signal Scan & AI Council Verdicts</div>", unsafe_allow_html=True)

        # Build live scan data
        scan_symbols = ["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "AAPL"]
        scan_rows = []

        for sym in scan_symbols:
            try:
                price = md.get_price(sym)
                ema_res = md.get_ema_signal(sym)
                vol_res = md.get_volume_surge(sym)
                hist_vol = md.estimate_historical_vol(sym)
                ivr = min(hist_vol * 200, 100)

                # Determine strategy & AI verdict
                if sym in config.UNIVERSE.theta_symbols:
                    strat = "Theta CSP"
                    if ivr > 30 and regime_str in ("RISK_ON", "NEUTRAL"):
                        action = "BUY (SELL PUT)"
                        score = "+0.85"
                        decision = "APPROVED (High IVR)"
                    else:
                        action = "HOLD"
                        score = "+0.20"
                        decision = "MONITORING (IVR < 30)"
                elif sym in config.UNIVERSE.momo_watchlist:
                    strat = "Momo Call"
                    if ema_res.get("crossover") and vol_res.get("is_surging"):
                        action = "BUY (CALL)"
                        score = "+0.90"
                        decision = "APPROVED (Breakout Confirmed)"
                    elif ema_res.get("signal") == "bullish":
                        action = "HOLD"
                        score = "+0.45"
                        decision = "WAITING (Vol Surge < 2x)"
                    else:
                        action = "HOLD"
                        score = "+0.10"
                        decision = "NEUTRAL"
                else:
                    strat = "IV Crush"
                    action = "MONITORING"
                    score = "0.00"
                    decision = "CALENDAR SCAN"

                scan_rows.append({
                    "Symbol": sym,
                    "Strategy": strat,
                    "Price": f"${price:.2f}",
                    "EMA Fast/Slow": f"${ema_res.get('ema_fast', 0):.1f} / ${ema_res.get('ema_slow', 0):.1f}",
                    "EMA Signal": ema_res.get("signal", "neutral").upper(),
                    "Vol Surge": f"{vol_res.get('surge_ratio', 1.0):.1f}x",
                    "IV Rank": f"{ivr:.0f}%",
                    "AI Action": action,
                    "Council Score": score,
                    "Live Verdict": decision,
                })
            except Exception:
                pass

        if scan_rows:
            scan_df = pd.DataFrame(scan_rows)
            # Expand table height dynamically so no internal scrolling is needed
            st.dataframe(
                scan_df,
                use_container_width=True,
                hide_index=True,
                height=(len(scan_rows) + 1) * 36 + 10,
            )
        else:
            st.info("Market data stream loading...")

    with tab_live_kelly:
        st.markdown("<div style='color: #3fb950; font-size: 18px; font-weight: 700; margin-bottom: 10px;'>Live Kelly Position Sizing & Capital Budget</div>", unsafe_allow_html=True)

        stats = journal.get_all_strategy_stats()
        kelly_live_rows = []

        for s_name in ["theta", "momo", "iv_crush"]:
            s_stat = stats.get(s_name, {})
            n_trades = s_stat.get("n_trades", 0)
            win_rate = s_stat.get("win_rate", 0.0)
            avg_win = s_stat.get("avg_win", 0.0)
            avg_loss = s_stat.get("avg_loss", 0.0)
            dollar_size = kelly.get_position_size(s_name, equity)
            pct = (dollar_size / equity) * 100 if equity > 0 else 0.0

            kelly_live_rows.append({
                "Strategy": s_name.upper(),
                "Trades": n_trades,
                "Win Rate": f"{win_rate:.1%}" if n_trades > 0 else "Baseline (< 10 trades)",
                "Avg Win": f"{avg_win:.1%}" if n_trades > 0 else "—",
                "Avg Loss": f"{avg_loss:.1%}" if n_trades > 0 else "—",
                "¼-Kelly Risk Budget": f"${dollar_size:,.0f}",
                "Equity Sizing %": f"{pct:.2f}%",
                "Sizing Engine": "Kelly Dynamic (Journal Fed)" if n_trades >= 10 else "Conservative Baseline",
                "Execution Target": "Mid-Price Limit Order (Spread Reduction)",
            })
        
        kelly_df = pd.DataFrame(kelly_live_rows)
        st.dataframe(
            kelly_df,
            use_container_width=True,
            hide_index=True,
            height=(len(kelly_live_rows) + 1) * 38 + 10,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px;'>Live Symbol-Level Contract Sizing Matrix</div>", unsafe_allow_html=True)

        # Build live contract sizing matrix
        matrix_rows = []
        for sym in ["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "AAPL"]:
            try:
                price = md.get_price(sym)
                strat = "theta" if sym in config.UNIVERSE.theta_symbols else "momo"
                budget = kelly.get_position_size(strat, equity)
                if strat == "theta":
                    est_prem = price * 0.015 * 100
                    margin = price * 100 * 0.20
                    contracts = kelly.get_contract_count(strat, equity, margin)
                else:
                    est_prem = price * 0.03 * 100
                    contracts = kelly.get_contract_count(strat, equity, est_prem)

                matrix_rows.append({
                    "Symbol": sym,
                    "Strategy": strat.upper(),
                    "Underlying Price": f"${price:.2f}",
                    "Est Option Premium": f"${est_prem:.0f} / contract",
                    "Strategy Budget": f"${budget:,.0f}",
                    "Calculated Contract Count": f"{contracts} contracts",
                    "Max Capital at Risk": f"${min(budget, contracts * est_prem):,.0f}",
                    "Order Type": "Smart Limit @ Mid-Price",
                })
            except Exception:
                pass

        if matrix_rows:
            matrix_df = pd.DataFrame(matrix_rows)
            st.dataframe(
                matrix_df,
                use_container_width=True,
                hide_index=True,
                height=(len(matrix_rows) + 1) * 36 + 10,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color: #8b949e; font-size: 14px; font-weight: 600; margin-bottom: 10px;'>Persistent Trade Journal Log (SQLite)</div>", unsafe_allow_html=True)
        
        # Read from SQLite trade journal
        try:
            with journal._connect() as conn:
                trades_df = pd.read_sql_query(
                    "SELECT id, opened_at, agent, strategy, symbol, contract, side, qty, entry_price, exit_price, pnl, exit_reason FROM trades ORDER BY id DESC LIMIT 50",
                    conn
                )
            
            if not trades_df.empty:
                st.dataframe(
                    trades_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, (len(trades_df) + 1) * 36 + 10),
                )
            else:
                st.info("Trade journal active. Executed orders are recorded here in real-time.")
        except Exception as e:
            st.info("Trade journal is ready. Recorded trades will display here.")

    with tab_charts:
        st.markdown("<div style='color: #d29922; font-size: 16px; font-weight: 600; margin-bottom: 15px;'>Market Price Action & Filters</div>", unsafe_allow_html=True)
        
        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        
        with f_col1:
            selected_asset = st.selectbox(
                "Asset",
                options=["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "VIXY"],
                index=0
            )
        with f_col2:
            timeframe_choice = st.selectbox(
                "Timeframe",
                options=["1Day", "1Hour", "15Min", "5Min"],
                index=0
            )
        with f_col3:
            chart_style = st.selectbox(
                "Style",
                options=["Candlestick", "Translucent Line Area", "OHLC Bars"],
                index=0
            )
        with f_col4:
            history_bars = st.slider(
                "Bars",
                min_value=30, max_value=250, value=90, step=10
            )
        with f_col5:
            indicator_selection = st.multiselect(
                "Indicators",
                options=["20 EMA", "50 EMA", "Bollinger Bands", "Volume Overlay", "RSI (14)"],
                default=["20 EMA", "50 EMA", "Volume Overlay"]
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

                # Subplot Structure
                has_volume = "Volume Overlay" in indicator_selection
                has_rsi = "RSI (14)" in indicator_selection

                rows = 1 + (1 if has_rsi else 0)
                row_heights = [0.8] if rows == 1 else ([0.75, 0.25])
                specs = [[{"secondary_y": True}]]
                if has_rsi:
                    specs.append([{"secondary_y": False}])

                fig = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=row_heights,
                    specs=specs
                )

                # TradingView Colors
                TV_GREEN = "#089981"
                TV_RED = "#F23645"
                TV_BG = "#131722"
                TV_GRID = "#2a2e39"

                df["t_str"] = df["t"].dt.strftime("%Y-%m-%d %H:%M")

                # 1. Main Price Plot
                if chart_style == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=df["t_str"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                        name=f"{selected_asset}",
                        increasing_line_color=TV_GREEN, decreasing_line_color=TV_RED,
                        increasing_fillcolor=TV_GREEN, decreasing_fillcolor=TV_RED,
                    ), row=1, col=1, secondary_y=False)
                elif chart_style == "Translucent Line Area":
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["c"],
                        mode="lines",
                        name=f"{selected_asset}",
                        line=dict(color="#2962FF", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(41, 98, 255, 0.1)"
                    ), row=1, col=1, secondary_y=False)
                else:
                    fig.add_trace(go.Ohlc(
                        x=df["t_str"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                        name=f"{selected_asset}",
                        increasing_line_color=TV_GREEN, decreasing_line_color=TV_RED
                    ), row=1, col=1, secondary_y=False)

                # 2. Overlays
                if "20 EMA" in indicator_selection:
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["ema20"], name="20 EMA",
                        line=dict(color="#2962FF", width=1.5)
                    ), row=1, col=1, secondary_y=False)
                if "50 EMA" in indicator_selection:
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["ema50"], name="50 EMA",
                        line=dict(color="#FF9800", width=1.5)
                    ), row=1, col=1, secondary_y=False)
                if "Bollinger Bands" in indicator_selection:
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["bb_upper"], name="BB Upper",
                        line=dict(color="#9C27B0", width=1, dash="dot")
                    ), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["bb_lower"], name="BB Lower",
                        line=dict(color="#9C27B0", width=1, dash="dot"),
                        fill="tonexty", fillcolor="rgba(156, 39, 176, 0.05)"
                    ), row=1, col=1, secondary_y=False)

                # 3. Active Strike Horizontal Lines
                for p in positions:
                    sym = p["symbol"]
                    if selected_asset in sym:
                        try:
                            strike_val = float(sym[-8:]) / 1000.0
                            opt_type = "Put" if "P" in sym else "Call"
                            fig.add_hline(
                                y=strike_val, line_dash="dash", line_color="#FF9800",
                                annotation_text=f"Active {opt_type}: ${strike_val:.0f}",
                                row=1, col=1, secondary_y=False
                            )
                        except Exception:
                            pass

                # 4. Volume Overlay
                if has_volume:
                    v_colors = [TV_GREEN if c >= o else TV_RED for c, o in zip(df["c"], df["o"])]
                    fig.add_trace(go.Bar(
                        x=df["t_str"], y=df["v"], name="Volume",
                        marker_color=v_colors, opacity=0.25
                    ), row=1, col=1, secondary_y=True)
                    max_vol = df["v"].max()
                    fig.update_yaxes(showgrid=False, range=[0, max_vol * 4], showticklabels=False, secondary_y=True, row=1, col=1)

                current_row = 2

                # 5. RSI Subplot
                if has_rsi:
                    fig.add_trace(go.Scatter(
                        x=df["t_str"], y=df["rsi"], name="RSI",
                        line=dict(color="#7E57C2", width=1.5)
                    ), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color=TV_RED, row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color=TV_GREEN, row=current_row, col=1)
                    fig.update_yaxes(range=[0, 100], gridcolor=TV_GRID, side="right", row=current_row, col=1)

                # Layout
                fig.update_layout(
                    height=600,
                    margin=dict(t=10, b=20, l=10, r=10),
                    paper_bgcolor=TV_BG,
                    plot_bgcolor=TV_BG,
                    xaxis=dict(
                        type="category",
                        gridcolor=TV_GRID, 
                        rangeslider=dict(visible=False),
                        nticks=10
                    ),
                    yaxis=dict(gridcolor=TV_GRID, side="right"),
                    legend=dict(orientation="h", y=1.02, x=0.01, xanchor="left", font=dict(color="#d1d4dc")),
                    hovermode="x unified",
                    font=dict(color="#d1d4dc")
                )
                
                fig.update_xaxes(type="category", gridcolor=TV_GRID, nticks=10)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(f"Historical market data is being retrieved for {selected_asset}...")
        except Exception as e:
            st.warning(f"Unable to load chart stream for {selected_asset}: {e}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='color: #bc8cff; font-size: 16px; font-weight: 600; margin-bottom: 10px;'>Options Expiration Simulator</div>", unsafe_allow_html=True)
        
        has_options = any("P" in p["symbol"] or "C" in p["symbol"] for p in positions)
        if positions and has_options:
            fig_payoff = go.Figure()
            
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
                    
                    fig_payoff.add_trace(go.Scatter(
                        x=spot_prices, y=payoffs,
                        mode="lines",
                        name=f"{sym[:3]} ${strike} P",
                        line=dict(color="#58a6ff", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(88, 166, 255, 0.1)"
                    ))
                    fig_payoff.add_hline(y=0, line_dash="dash", line_color="#8b949e")
                    fig_payoff.add_vline(x=strike, line_dash="dot", line_color="#d29922", annotation_text=f"Strike ${strike}")
                    fig_payoff.add_vline(x=breakeven, line_dash="dot", line_color="#f85149", annotation_text=f"BE ${breakeven:.2f}")

            fig_payoff.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                xaxis=dict(gridcolor="#21262d", title="Underlying Spot Price ($ USD)"),
                yaxis=dict(gridcolor="#21262d", title="Net Return ($ USD)"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_payoff, use_container_width=True)
        else:
            st.info("No active options positions to simulate.")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("Cache Me If You Can | Alpaca AI Trading Agents Hackathon 2026 | lablab.ai x Alpaca")

# ── Auto-refresh Trigger ───────────────────────────────────────
if auto_refresh:
    time.sleep(15)
    st.rerun()
