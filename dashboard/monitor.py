"""
dashboard/monitor.py — Real-Time Trading Terminal with Live AI Decision Feed & Kelly Sizing.

Professional financial dashboard for the Cache Me If You Can trading system.
Features:
  - Live Portfolio Equity, Positions, and 6-Gate Risk Monitor
  - Real-Time Live AI Decisions & 3-Agent Quantitative Strategy Council Signals
  - Dynamic Kelly Criterion Position Sizing & Real-Time Capital Allocations
  - Persistent SQLite Trade Journal History
  - Interactive Plotly Technical Chart (Candlesticks, EMAs, Bollinger Bands, Volume, RSI)

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

import threading
from collections import deque
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# Automatically bridge Streamlit Cloud Secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

import config
from core.alpaca_client import AlpacaClient
from core.market_data import MarketData
from core.trade_journal import TradeJournal
from core.kelly_sizer import KellySizer
from core.opportunity_scorer import OpportunityScorer
from agents.orchestrator import Orchestrator

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Cache Me If You Can — Options Alpha Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistent Keep-Alive (Prevents Streamlit Cloud Hibernation) ──
if hasattr(st, "html"):
    st.html("""
    <script>
        setInterval(function() {
            try {
                fetch(window.location.href, { method: 'HEAD', mode: 'no-cors' });
                console.log('[CacheMe] Keep-alive ping sent at ' + new Date().toISOString());
            } catch(e) {}
        }, 270000);
    </script>
    """, unsafe_allow_html=True)
else:
    components.html("""
    <script>
        setInterval(function() {
            try {
                fetch(window.location.href, { method: 'HEAD', mode: 'no-cors' });
                console.log('[CacheMe] Keep-alive ping sent at ' + new Date().toISOString());
            } catch(e) {}
        }, 270000);
    </script>
    """, height=0)

# ── Modern Minimalist CSS Theme ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #090d16 100%);
        color: #f1f5f9;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #090d16;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Top Hero Command Bar */
    .hero-bar {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 18px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0;
    }
    .hero-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* Pulsing live indicator */
    .pulsing-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* High-End Glassmorphic KPI Cards */
    .kpi-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.6) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 18px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.35);
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-1px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.3px;
    }
    .kpi-pill-pos {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 3px;
    }
    .kpi-pill-neg {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 3px;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
        margin-top: 3px;
    }

    /* Circuit Breaker Progress Bars */
    .circuit-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .progress-bg {
        width: 100%;
        height: 5px;
        background: #1e293b;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        border-color: rgba(255, 255, 255, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
    }

    /* Streamlit DataFrame styling override */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── Background Autonomous Auto-Pilot Engine ─────────────────────
@st.cache_resource
def get_background_engine():
    session_logs = deque(maxlen=150)
    engine_lock = threading.Lock()
    engine_state = {
        "status": "RUNNING (Market Scheduler)",
        "last_run": "Never",
        "last_regime": "N/A",
        "last_pnl": 0.0,
        "runs_count": 0,
        "is_busy": False,
        "last_error": None,
    }

    def _auto_pilot_worker():
        hybrid = getattr(config, "HYBRID", None)
        interval_mins = getattr(hybrid, "session_interval_minutes", 10) if hybrid else 10
        
        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                is_weekday = now_utc.weekday() < 5
                market_open = now_utc.replace(hour=13, minute=30, second=0, microsecond=0)
                market_close = now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
                is_market_hours = is_weekday and (market_open <= now_utc <= market_close)

                if is_market_hours:
                    with engine_lock:
                        engine_state["is_busy"] = True
                        engine_state["status"] = "RUNNING (Trading Session Active)"

                    orch = Orchestrator()
                    summary = orch.run_session()

                    pnl = summary.get("daily_pnl", 0.0)
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                    with engine_lock:
                        engine_state["last_run"] = now_str
                        engine_state["last_regime"] = summary.get("regime", "neutral").upper()
                        engine_state["last_pnl"] = pnl
                        engine_state["runs_count"] += 1
                        engine_state["is_busy"] = False
                        engine_state["status"] = "WAITING (Next Interval)"
                        session_logs.appendleft({
                            "time": now_str,
                            "type": "AUTO_PILOT",
                            "regime": engine_state["last_regime"],
                            "pnl": pnl,
                            "actions": summary.get("actions_taken", 0),
                            "details": summary.get("actions", []),
                        })
                else:
                    with engine_lock:
                        engine_state["is_busy"] = False
                        engine_state["status"] = "MARKET_CLOSED (Awaiting 13:30 UTC)"

            except Exception as e:
                with engine_lock:
                    engine_state["is_busy"] = False
                    engine_state["last_error"] = str(e)
                    engine_state["status"] = f"ERROR ({str(e)[:30]})"

            time.sleep(interval_mins * 60)

    t = threading.Thread(target=_auto_pilot_worker, daemon=True, name="CacheMeAutoPilot")
    t.start()
    return engine_state, session_logs, engine_lock

bg_state, bg_logs, bg_lock = get_background_engine()

# ── Alpaca & System Initialization ─────────────────────────────
@st.cache_resource
def get_system():
    client = AlpacaClient()
    md = MarketData(client)
    journal = TradeJournal()
    kelly = KellySizer(journal=journal)
    return client, md, journal, kelly

# ── Sidebar Controls ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size: 14px; font-weight: 800; color: #ffffff; letter-spacing: -0.2px; margin-bottom: 8px;">
        ALPHA CONTROLLER
    </div>
    """, unsafe_allow_html=True)
    
    auto_refresh = st.toggle("Auto-refresh Terminal (30s)", value=True)
    
    status_color = "#10b981" if not bg_state["is_busy"] else "#f59e0b"
    st.markdown(f"""
    <div class="circuit-card" style="border-left: 3px solid {status_color};">
        <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Engine State</div>
        <div style="font-size: 12px; font-weight: 800; color: {status_color}; margin-top: 2px;">{bg_state['status']}</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px;">Last: {bg_state['last_run']}</div>
        <div style="font-size: 11px; color: #64748b;">Completed: {bg_state['runs_count']} sessions</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Autonomous Session Now", width="stretch", disabled=bg_state["is_busy"]):
        with st.status("Executing Multi-Agent Autonomous Session...", expanded=True) as status_box:
            st.write("Reading market feeds & querying 3-Agent Quantitative Council...")
            try:
                orch = Orchestrator()
                st.write(f"Detected VIX Regime: **{orch.rm.current_vix:.1f}**")
                st.write("Executing strategy fleet: Hedge -> Theta -> Momo -> IV Crush...")
                summary = orch.run_session()
                pnl = summary.get("daily_pnl", 0.0)
                pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                st.write(f"Session complete! Actions taken: **{summary.get('actions_taken', 0)}** | Daily P&L: **{pnl_str}**")
                status_box.update(label=f"Session Complete ({pnl_str})", state="complete", expanded=False)
                
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with bg_lock:
                    bg_state["last_run"] = now_str
                    bg_state["last_regime"] = summary.get("regime", "neutral").upper()
                    bg_state["last_pnl"] = pnl
                    bg_state["runs_count"] += 1
                    bg_logs.appendleft({
                        "time": now_str,
                        "type": "MANUAL_TRIGGER",
                        "regime": bg_state["last_regime"],
                        "pnl": pnl,
                        "actions": summary.get("actions_taken", 0),
                        "details": summary.get("actions", []),
                    })
                st.rerun()
            except Exception as e:
                st.error(f"Trading session error: {e}")
                status_box.update(label="Session Failed", state="error", expanded=True)

    st.divider()
    st.markdown("### Strategy Fleet")
    st.markdown("- **Theta Collector**: Cash-Secured Puts on SPY/QQQ")
    st.markdown("- **Momo Breakout**: Dual-directional Calls & Puts")
    st.markdown("- **IV Crush**: Pre-Earnings Straddles")
    st.markdown("- **Hedge Agent**: SPY Put Portfolio Defense")
    st.divider()
    st.caption("Cache Me If You Can | Alpaca Hackathon 2026")

# ── Top Command Bar ────────────────────────────────────────────
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class="hero-bar">
    <div>
        <div class="hero-title">
            <span>CACHE ME IF YOU CAN</span>
            <span style="font-size: 11px; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.12); padding: 2px 8px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3);">3-AGENT QUANT COUNCIL</span>
        </div>
        <div class="hero-sub">Autonomous Options Trading System | Alpaca Paper Environment | Clock: {now_utc}</div>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); padding: 4px 10px; border-radius: 14px; font-size: 11px; font-weight: 700; color: #10b981; display: flex; align-items: center;">
            <span class="pulsing-dot"></span>LIVE AUTONOMOUS
        </div>
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
    st.info("Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are configured in .env or Streamlit Cloud Secrets.")
    data_loaded = False

if data_loaded:
    # ── 3 Accessible, Non-Scrolling Tabs ───────────────────────────
    tab_trading, tab_chart, tab_execution = st.tabs([
        "Trading & Signals",
        "Market Analytics",
        "Sizing & Execution",
    ])

    # ═════════════════════════════════════════════════════════════════
    # TAB 1: TRADING & SIGNALS
    # ═════════════════════════════════════════════════════════════════
    with tab_trading:
        # Top KPI Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        pnl_class = "kpi-pill-pos" if total_pnl >= 0 else "kpi-pill-neg"
        pnl_sign = "+" if total_pnl >= 0 else ""

        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Portfolio Equity</div>
                <div class="kpi-value">${equity:,.2f}</div>
                <div><span class="{pnl_class}">{pnl_sign}${total_pnl:,.2f} ({pnl_sign}{total_pnl_pct:.2f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Free Cash Balance</div>
                <div class="kpi-value">${cash:,.2f}</div>
                <div class="kpi-sub">Collected Premium: +${max(0.0, cash - STARTING_BALANCE):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            opt_count = len(positions)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Active Positions</div>
                <div class="kpi-value">{opt_count} <span style="font-size: 13px; color: #94a3b8; font-weight: 500;">Contracts</span></div>
                <div class="kpi-sub">Dynamic Kelly Sized</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Buying Power</div>
                <div class="kpi-value">${buying_power:,.2f}</div>
                <div class="kpi-sub">Account: {account_id[:8]}...</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Volatility Regime & Council State
        vix_val = md.get_vix()
        regime_str = "RISK_ON" if vix_val < 18.0 else ("NEUTRAL" if vix_val < 28.0 else "RISK_OFF")
        vix_color = "#10b981" if regime_str == "RISK_ON" else ("#f59e0b" if regime_str == "NEUTRAL" else "#ef4444")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid {vix_color};">
                <div class="kpi-label">Market Volatility Regime</div>
                <div class="kpi-value" style="color: {vix_color}; font-size: 20px;">{regime_str}</div>
                <div class="kpi-sub">VIX Proxy: {vix_val:.1f} | Adaptive Sizing Active</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #38bdf8;">
                <div class="kpi-label">3-Agent Quantitative Council</div>
                <div class="kpi-value" style="color: #38bdf8; font-size: 20px;">ACTIVE (3 AGENTS)</div>
                <div class="kpi-sub">Trend &bull; Volatility &bull; Risk | Unanimous Approval</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #a855f7;">
                <div class="kpi-label">Fleet Allocation</div>
                <div class="kpi-value" style="color: #a855f7; font-size: 20px;">{'All 4 Agents Active' if regime_str in ('RISK_ON', 'NEUTRAL') else 'Hedge Defense Only'}</div>
                <div class="kpi-sub">Greedy Multipliers Enabled (up to 2.0x Kelly)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # 2-Column Split: Open Positions (Left) & Risk Circuit Breakers (Right)
        col_pos, col_risk = st.columns([1.35, 1])

        with col_pos:
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>Open Positions</div>", unsafe_allow_html=True)
            if positions:
                df_pos = pd.DataFrame(positions)
                display_df = pd.DataFrame({
                    "Contract": df_pos["symbol"],
                    "Qty": df_pos["qty"].astype(float).map("{:+.1f}".format),
                    "Entry": df_pos["avg_entry_price"].astype(float).map("${:,.2f}".format),
                    "Market Value": df_pos["market_value"].astype(float).map("${:,.2f}".format),
                    "Unrealized P&L": df_pos["unrealized_pl"].astype(float).map("${:+,.2f}".format),
                })
                st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.3); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 24px; text-align: center;">
                    <div style="font-size: 13px; font-weight: 600; color: #cbd5e1;">No Open Positions</div>
                    <div style="font-size: 11px; color: #64748b; margin-top: 3px;">Fleet scanning watchlist in real-time. Unanimous council approval required.</div>
                </div>
                """, unsafe_allow_html=True)

        with col_risk:
            st.markdown("<div style='font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>6-Gate Risk Breakers</div>", unsafe_allow_html=True)
            
            daily_loss_limit = 2000.0
            daily_pnl = total_pnl
            loss_remaining = max(0.0, daily_loss_limit + daily_pnl)
            loss_pct = min(100.0, max(0.0, abs(min(0.0, daily_pnl)) / daily_loss_limit * 100.0))

            options_positions = [p for p in positions if "option" in str(p.get("asset_class", "")).lower()]
            opt_exposure = sum(abs(float(p["market_value"])) for p in options_positions)
            opt_pct = (opt_exposure / equity) * 100 if equity > 0 else 0.0
            exposure_fill = min(100.0, (opt_pct / 30.0) * 100.0)
            exposure_color = "#10b981" if opt_pct <= 20.0 else ("#f59e0b" if opt_pct <= 28.0 else "#ef4444")
            vix_fill = min(100.0, (vix_val / 35.0) * 100.0)

            st.markdown(f"""
            <div class="circuit-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #e2e8f0;">Daily Drawdown (-2% Cap)</span>
                    <span style="font-size: 11px; font-weight: 700; color: #10b981;">${loss_remaining:,.0f} Buffer</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {loss_pct:.1f}%; background: {'#10b981' if loss_pct < 70 else '#ef4444'};"></div>
                </div>
            </div>

            <div class="circuit-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #e2e8f0;">Options Exposure (30% Max)</span>
                    <span style="font-size: 11px; font-weight: 700; color: {exposure_color};">{opt_pct:.1f}% / 30%</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {exposure_fill:.1f}%; background: {exposure_color};"></div>
                </div>
            </div>

            <div class="circuit-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #e2e8f0;">VIX Volatility Guard (35.0 Cap)</span>
                    <span style="font-size: 11px; font-weight: 700; color: {vix_color};">{vix_val:.1f} ({regime_str})</span>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {vix_fill:.1f}%; background: {vix_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Watchlist Signal Scan & Quantitative Decisions
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>Watchlist Signal Scan & Quantitative Council Verdicts</div>", unsafe_allow_html=True)
        
        opp_scorer = OpportunityScorer()
        scan_symbols = ["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "AAPL"]
        scan_rows = []
        open_sym_list = [p["symbol"] for p in positions]

        for sym in scan_symbols:
            try:
                price = md.get_price(sym)
                ema_res = md.get_ema_signal(sym)
                vol_res = md.get_volume_surge(sym)
                hist_vol = md.estimate_historical_vol(sym)
                ivr = min(hist_vol * 200, 100)

                if sym in config.UNIVERSE.theta_symbols:
                    strat = "Theta CSP"
                    opp_val = opp_scorer.score({
                        "ivr": ivr, "vix": vix_val,
                        "ema_signal": "bullish", "strategy": "theta"
                    }, open_positions=open_sym_list, session_pnl=total_pnl)
                    
                    if ivr > 30 and regime_str in ("RISK_ON", "NEUTRAL"):
                        tier = "STRONG" if ivr >= 50 else "MODERATE"
                        size_mult = "1.00x" if tier == "STRONG" else "0.70x"
                        action = "SELL PUT"
                        decision = f"APPROVED ({tier})"
                    else:
                        tier = "PILOT" if ivr >= 20 else "VETO"
                        size_mult = "0.40x" if tier == "PILOT" else "0.00x"
                        action = "HOLD"
                        decision = f"MONITORING ({tier})"
                elif sym in config.UNIVERSE.momo_watchlist:
                    strat = "Momo Call"
                    opp_val = opp_scorer.score({
                        "ivr": ivr, "vix": vix_val,
                        "ema_signal": ema_res.get("signal", "neutral"),
                        "ema_crossover": ema_res.get("crossover", False),
                        "volume_surge_ratio": vol_res.get("surge_ratio", 1.0),
                        "strategy": "momo"
                    }, open_positions=open_sym_list, session_pnl=total_pnl)
                    
                    if ema_res.get("crossover") and vol_res.get("is_surging"):
                        tier = "STRONG"
                        size_mult = "1.00x"
                        action = "BUY CALL"
                        decision = "APPROVED (Breakout)"
                    elif ema_res.get("signal") == "bullish":
                        tier = "MODERATE"
                        size_mult = "0.70x"
                        action = "PILOT CALL"
                        decision = "PILOT ENTRY"
                    else:
                        tier = "VETO"
                        size_mult = "0.00x"
                        action = "HOLD"
                        decision = "NEUTRAL"
                else:
                    strat = "IV Crush"
                    opp_val = 1.0
                    tier = "MODERATE"
                    size_mult = "0.70x"
                    action = "CALENDAR"
                    decision = "CALENDAR SCAN"

                scan_rows.append({
                    "Symbol": sym,
                    "Strategy": strat,
                    "Price": f"${price:.2f}",
                    "EMA Signal": ema_res.get("signal", "neutral").upper(),
                    "Vol Surge": f"{vol_res.get('surge_ratio', 1.0):.1f}x",
                    "IV Rank": round(ivr, 1),
                    "Conviction": tier,
                    "Greed": f"{opp_val:.2f}x",
                    "Size": size_mult,
                    "AI Action": action,
                    "Live Verdict": decision,
                })
            except Exception:
                pass

        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            filter_opt = st.selectbox(
                "Filter Watchlist",
                ["All Watchlist Tickers", "Actionable / Approved Only", "Theta CSP Only", "Momentum Only"],
                label_visibility="collapsed",
            )

        filtered_rows = scan_rows
        if filter_opt == "Actionable / Approved Only":
            filtered_rows = [r for r in scan_rows if r["AI Action"] not in ("HOLD", "CALENDAR")]
        elif filter_opt == "Theta CSP Only":
            filtered_rows = [r for r in scan_rows if "Theta" in r["Strategy"]]
        elif filter_opt == "Momentum Only":
            filtered_rows = [r for r in scan_rows if "Momo" in r["Strategy"]]

        if filtered_rows:
            scan_df = pd.DataFrame(filtered_rows)
            st.dataframe(
                scan_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "IV Rank": st.column_config.ProgressColumn(
                        "IV Rank",
                        help="Implied Volatility Rank (0-100%)",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )
        else:
            st.info("No actionable signals matching current filter criteria. Autonomous engine continuously evaluates incoming ticks.")

    # ═════════════════════════════════════════════════════════════════
    # TAB 2: MARKET ANALYTICS (ONLY 1 GRAPH KEPT)
    # ═════════════════════════════════════════════════════════════════
    with tab_chart:
        f_col1, f_col2, f_col3 = st.columns([1, 1.4, 1])
        with f_col1:
            selected_asset = st.selectbox(
                "Underlying Ticker",
                options=["SPY", "QQQ", "IWM", "GLD", "SLV", "PLTR", "SOFI", "NVDA", "TSLA", "VIXY"],
                index=0,
            )
            timeframe_choice = st.selectbox(
                "Bar Timeframe",
                options=["1Day", "1Hour", "15Min", "5Min"],
                index=0,
            )
        with f_col2:
            chart_style = st.selectbox(
                "Rendering Style",
                options=["Candlestick", "Translucent Line Area", "OHLC Bars"],
                index=0,
            )
            indicator_selection = st.multiselect(
                "Overlay Indicators",
                options=["20 EMA", "50 EMA", "Bollinger Bands", "Volume Overlay", "RSI (14)"],
                default=["20 EMA", "50 EMA", "Volume Overlay"],
            )
        with f_col3:
            history_bars = st.slider("Lookback Bars", min_value=30, max_value=250, value=90, step=10)

        try:
            bars = client.get_bars(selected_asset, timeframe=timeframe_choice, limit=history_bars)
            if bars and len(bars) >= 5:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"])
                df = df.sort_values("t").reset_index(drop=True)

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
                    specs=specs,
                )

                if chart_style == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                        name=selected_asset,
                        increasing=dict(line=dict(color="#10b981", width=1), fillcolor="#10b981"),
                        decreasing=dict(line=dict(color="#f43f5e", width=1), fillcolor="#f43f5e"),
                    ), row=1, col=1, secondary_y=False)
                elif chart_style == "Translucent Line Area":
                    fig.add_trace(go.Scatter(
                        x=df["t"], y=df["c"], mode="lines",
                        name="Close Price", line=dict(color="#38bdf8", width=2),
                        fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.08)"
                    ), row=1, col=1, secondary_y=False)
                else:
                    fig.add_trace(go.Ohlc(
                        x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                        name=selected_asset,
                        increasing=dict(line=dict(color="#10b981")),
                        decreasing=dict(line=dict(color="#f43f5e"))
                    ), row=1, col=1, secondary_y=False)

                if "20 EMA" in indicator_selection:
                    fig.add_trace(go.Scatter(x=df["t"], y=df["ema20"], mode="lines", name="20 EMA", line=dict(color="#38bdf8", width=1.5)), row=1, col=1, secondary_y=False)
                if "50 EMA" in indicator_selection:
                    fig.add_trace(go.Scatter(x=df["t"], y=df["ema50"], mode="lines", name="50 EMA", line=dict(color="#f59e0b", width=1.5)), row=1, col=1, secondary_y=False)
                if "Bollinger Bands" in indicator_selection:
                    fig.add_trace(go.Scatter(x=df["t"], y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)), row=1, col=1, secondary_y=False)
                    fig.add_trace(go.Scatter(x=df["t"], y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.4)", width=1), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.04)"), row=1, col=1, secondary_y=False)

                if has_volume:
                    v_colors = ["#10b981" if c >= o else "#f43f5e" for c, o in zip(df["c"], df["o"])]
                    fig.add_trace(go.Bar(x=df["t"], y=df["v"], name="Volume", marker=dict(color=v_colors, opacity=0.3), showlegend=False), row=1, col=1, secondary_y=True)

                if has_rsi:
                    fig.add_trace(go.Scatter(x=df["t"], y=df["rsi"], mode="lines", name="RSI (14)", line=dict(color="#a855f7", width=1.5)), row=2, col=1)
                    fig.add_hline(y=70, line=dict(color="#f43f5e", width=1, dash="dash"), row=2, col=1)
                    fig.add_hline(y=30, line=dict(color="#10b981", width=1, dash="dash"), row=2, col=1)

                fig.update_layout(
                    height=480,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(color="#94a3b8", size=11)),
                    xaxis=dict(gridcolor="#1e293b", rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor="#1e293b"),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info(f"Retrieving market feed for {selected_asset}...")
        except Exception as e:
            st.warning(f"Unable to load chart stream for {selected_asset}: {e}")

    # ═════════════════════════════════════════════════════════════════
    # TAB 3: SIZING, JOURNAL & SYSTEM
    # ═════════════════════════════════════════════════════════════════
    with tab_execution:
        stats = journal.get_all_strategy_stats()
        k1, k2, k3 = st.columns(3)
        
        for idx, (s_name, col) in enumerate(zip(["theta", "momo", "iv_crush"], [k1, k2, k3])):
            s_stat = stats.get(s_name, {})
            n_trades = s_stat.get("n_trades", 0)
            win_rate = s_stat.get("win_rate", 0.0)
            dollar_size = kelly.get_position_size(s_name, equity)
            pct = (dollar_size / equity) * 100 if equity > 0 else 0.0

            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{s_name.upper()} STRATEGY EDGE</div>
                    <div class="kpi-value">${dollar_size:,.0f} <span style="font-size: 13px; color: #a855f7; font-weight: 600;">({pct:.1f}% equity)</span></div>
                    <div class="kpi-sub">Win Rate: {win_rate:.1%} ({n_trades} closed trades)</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        hub_view = st.radio(
            "Hub View Switcher",
            ["Symbol Contract Sizing Matrix", "SQLite Trade Journal (Audit Trail)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if hub_view == "Symbol Contract Sizing Matrix":
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
                        "Est Premium": f"${est_prem:.0f} / contract",
                        "Strategy Budget": f"${budget:,.0f}",
                        "Kelly Contracts": f"{contracts} contracts",
                        "Max Capital Risk": f"${min(budget, contracts * est_prem):,.0f}",
                        "Order Routing": "Smart Mid-Price Limit",
                    })
                except Exception:
                    pass

            if matrix_rows:
                st.dataframe(pd.DataFrame(matrix_rows), width="stretch", hide_index=True)
        else:
            try:
                with journal._connect() as conn:
                    trades_df = pd.read_sql_query(
                        "SELECT id, opened_at, agent, strategy, symbol, contract, side, qty, entry_price, exit_price, pnl, exit_reason FROM trades ORDER BY id DESC LIMIT 50",
                        conn
                    )
                if not trades_df.empty:
                    st.dataframe(trades_df, width="stretch", hide_index=True)
                else:
                    st.info("No trades executed yet. Completed orders will appear in this persistent audit journal.")
            except Exception as e:
                st.info(f"Trade journal is ready. Recorded trades will display here ({e}).")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>Autonomous Engine & Auto-Pilot Feed</div>", unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #10b981;">
                <div class="kpi-label">Auto-Pilot Scheduler</div>
                <div class="kpi-value" style="color: #10b981; font-size: 18px;">ACTIVE</div>
                <div class="kpi-sub">Mon-Fri 13:30-21:00 UTC</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #38bdf8;">
                <div class="kpi-label">Last Session Run</div>
                <div class="kpi-value" style="font-size: 15px;">{bg_state['last_run'][:19]}</div>
                <div class="kpi-sub">Regime: {bg_state['last_regime']}</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            hybrid = getattr(config, "HYBRID", None)
            interval_mins = getattr(hybrid, "session_interval_minutes", 10) if hybrid else 10
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #a855f7;">
                <div class="kpi-label">Completed Sessions</div>
                <div class="kpi-value" style="font-size: 18px;">{bg_state['runs_count']}</div>
                <div class="kpi-sub">Interval: Every {interval_mins}m</div>
            </div>
            """, unsafe_allow_html=True)
        with e4:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 3px solid #f59e0b;">
                <div class="kpi-label">Keep-Alive Heartbeat</div>
                <div class="kpi-value" style="color: #f59e0b; font-size: 18px;">ONLINE</div>
                <div class="kpi-sub">Ping interval 4.5m</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if bg_logs:
            for item in list(bg_logs)[:10]:
                pnl_c = "#10b981" if item.get("pnl", 0) >= 0 else "#ef4444"
                pnl_txt = f"+${item.get('pnl', 0):,.2f}" if item.get("pnl", 0) >= 0 else f"-${abs(item.get('pnl', 0)):,.2f}"
                badge_bg = "rgba(56, 189, 248, 0.15)" if item['type'] == "AUTO_PILOT" else "rgba(168, 85, 247, 0.15)"
                badge_c = "#38bdf8" if item['type'] == "AUTO_PILOT" else "#a855f7"
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 6px; padding: 10px 14px; margin-bottom: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 10px; font-weight: 800; background: {badge_bg}; color: {badge_c}; padding: 2px 7px; border-radius: 4px;">{item['type']}</span>
                            <span style="font-weight: 700; font-size: 12px; color: #e2e8f0;">Trading Session</span>
                        </div>
                        <span style="font-size: 11px; color: #64748b;">{item['time']}</span>
                    </div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                        Market Regime: <b style="color: #f8fafc;">{item['regime']}</b> &bull; Actions: <b style="color: #f8fafc;">{item['actions']}</b> &bull; Session P&L: <b style="color: {pnl_c};">{pnl_txt}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No trading sessions logged yet this runtime. Trigger a session manually via the sidebar or await market open.")

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption("Cache Me If You Can | Autonomous AI Options Trading System | Alpaca Hackathon 2026")

# ── Auto-refresh Trigger (30s) ─────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
