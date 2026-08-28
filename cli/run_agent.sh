#!/usr/bin/env bash
# cli/run_agent.sh — Alpaca CLI-based agent runner
#
# Uses Alpaca CLI for:
#   - Account checks
#   - Position monitoring
#   - Pre/post session reporting
#   - Cron-compatible structured JSON output
#
# Alpaca CLI docs: https://docs.alpaca.markets/us/docs/alpacas-cli
#
# Usage:
#   bash cli/run_agent.sh run       # run trading session
#   bash cli/run_agent.sh status    # show account + positions
#   bash cli/run_agent.sh pnl       # show daily P&L
#
# Cron (runs daily at 9:30 AM ET = 3:00 PM IST):
#   30 9 * * 1-5 cd /path/to/project && bash cli/run_agent.sh run >> logs/cron.log 2>&1

set -euo pipefail

MODE="${1:-status}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════"
echo "  Cache Me Agent CLI — $TIMESTAMP"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════"

# ── Check Alpaca CLI is installed ─────────────────────────────
if ! command -v alpaca &> /dev/null; then
    echo "ERROR: Alpaca CLI not found. Install: https://github.com/alpacahq/cli"
    echo "  pip install alpaca-cli"
    exit 1
fi

# ── Account Status ─────────────────────────────────────────────
check_account() {
    echo ""
    echo "📊 Account Status:"
    alpaca account get --output json | tee "$LOG_DIR/account_${TIMESTAMP}.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'  Equity:       \${float(data[\"equity\"]):,.2f}')
print(f'  Cash:         \${float(data[\"cash\"]):,.2f}')
print(f'  Buying Power: \${float(data[\"buying_power\"]):,.2f}')
"
}

# ── Open Positions ─────────────────────────────────────────────
check_positions() {
    echo ""
    echo "📈 Open Positions:"
    alpaca positions list --output json 2>/dev/null | tee "$LOG_DIR/positions_${TIMESTAMP}.json" | python3 -c "
import json, sys
positions = json.load(sys.stdin)
if not positions:
    print('  No open positions.')
else:
    print(f'  {'Symbol':<20} {'Qty':>8} {'Market Value':>15} {'Unrealized P&L':>15}')
    print(f'  {'-'*60}')
    for p in positions:
        pl = float(p.get('unrealized_pl', 0))
        sign = '+' if pl >= 0 else ''
        print(f'  {p[\"symbol\"]:<20} {float(p[\"qty\"]):>8.1f} \${float(p[\"market_value\"]):>14,.2f} {sign}\${pl:>13,.2f}')
" 2>/dev/null || echo "  (No positions or CLI error)"
}

# ── Daily P&L ─────────────────────────────────────────────────
check_pnl() {
    echo ""
    echo "💰 Daily P&L Summary:"
    alpaca account get --output json | python3 -c "
import json, sys
data = json.load(sys.stdin)
equity = float(data['equity'])
start = 100000  # hackathon starting balance
pnl = equity - start
pct = (pnl / start) * 100
sign = '+' if pnl >= 0 else ''
print(f'  Starting Balance: \$100,000.00')
print(f'  Current Equity:   \${equity:,.2f}')
print(f'  Total P&L:        {sign}\${pnl:,.2f} ({sign}{pct:.2f}%)')
"
}

# ── Run Trading Session ────────────────────────────────────────
run_session() {
    echo ""
    echo "🚀 Starting trading session..."
    check_account
    check_positions
    echo ""
    echo "🤖 Launching Python orchestrator..."
    python3 main.py run 2>&1 | tee "$LOG_DIR/session_${TIMESTAMP}.log"
    echo ""
    echo "✅ Session complete."
    check_pnl
    check_positions
}

# ── Mode Dispatch ──────────────────────────────────────────────
case "$MODE" in
    run)
        run_session
        ;;
    status)
        check_account
        check_positions
        check_pnl
        ;;
    pnl)
        check_pnl
        ;;
    positions)
        check_positions
        ;;
    *)
        echo "Usage: $0 [run|status|pnl|positions]"
        exit 1
        ;;
esac

echo ""
echo "Done. Logs saved to $LOG_DIR/"
