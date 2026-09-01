"""
core/opportunity_scorer.py — Dynamic Opportunity Scorer for Hybrid Council.

Computes a greedy multiplier in the range [1.0, 2.0] by stacking independent
market criteria onto a baseline of 1.0. High-conviction, favorable market
conditions dynamically scale position sizes (e.g. up to Half-Kelly) while
mediocre or uncertain setups remain at baseline sizing.

Criteria Stack (Base = 1.0, Cap = 2.0):
  +0.20  IVR > 50 (elevated implied volatility rank)
  +0.20  EMA trend aligns with strategy direction (bullish for momo/theta, any for iv_crush)
  +0.15  VIX in sweet spot [15.0, 22.0] (ideal liquidity & option pricing)
  +0.15  Volume surge ratio > 1.8 (strong institutional participation)
  +0.15  Symbol not in open positions (portfolio diversification)
  +0.15  Session P&L > 0 (scaling into winning sessions)

Usage:
    scorer = OpportunityScorer()
    multiplier = scorer.score(
        market_context={
            "symbol": "NVDA",
            "strategy": "momo",
            "ivr": 65.0,
            "vix": 18.5,
            "ema_signal": "bullish",
            "volume_surge_ratio": 2.1,
        },
        open_positions=["SPY", "QQQ"],
        session_pnl=350.0,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rich.console import Console

import config

console = Console()
log = logging.getLogger(__name__)


class OpportunityScorer:
    """
    Evaluates market conditions to compute an opportunity scale factor.

    Stacks additive multipliers when specific market signals are met,
    capped at config.HYBRID.max_greedy_multiplier (default 2.0).
    """

    def __init__(
        self,
        greedy_enabled: Optional[bool] = None,
        max_greedy_multiplier: Optional[float] = None,
    ) -> None:
        """
        Initialise OpportunityScorer with optional config overrides.

        Args:
            greedy_enabled: Enable/disable greedy scaling. Defaults to config.HYBRID.greedy_enabled.
            max_greedy_multiplier: Upper cap on the computed multiplier. Defaults to config.HYBRID.max_greedy_multiplier.
        """
        hybrid = getattr(config, "HYBRID", None)
        self.greedy_enabled: bool = (
            getattr(hybrid, "greedy_enabled", True)
            if greedy_enabled is None
            else greedy_enabled
        )
        self.max_greedy_multiplier: float = (
            getattr(hybrid, "max_greedy_multiplier", 2.0)
            if max_greedy_multiplier is None
            else max_greedy_multiplier
        )

    def score(
        self,
        market_context: dict[str, Any],
        open_positions: Optional[list[str]] = None,
        session_pnl: float = 0.0,
        verbose: bool = False,
    ) -> float:
        """
        Compute greedy multiplier [1.0, max_greedy_multiplier] for position sizing.

        Args:
            market_context: Dictionary with keys such as:
                            - 'symbol' or 'ticker' (str)
                            - 'ivr' (float 0-100)
                            - 'vix' (float)
                            - 'ema_signal' (str: 'bullish'/'bearish'/'neutral')
                            - 'ema_crossover' (bool)
                            - 'volume_surge_ratio' (float)
                            - 'strategy' (str: 'theta'/'momo'/'iv_crush')
            open_positions: List of ticker symbols currently in the portfolio.
            session_pnl: Realised/unrealised session P&L in USD.
            verbose: If True, prints formatted multi-line criteria tree to console.

        Returns:
            Float multiplier in [1.0, max_greedy_multiplier].
            Returns 1.0 if greedy mode is disabled.
        """
        if not self.greedy_enabled:
            log.info("[OpportunityScorer] Greedy mode disabled — returning 1.0")
            return 1.0

        if open_positions is None:
            open_positions = []

        # ── 1. Extract Context Fields ─────────────────────────────────
        symbol = str(
            market_context.get("symbol")
            or market_context.get("ticker")
            or ""
        ).strip().upper()

        strategy = str(
            market_context.get("strategy") or ""
        ).strip().lower()

        # Volatility & IVR
        vol_dict = market_context.get("volatility")
        ivr_val = (
            market_context.get("ivr")
            or (vol_dict.get("ivr") if isinstance(vol_dict, dict) else None)
            or 0.0
        )
        try:
            ivr = float(ivr_val)
        except (ValueError, TypeError):
            ivr = 0.0

        # VIX
        mkt_cond = market_context.get("market_conditions")
        vix_val = (
            market_context.get("vix")
            or (mkt_cond.get("vix") if isinstance(mkt_cond, dict) else None)
            or (vol_dict.get("vix") if isinstance(vol_dict, dict) else None)
            or 0.0
        )
        try:
            vix = float(vix_val)
        except (ValueError, TypeError):
            vix = 0.0

        # Technical signals
        tech_dict = market_context.get("technical_signals")
        tech = tech_dict if isinstance(tech_dict, dict) else {}

        ema_signal = str(
            market_context.get("ema_signal")
            or tech.get("ema_signal")
            or ""
        ).strip().lower()

        ema_crossover = bool(
            market_context.get("ema_crossover", tech.get("ema_crossover", False))
        )

        vol_surge_val = (
            market_context.get("volume_surge_ratio")
            or tech.get("volume_surge_ratio")
        )
        try:
            volume_surge_ratio = float(vol_surge_val) if vol_surge_val is not None else 1.0
        except (ValueError, TypeError):
            volume_surge_ratio = 1.0

        # ── 2. Evaluate Criteria ──────────────────────────────────────
        criteria_results: list[dict[str, Any]] = []

        # Criterion 1: IVR > 50 (+0.20)
        ivr_fired = ivr > 50.0
        criteria_results.append({
            "name": "IVR > 50",
            "fired": ivr_fired,
            "boost": 0.20 if ivr_fired else 0.0,
            "detail": f"ivr={ivr:.1f}",
        })

        # Criterion 2: EMA trend aligns with strategy direction (+0.20)
        # Bullish for momo/theta, any for iv_crush (delta-neutral)
        is_iv_crush = any(k in strategy for k in ("iv_crush", "crush", "straddle", "earnings"))
        is_bullish = ema_signal in ("bullish", "buy") or "bull" in ema_signal or ema_crossover

        if is_iv_crush:
            ema_fired = True
            ema_detail = f"strategy={strategy or 'iv_crush'} (neutral/any trend)"
        else:
            ema_fired = is_bullish
            strat_label = strategy or "momo/theta"
            ema_detail = f"strategy={strat_label}, signal={ema_signal or 'neutral'}, crossover={ema_crossover}"

        criteria_results.append({
            "name": "EMA Trend Aligned",
            "fired": ema_fired,
            "boost": 0.20 if ema_fired else 0.0,
            "detail": ema_detail,
        })

        # Criterion 3: VIX in 15-22 sweet spot (+0.15)
        vix_fired = 15.0 <= vix <= 22.0
        criteria_results.append({
            "name": "VIX Sweet Spot [15-22]",
            "fired": vix_fired,
            "boost": 0.15 if vix_fired else 0.0,
            "detail": f"vix={vix:.1f}",
        })

        # Criterion 4: Volume surge ratio > 1.8 (+0.15)
        vol_fired = volume_surge_ratio > 1.8
        criteria_results.append({
            "name": "Volume Surge > 1.8",
            "fired": vol_fired,
            "boost": 0.15 if vol_fired else 0.0,
            "detail": f"surge_ratio={volume_surge_ratio:.2f}",
        })

        # Criterion 5: Symbol not in open_positions list (+0.15)
        def _extract_underlying(pos_sym: Any) -> str:
            raw = str(pos_sym).strip().upper()
            for i, ch in enumerate(raw):
                if ch.isdigit():
                    return raw[:i]
            return raw

        open_set = {_extract_underlying(p) for p in open_positions if p}
        target_sym = _extract_underlying(symbol) if symbol else ""
        if target_sym:
            not_in_open = target_sym not in open_set
            sym_detail = f"{target_sym} not in {sorted(list(open_set)) if open_set else '[]'}"
        else:
            not_in_open = len(open_set) == 0
            sym_detail = "no open positions" if not_in_open else f"open: {sorted(list(open_set))}"

        criteria_results.append({
            "name": "Fresh Symbol (Diversification)",
            "fired": not_in_open,
            "boost": 0.15 if not_in_open else 0.0,
            "detail": sym_detail,
        })

        # Criterion 6: Session P&L > 0 (+0.15)
        pnl_fired = session_pnl > 0.0
        criteria_results.append({
            "name": "Session P&L > 0 (Winning)",
            "fired": pnl_fired,
            "boost": 0.15 if pnl_fired else 0.0,
            "detail": f"session_pnl=${session_pnl:,.2f}",
        })

        # ── 3. Stacking & Capping ─────────────────────────────────────
        base_multiplier = 1.00
        total_boost = sum(c["boost"] for c in criteria_results)
        raw_multiplier = base_multiplier + total_boost
        capped_multiplier = min(raw_multiplier, self.max_greedy_multiplier)
        final_multiplier = round(capped_multiplier, 2)

        # ── 4. Logging & Rich Console Display ─────────────────────────
        fired_count = sum(1 for c in criteria_results if c["fired"])
        log.info(
            f"[OpportunityScorer] {symbol or 'TRADE'} greedy_multiplier={final_multiplier:.2f} "
            f"(base=1.00, boost=+{total_boost:.2f}, capped_at={self.max_greedy_multiplier:.2f}, "
            f"fired={fired_count}/{len(criteria_results)} criteria)"
        )

        sym_tag = f" [{symbol}]" if symbol else ""
        if verbose or log.isEnabledFor(logging.DEBUG):
            console.print(f"[bold cyan]┌─ Opportunity Scorer{sym_tag} ─────────────────────────────[/bold cyan]")
            for c in criteria_results:
                if c["fired"]:
                    console.print(
                        f"[bold cyan]│[/bold cyan]  [bold green]✓[/bold green] [white]{c['name']:<30}[/white] "
                        f"[green]+{c['boost']:.2f}[/green]  [dim]({c['detail']})[/dim]"
                    )
                else:
                    console.print(
                        f"[bold cyan]│[/bold cyan]  [dim red]✗[/dim red] [dim]{c['name']:<30}[/dim] "
                        f"[dim]+0.00  ({c['detail']})[/dim]"
                    )

            status_color = "bold green" if final_multiplier > 1.0 else "bold yellow"
            console.print(
                f"[bold cyan]└─ Multiplier:[/bold cyan] [{status_color}]{final_multiplier:.2f}x[/{status_color}] "
                f"[dim](Base: 1.00 + Boost: {total_boost:.2f} → Cap: {self.max_greedy_multiplier:.2f}x)[/dim]"
            )

        return final_multiplier
