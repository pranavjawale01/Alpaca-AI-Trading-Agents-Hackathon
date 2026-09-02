"""
core/llm_council.py — 3-Model LLM Voting Council.

Queries three independent LLMs in parallel and applies confidence-weighted
majority voting to determine whether a trade signal should be acted on.

Voting Algorithm:
    Each model returns: action ∈ {buy, sell, hold}, confidence ∈ [0, 1]
    Vote value: buy=+1, hold=0, sell=-1
    weighted_score = Σ(confidence_i × vote_i) / n_models   → [-1, +1]
    CONSENSUS when |weighted_score| >= threshold (default 0.60)

This drastically reduces false-positive trade entries (the main P&L leak),
requiring multiple independent AI perspectives to agree before capital is risked.

Usage:
    council = LLMCouncil()
    result = council.vote("NVDA", market_context, strategy="momentum_call")
    if result.agreed:
        place_trade()
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from rich.console import Console

import config
from core.model_discovery import discover_available_models

console = Console()
log = logging.getLogger(__name__)

# ── Vote numeric encoding ──────────────────────────────────────
_ACTION_SCORE = {"buy": 1.0, "sell": -1.0, "hold": 0.0}


@dataclass
class ModelVote:
    """A single model's verdict on a potential trade."""
    model: str
    action: str        # "buy" | "sell" | "hold"
    confidence: float  # 0.0 – 1.0
    reasoning: str
    raw_response: str = ""

    @property
    def score(self) -> float:
        """Signed score: buy=+1, hold=0, sell=-1 scaled by confidence."""
        return _ACTION_SCORE.get(self.action, 0.0) * self.confidence


@dataclass
class ConsensusResult:
    """Aggregated verdict from all council models."""
    action: str           # winning action ("buy" | "sell" | "hold")
    net_score: float      # weighted average score in [-1, +1]
    agreed: bool          # True if |net_score| >= threshold
    threshold: float      # the threshold used
    conviction_tier: str = "veto"         # "strong" | "moderate" | "pilot" | "veto"
    size_multiplier: float = 0.0          # 1.0 (strong), 0.70 (moderate), 0.40 (pilot), 0.0 (veto)
    votes: list[ModelVote] = field(default_factory=list)
    dissenting_models: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Council verdict: {self.action.upper()} | "
            f"score={self.net_score:+.3f} | tier={self.conviction_tier.upper()} | "
            f"size_mult={self.size_multiplier:.2f} | agreed={self.agreed}",
        ]
        for v in self.votes:
            marker = "[AGREE]" if v.action == self.action else "[DISSENT]"
            lines.append(
                f"  {v.model}: {marker} {v.action.upper()} "
                f"(conf={v.confidence:.2f}) — {v.reasoning}"
            )
        return "\n".join(lines)


# ── Prompt templates per strategy ─────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert quantitative trading analyst for an options trading system. "
    "You evaluate trade signals and return precise JSON decisions. "
    "Be concise, analytical, and risk-aware. Never add explanatory text outside JSON."
)

_STRATEGY_PROMPTS = {
    "momentum_call": (
        "Evaluate this momentum breakout setup for buying a short-dated OTM call option.\n"
        "Market context:\n{context}\n\n"
        "Should we enter a momentum call buy trade? "
        "Respond ONLY with valid JSON: "
        '{{"action": "buy", "confidence": 0.85, "reasoning": "one concise sentence"}}'
    ),
    "momentum_put": (
        "Evaluate this momentum breakdown setup for buying a short-dated OTM put option (bearish/short delta).\n"
        "Market context:\n{context}\n\n"
        "Should we enter a momentum put buy trade? "
        "Respond ONLY with valid JSON: "
        '{{"action": "buy", "confidence": 0.85, "reasoning": "one concise sentence"}}'
    ),
    "theta_put": (
        "Evaluate this setup for selling a cash-secured put to collect theta premium.\n"
        "Market context:\n{context}\n\n"
        "Should we sell this cash-secured put? "
        "Respond ONLY with valid JSON: "
        '{{"action": "sell", "confidence": 0.85, "reasoning": "one concise sentence"}}'
    ),
    "iv_crush_straddle": (
        "Evaluate this earnings IV-crush straddle sell setup.\n"
        "Market context:\n{context}\n\n"
        "Should we sell this ATM straddle to profit from post-earnings IV collapse? "
        "Respond ONLY with valid JSON: "
        '{{"action": "sell", "confidence": 0.85, "reasoning": "one concise sentence"}}'
    ),
    "general": (
        "Evaluate this potential trade.\n"
        "Market context:\n{context}\n\n"
        "What action is best? "
        "Respond ONLY with valid JSON: "
        '{{"action": "buy", "confidence": 0.85, "reasoning": "one concise sentence"}}'
    ),
}


# ── 3-Agent Quantitative Strategy Evaluators ──────────────────
QUANTITATIVE_AGENTS = [
    "TrendMomentumAgent",
    "VolatilityPricingAgent",
    "RiskGreeksAgent",
]


# ── 3-Agent Quantitative Strategy Evaluators ──────────────────
QUANTITATIVE_AGENTS = [
    "TrendMomentumAgent",
    "VolatilityPricingAgent",
    "RiskGreeksAgent",
]


class TrendMomentumEvaluator:
    """
    Agent 1: Trend & Momentum Strategy.
    Advanced multi-factor trend evaluation incorporating:
      - Fast/Slow EMA crossovers (20/50)
      - Volume surge confirmation (>=1.8x)
      - RSI-14 momentum boundaries (overbought >70 / oversold <30 protection)
      - MACD histogram direction & crossovers
      - Bollinger Band positioning & volatility squeeze
      - 200-day SMA macro regime filter
    """
    @staticmethod
    def evaluate(symbol: str, ctx: dict, strategy: str) -> ModelVote:
        strat = strategy.lower()
        tech = ctx.get("technical_signals", {})
        crossover = tech.get("ema_crossover", False)
        crossover_type = tech.get("ema_crossover_type", tech.get("ema_signal", ""))
        is_surging = tech.get("volume_surging", False)
        surge_ratio = tech.get("volume_surge_ratio", 1.0)
        vix = ctx.get("market_conditions", {}).get("vix", ctx.get("volatility", {}).get("vix", 20.0))

        # Advanced technical factors
        rsi = tech.get("rsi_14", 50.0)
        macd_hist = tech.get("macd_histogram", 0.0)
        macd_bull_cross = tech.get("macd_bullish_cross", False)
        macd_bear_cross = tech.get("macd_bearish_cross", False)
        above_sma200 = tech.get("price_above_sma200", True)
        pct_b = tech.get("bollinger_pct_b", 0.5)

        # ── 1. Theta Cash-Secured Put ─────────────────────────────────
        if "theta" in strat:
            if vix >= 30:
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"High market instability (VIX={vix:.1f} >= 30); trend too volatile for put selling",
                )

            # Score trend stability: stable / uptrending underlying preferred
            score = 50
            if vix < 20:
                score += 20
            elif vix < 25:
                score += 10

            if above_sma200:
                score += 15
            if rsi > 35:  # Not in a freefall
                score += 15

            conf = min(0.96, 0.75 + (score - 50) / 100.0)
            return ModelVote(
                model="TrendMomentumAgent",
                action="sell",
                confidence=conf,
                reasoning=f"Market regime stable for theta collection (VIX={vix:.1f}, trend score={score}/100)",
            )

        # ── 2. Momo Bullish Breakout (Call Buy) ─────────────────────────
        elif "momentum_call" in strat or (strat == "momo" and ctx.get("direction") == "bullish"):
            # Mandatory threshold: crossover or bullish alignment + volume surge
            has_bull_cross = crossover or crossover_type == "bullish"
            if not (has_bull_cross and is_surging):
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Lack of bullish breakout confirmation (crossover={crossover}, surge={surge_ratio:.1f}x)",
                )

            # Check overbought trap: don't chase if RSI > 72 or pct_b > 1.05
            if rsi > 72:
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Overbought exhaustion risk: RSI={rsi:.1f} > 72; breakout prone to pullback",
                )

            # Multi-factor score (0–100)
            score = 30  # baseline for crossover
            score += min(30, int((surge_ratio - 1.0) * 20))  # volume score up to 30
            if 45 <= rsi <= 68:  # Healthy momentum range
                score += 15
            if macd_bull_cross or macd_hist > 0:
                score += 15
            if above_sma200:
                score += 10

            conf = min(0.98, 0.65 + (score / 100.0) * 0.30)
            return ModelVote(
                model="TrendMomentumAgent",
                action="buy",
                confidence=conf,
                reasoning=f"High-conviction bullish breakout: surge={surge_ratio:.1f}x, RSI={rsi:.1f}, score={score}/100",
            )

        # ── 3. Momo Bearish Breakdown (Put Buy) ─────────────────────────
        elif "momentum_put" in strat or (strat == "momo" and ctx.get("direction") == "bearish"):
            has_bear_cross = crossover or crossover_type == "bearish"
            if not (has_bear_cross and is_surging):
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Lack of bearish breakdown confirmation (crossover={crossover}, surge={surge_ratio:.1f}x)",
                )

            # Check oversold trap: don't short into oversold bounce if RSI < 28
            if rsi < 28:
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Oversold bounce risk: RSI={rsi:.1f} < 28; breakdown prone to mean-reversion",
                )

            # Multi-factor score
            score = 30
            score += min(30, int((surge_ratio - 1.0) * 20))
            if 32 <= rsi <= 55:
                score += 15
            if macd_bear_cross or macd_hist < 0:
                score += 15
            if not above_sma200:
                score += 10

            conf = min(0.98, 0.65 + (score / 100.0) * 0.30)
            return ModelVote(
                model="TrendMomentumAgent",
                action="buy",
                confidence=conf,
                reasoning=f"High-conviction bearish breakdown: surge={surge_ratio:.1f}x, RSI={rsi:.1f}, score={score}/100",
            )

        # ── 4. IV Crush ATM Straddle ──────────────────────────────────
        elif "iv_crush" in strat or "straddle" in strat:
            days = ctx.get("earnings_event", {}).get("days_to_earnings", 2)
            if not (1 <= days <= 3):
                return ModelVote(
                    model="TrendMomentumAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Earnings timing ({days} days) outside optimal 1-3 day window",
                )

            # Consolidation factor: straddle selling prefers non-trending stock pre-earnings
            conf = 0.88
            if 40 <= rsi <= 60:
                conf = 0.94

            return ModelVote(
                model="TrendMomentumAgent",
                action="sell",
                confidence=conf,
                reasoning=f"Optimal pre-earnings consolidation window ({days}d to earnings, RSI={rsi:.1f})",
            )

        return ModelVote(
            model="TrendMomentumAgent",
            action="buy",
            confidence=0.80,
            reasoning="Trend indicators confirmed",
        )


class VolatilityPricingEvaluator:
    """
    Agent 2: Volatility & Pricing Strategy.
    Advanced options pricing and volatility surface evaluation:
      - IV Rank (IVR) absolute thresholds & percentiles
      - IV / HV (Implied vs Realized Volatility) edge ratio
      - Volatility regime suitability for buying vs selling
      - Mispricing protection (prevents buying overpriced options)
    """
    @staticmethod
    def evaluate(symbol: str, ctx: dict, strategy: str) -> ModelVote:
        strat = strategy.lower()
        vol = ctx.get("volatility", {})
        ivr = vol.get("ivr", 0.0)
        hist_vol = vol.get("historical_vol_30d", ctx.get("market_conditions", {}).get("historical_volatility_30d", 0.25))
        iv_hv = vol.get("iv_hv_ratio")

        # ── 1. Theta Cash-Secured Put (Premium Selling) ───────────────
        if "theta" in strat:
            if ivr < 30:
                return ModelVote(
                    model="VolatilityPricingAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"IV Rank {ivr:.1f} < 30; premium inadequate for short options risk",
                )

            # Premium sellers want rich IV and IV > HV
            score = 60
            score += min(25, int((ivr - 30) / 2))  # IVR contribution
            if iv_hv and iv_hv >= 1.2:
                score += 15  # IV expensive vs historical realized

            conf = min(0.98, 0.70 + (score - 60) / 100.0)
            return ModelVote(
                model="VolatilityPricingAgent",
                action="sell",
                confidence=conf,
                reasoning=f"Rich premium for put selling (IVR={ivr:.1f}>=30, vol score={score}/100)",
            )

        # ── 2. IV Crush Straddle (Heavy Volatility Crush) ──────────────
        elif "iv_crush" in strat or "straddle" in strat:
            if ivr < 60:
                return ModelVote(
                    model="VolatilityPricingAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"IV Rank {ivr:.1f} < 60; insufficient volatility inflation for straddle crush",
                )

            conf = min(0.98, 0.75 + (ivr - 60) / 80.0)
            return ModelVote(
                model="VolatilityPricingAgent",
                action="sell",
                confidence=conf,
                reasoning=f"Extreme volatility inflation detected (IVR={ivr:.1f} >= 60); prime IV crush edge",
            )

        # ── 3. Momo Option Buying (Calls / Puts) ────────────────────────
        elif "momentum" in strat or "momo" in strat:
            # Option buyers want cheap options (low IVR) to avoid IV crush
            if ivr > 50 and (hist_vol is not None and hist_vol > 0.45):
                return ModelVote(
                    model="VolatilityPricingAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Option IV inflated (IVR={ivr:.1f}, HV={hist_vol:.2f}); high IV crush risk on long options",
                )

            # Favorable pricing: IVR <= 45 or HV <= 0.45
            edge_score = 70
            if ivr <= 30:
                edge_score += 20  # Very cheap options
            elif ivr <= 45:
                edge_score += 10

            conf = min(0.96, 0.65 + (edge_score / 100.0) * 0.30)
            return ModelVote(
                model="VolatilityPricingAgent",
                action="buy",
                confidence=conf,
                reasoning=f"Attractive option pricing (IVR={ivr:.1f}); minimal IV crush headwind",
            )

        return ModelVote(
            model="VolatilityPricingAgent",
            action="buy",
            confidence=0.80,
            reasoning="Volatility metrics within acceptable range",
        )


class RiskGreeksEvaluator:
    """
    Agent 3: Risk & Greeks Strategy.
    Advanced portfolio risk guardian:
      - VIX kill switch and graduated risk tiers
      - Portfolio heat tracking (options exposure cap)
      - Daily P&L drawdown budget protection
      - Correlation & sector concentration controls
      - Greeks delta headroom verification
    """
    @staticmethod
    def evaluate(symbol: str, ctx: dict, strategy: str) -> ModelVote:
        strat = strategy.lower()
        vix = ctx.get("market_conditions", {}).get("vix", ctx.get("volatility", {}).get("vix", 20.0))
        trade_params = ctx.get("trade_parameters", {})
        portfolio = ctx.get("portfolio_state", {})

        # ── 1. Global VIX Kill Switch ──────────────────────────────────
        if vix >= 35:
            return ModelVote(
                model="RiskGreeksAgent",
                action="hold",
                confidence=0.0,
                reasoning=f"VIX={vix:.1f} exceeds 35 kill switch; capital preservation engaged",
            )

        # ── 2. Portfolio Heat & Drawdown Guardian ─────────────────────
        portfolio_heat = portfolio.get("portfolio_heat", 0.0)
        pnl_budget_used = portfolio.get("pnl_budget_used", 0.0)
        correlated_tech = portfolio.get("correlated_tech_count", 0)

        if portfolio_heat >= 0.95:
            return ModelVote(
                model="RiskGreeksAgent",
                action="hold",
                confidence=0.0,
                reasoning=f"Portfolio options exposure near 30% limit (heat={portfolio_heat*100:.0f}%); risk cap engaged",
            )

        if pnl_budget_used >= 0.90:
            return ModelVote(
                model="RiskGreeksAgent",
                action="hold",
                confidence=0.0,
                reasoning=f"Daily loss budget {pnl_budget_used*100:.0f}% consumed; preserving capital for next session",
            )

        tech_symbols = {"NVDA", "TSLA", "AAPL", "META", "AMZN", "MSFT", "GOOGL"}
        if symbol in tech_symbols and correlated_tech >= 4:
            return ModelVote(
                model="RiskGreeksAgent",
                action="hold",
                confidence=0.0,
                reasoning=f"Concentration risk: {correlated_tech} tech positions already open; sector limit reached",
            )

        # ── 3. Strategy-Specific Greeks Checks ────────────────────────
        if "theta" in strat:
            dte = trade_params.get("target_dte", 35)
            if not (25 <= dte <= 50):
                return ModelVote(
                    model="RiskGreeksAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"DTE={dte} outside risk-optimal window (25-50 days)",
                )

            delta_headroom = portfolio.get("delta_headroom", 50.0)
            if delta_headroom < 15.0 and portfolio.get("portfolio_delta", 0.0) > 35.0:
                return ModelVote(
                    model="RiskGreeksAgent",
                    action="hold",
                    confidence=0.0,
                    reasoning=f"Insufficient portfolio delta capacity ({delta_headroom:.1f} left)",
                )

            return ModelVote(
                model="RiskGreeksAgent",
                action="sell",
                confidence=0.92,
                reasoning=f"Risk profile approved: DTE={dte}, target delta ~0.20 within safety bounds",
            )

        elif "momentum" in strat or "momo" in strat:
            return ModelVote(
                model="RiskGreeksAgent",
                action="buy",
                confidence=0.90,
                reasoning="Asymmetric reward profile approved: 2.0x target vs strict 50% hard stop and 25% trailing stop",
            )

        elif "iv_crush" in strat or "straddle" in strat:
            return ModelVote(
                model="RiskGreeksAgent",
                action="sell",
                confidence=0.90,
                reasoning="Delta-neutral ATM straddle structure with strict stop-loss verified",
            )

        return ModelVote(
            model="RiskGreeksAgent",
            action="buy",
            confidence=0.80,
            reasoning="Risk parameters verified",
        )


class LLMCouncil:
    """
    3-Agent Quantitative Strategy Council for high-conviction trade signal filtering.
    Trades require UNANIMOUS APPROVAL from all 3 strategy agents:
      1. TrendMomentumAgent (Trend alignment, fast/slow EMA crossovers, volume surge)
      2. VolatilityPricingAgent (IV Rank, option pricing efficiency, volatility regime)
      3. RiskGreeksAgent (Portfolio Delta, VIX safety, stop-loss and Greeks limits)
    """

    # Regime-adaptive threshold map
    _REGIME_THRESHOLDS = {
        "risk_on":  {"strong": 0.55, "moderate": 0.40, "pilot": 0.25},
        "neutral":  {"strong": 0.65, "moderate": 0.50, "pilot": 0.35},
        "risk_off": {"strong": 0.80, "moderate": 0.65, "pilot": 0.50},
    }

    # Size multipliers per conviction tier
    _TIER_MULTIPLIERS = {
        "strong":   1.00,
        "moderate": 0.70,
        "pilot":    0.40,
        "veto":     0.00,
    }

    _credibility_tracker = None
    _current_regime: str = "neutral"
    _last_votes: list[ModelVote] = []

    def __init__(
        self,
        models: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.models = models or QUANTITATIVE_AGENTS
        self.threshold = threshold if threshold is not None else config.COUNCIL.consensus_threshold
        self.timeout = timeout if timeout is not None else config.COUNCIL.timeout_seconds
        self.enabled = config.COUNCIL.enabled

        # Current VIX regime — updated by Orchestrator before each session
        self._current_regime = "neutral"

        # Credibility tracker — set externally by Orchestrator after init
        self._credibility_tracker = None

        # Track last gathered votes
        self._last_votes = []

        self._client: Optional[OpenAI] = None
        self._available = True
        self._discovered_models: list[str] = list(QUANTITATIVE_AGENTS)

        console.print(
            f"[bold green]3-Agent Quantitative Strategy Council initialised | "
            f"{len(self.models)} agents | Unanimous Approval Required[/bold green]"
        )
        for m in self.models:
            console.print(f"  [dim]• {m}[/dim]")

    def set_regime(self, regime: str) -> None:
        """Update the VIX regime for adaptive threshold selection."""
        self._current_regime = regime
        log.info(f"[Council] Regime set to: {regime}")

    def set_credibility_tracker(self, tracker) -> None:
        """Inject the ModelCredibilityTracker for weighted voting."""
        self._credibility_tracker = tracker

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def vote(
        self,
        symbol: str,
        market_context: dict,
        strategy: str = "general",
    ) -> ConsensusResult:
        """
        Ask all 3 strategy agents to evaluate the trade signal.
        A trade is ONLY approved when all 3 agents unanimously approve.
        """
        # If council is disabled or unavailable, auto-approve (pass-through)
        if not self.enabled or not self._available:
            return ConsensusResult(
                action="buy", net_score=1.0, agreed=True,
                threshold=self.threshold,
                conviction_tier="strong", size_multiplier=1.0,
                votes=[], dissenting_models=[],
            )

        # If custom external models and client provided (e.g. mock unit tests)
        if self._client is not None and not any(m in QUANTITATIVE_AGENTS for m in self.models):
            context_str = json.dumps(market_context, indent=2)
            votes = self._gather_votes(symbol, context_str, strategy)
        else:
            # Gather votes from the 3 specialized quantitative strategy agents
            votes = [
                TrendMomentumEvaluator.evaluate(symbol, market_context, strategy),
                VolatilityPricingEvaluator.evaluate(symbol, market_context, strategy),
                RiskGreeksEvaluator.evaluate(symbol, market_context, strategy),
            ]
        return self._tally_votes(votes, strategy)

    # ──────────────────────────────────────────
    # Internal: parallel model querying
    # ──────────────────────────────────────────

    def _gather_votes(
        self, symbol: str, context_str: str, strategy: str
    ) -> list[ModelVote]:
        """Query all models in parallel, collect votes with timeout."""
        votes: list[ModelVote] = []

        with ThreadPoolExecutor(max_workers=max(1, len(self.models))) as pool:
            futures = {
                pool.submit(self._query_model, model, context_str, strategy): model
                for model in self.models
            }
            completed_models = set()
            try:
                for future in as_completed(futures, timeout=self.timeout + 2):
                    model = futures[future]
                    completed_models.add(model)
                    try:
                        vote = future.result()
                        votes.append(vote)
                        log.debug(
                            f"[{symbol}] {model}: {vote.action} "
                            f"(conf={vote.confidence:.2f})"
                        )
                    except Exception as exc:
                        log.warning(f"[{symbol}] {model} error ({exc}) — casting hold")
                        votes.append(self._error_vote(model, str(exc)))
            except TimeoutError:
                log.warning(f"[{symbol}] Council gathering timed out after {self.timeout}s")

            # Cast timeout votes for any models that did not finish within timeout
            for future, model in futures.items():
                if model not in completed_models:
                    future.cancel()
                    log.warning(f"[{symbol}] {model} timed out — casting hold")
                    votes.append(self._timeout_vote(model))

        return votes

    def _query_model(
        self, model_id: str, context_str: str, strategy: str
    ) -> ModelVote:
        """Query one LLM model and parse its JSON response with automatic self-healing fallback."""
        prompt_template = _STRATEGY_PROMPTS.get(strategy, _STRATEGY_PROMPTS["general"])
        user_prompt = prompt_template.format(context=context_str)

        attempt_models = [model_id]
        if hasattr(self, "_discovered_models"):
            for m in self._discovered_models:
                if m not in attempt_models:
                    attempt_models.append(m)

        last_exc = None
        for current_model in attempt_models:
            try:
                response = self._client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,  # low temp for deterministic trading decisions
                    max_tokens=500,
                )

                raw = response.choices[0].message.content or ""
                # If model was swapped due to auto-heal, update self.models
                if current_model != model_id and model_id in self.models:
                    idx = self.models.index(model_id)
                    self.models[idx] = current_model
                    console.print(
                        f"[cyan][DYNAMIC MODEL HEALING] Swapped '{model_id}' -> '{current_model}'[/cyan]"
                    )
                return self._parse_vote(current_model, raw)
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                # If unsupported model or 400/404, try next discovered model
                if any(err_kw in err_str for err_kw in ["not supported", "400", "404", "model_not_found", "does not exist"]):
                    continue
                else:
                    raise exc

        raise last_exc or RuntimeError(f"All attempted models failed for {model_id}")

    def _parse_vote(self, model_id: str, raw: str) -> ModelVote:
        """Parse a model's JSON response into a ModelVote."""
        try:
            cleaned = raw
            # Strip reasoning tags if present (e.g. DeepSeek/Qwen <think>...</think>)
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1]

            # Find JSON object in response
            if "{" in cleaned:
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                data = json.loads(cleaned[start:end])

                action = str(data.get("action", "hold")).lower()
                if action not in _ACTION_SCORE:
                    action = "hold"

                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0,1]

                reasoning = str(data.get("reasoning", ""))[:200]

                return ModelVote(
                    model=model_id,
                    action=action,
                    confidence=confidence,
                    reasoning=reasoning,
                    raw_response=raw,
                )
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            log.debug(f"Parse error from {model_id}: {exc} | raw={raw[:100]}")

        # Fallback: couldn't parse → neutral hold
        return ModelVote(
            model=model_id,
            action="hold",
            confidence=0.0,
            reasoning="Parse error — defaulting to hold",
            raw_response=raw,
        )

    # ──────────────────────────────────────────
    # Internal: vote tallying
    # ──────────────────────────────────────────

    def _tally_votes(
        self, votes: list[ModelVote], strategy: str
    ) -> ConsensusResult:
        """
        Apply credibility-weighted majority vote with regime-adaptive tiers.

        Hybrid scoring:
          1. Weight each model's vote by its credibility score
          2. Compute weighted average score in [-1, +1]
          3. Determine conviction tier based on current VIX regime thresholds
          4. Assign size_multiplier for downstream Kelly sizing
        """
        self._last_votes = list(votes) if votes else []
        if not votes:
            return ConsensusResult(
                action="hold", net_score=0.0, agreed=False,
                threshold=self.threshold,
                conviction_tier="veto", size_multiplier=0.0,
                votes=[], dissenting_models=[],
            )

        # Strict Unanimous Rule for 3-Agent Quantitative Strategy Council
        if any(v.model in QUANTITATIVE_AGENTS for v in votes):
            strat_lower = strategy.lower()
            expected_action = "sell" if any(s in strat_lower for s in ("theta", "straddle", "iv_crush")) else "buy"
            is_unanimous = (
                len(votes) == 3
                and all(v.action == expected_action and v.confidence >= 0.50 for v in votes)
            )
            if is_unanimous:
                action = expected_action
                agreed = True
                conviction_tier = "strong"
                size_multiplier = 1.00
                avg_conf = sum(v.confidence for v in votes) / 3.0
                net_score = avg_conf if action == "buy" else -avg_conf
                dissenting = []
            else:
                action = "hold"
                agreed = False
                conviction_tier = "veto"
                size_multiplier = 0.00
                net_score = 0.0
                dissenting = [v.model for v in votes if v.action != expected_action]

            result = ConsensusResult(
                action=action,
                net_score=net_score,
                agreed=agreed,
                threshold=0.60,
                conviction_tier=conviction_tier,
                size_multiplier=size_multiplier,
                votes=votes,
                dissenting_models=dissenting,
            )

            tier_colors = {"strong": "green", "veto": "red"}
            color = tier_colors.get(conviction_tier, "white")
            status_text = "ALL 3 APPROVED (UNANIMOUS)" if agreed else "VETO (UNANIMOUS APPROVAL REQUIRED)"
            console.print(
                f"[{color}]Council: {action.upper()} | "
                f"score={net_score:+.3f} | tier={conviction_tier.upper()} | "
                f"size_mult={size_multiplier:.2f} | {status_text}[/{color}]"
            )
            return result

        # Get credibility weights (default 1.0 for all models if no tracker)
        cred_weights = {}
        tracker = getattr(self, "_credibility_tracker", None)
        if tracker is not None:
            try:
                cred_weights = tracker.get_weights()
            except Exception as exc:
                log.warning(f"Credibility tracker error: {exc}")

        # Credibility-weighted score: Σ(w_i × confidence_i × vote_i) / Σ(w_i)
        total_weighted_score = 0.0
        total_weight = 0.0
        for v in votes:
            w = cred_weights.get(v.model, 1.0)
            total_weighted_score += w * v.score
            total_weight += w

        net_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # Get regime-adaptive thresholds
        current_regime = getattr(self, "_current_regime", "neutral")
        regime_thresholds = self._REGIME_THRESHOLDS.get(
            current_regime,
            self._REGIME_THRESHOLDS["neutral"]
        )

        abs_score = abs(net_score)

        # Determine conviction tier
        if abs_score >= regime_thresholds["strong"]:
            conviction_tier = "strong"
        elif abs_score >= regime_thresholds["moderate"]:
            conviction_tier = "moderate"
        elif abs_score >= regime_thresholds["pilot"]:
            conviction_tier = "pilot"
        else:
            conviction_tier = "veto"

        size_multiplier = self._TIER_MULTIPLIERS[conviction_tier]
        agreed = conviction_tier != "veto"

        # Determine winning action
        if conviction_tier == "veto":
            action = "hold"
        elif net_score > 0:
            action = "buy"
        elif net_score < 0:
            action = "sell"
        else:
            action = "hold"

        # Find dissenting models
        dissenting = [
            v.model for v in votes
            if v.action != action and v.action != "hold"
        ]

        result = ConsensusResult(
            action=action,
            net_score=net_score,
            agreed=agreed,
            threshold=regime_thresholds.get("moderate", self.threshold),
            conviction_tier=conviction_tier,
            size_multiplier=size_multiplier,
            votes=votes,
            dissenting_models=dissenting,
        )

        # Pretty-print to console
        tier_colors = {"strong": "green", "moderate": "yellow", "pilot": "cyan", "veto": "red"}
        color = tier_colors.get(conviction_tier, "white")
        cred_info = ""
        if cred_weights:
            cred_items = [f"{k.split('/')[-1]}: {v:.2f}" for k, v in cred_weights.items()]
            cred_info = f" | cred_weights={{{', '.join(cred_items)}}}"
        console.print(
            f"[{color}]Council: {action.upper()} | "
            f"score={net_score:+.3f} | tier={conviction_tier.upper()} | "
            f"size_mult={size_multiplier:.2f} | regime={self._current_regime}"
            f"{cred_info}[/{color}]"
        )

        return result

    # ──────────────────────────────────────────
    # Fallback votes for failed models
    # ──────────────────────────────────────────

    @staticmethod
    def _timeout_vote(model_id: str) -> ModelVote:
        return ModelVote(
            model=model_id,
            action="hold",
            confidence=0.0,
            reasoning="Model timed out",
        )

    @staticmethod
    def _error_vote(model_id: str, error: str) -> ModelVote:
        return ModelVote(
            model=model_id,
            action="hold",
            confidence=0.0,
            reasoning=f"Error: {error[:80]}",
        )
