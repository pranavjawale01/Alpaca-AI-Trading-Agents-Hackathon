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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from rich.console import Console

import config

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


class LLMCouncil:
    """
    Three-model voting council for high-conviction trade signal filtering.

    Hybrid greedy-voting features:
      - Credibility-weighted votes: models with better track records vote louder
      - Regime-adaptive thresholds: relaxed in bull markets, strict in panic
      - Conviction tiers: STRONG/MODERATE/PILOT instead of binary pass/fail
      - Pilot positions: weak consensus still trades at reduced size

    All three models are queried **in parallel** (ThreadPoolExecutor) to keep
    latency similar to a single model call. If a model times out or errors,
    it casts a 'hold' vote at zero confidence (neutral, does not block trade).
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

    def __init__(
        self,
        models: Optional[list[str]] = None,
        threshold: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.models = models or config.COUNCIL.models
        self.threshold = threshold if threshold is not None else config.COUNCIL.consensus_threshold
        self.timeout = timeout if timeout is not None else config.COUNCIL.timeout_seconds
        self.enabled = config.COUNCIL.enabled

        # Current VIX regime — updated by Orchestrator before each session
        self._current_regime: str = "neutral"

        # Credibility tracker — set externally by Orchestrator after init
        self._credibility_tracker = None

        self._client: Optional[OpenAI] = None
        self._available = False

        if config.FEATHERLESS_API_KEY:
            try:
                self._client = OpenAI(
                    api_key=config.FEATHERLESS_API_KEY,
                    base_url=config.FEATHERLESS_BASE_URL,
                )
                self._available = True
                console.print(
                    f"[bold green]LLMCouncil initialised (HYBRID MODE) | "
                    f"{len(self.models)} models | base_threshold={self.threshold:.2f}[/bold green]"
                )
                for m in self.models:
                    console.print(f"  [dim]• {m}[/dim]")
            except Exception as exc:
                log.warning(f"LLMCouncil: client init failed ({exc}) — council disabled")
        else:
            console.print(
                "[yellow]LLMCouncil: No FEATHERLESS_API_KEY — "
                "council disabled, all signals pass through[/yellow]"
            )

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
        Ask all council models to vote on a trade signal.

        Args:
            symbol: Ticker symbol (for logging)
            market_context: Dict of market data (price, VIX, IVR, signals, etc.)
            strategy: One of 'momentum_call', 'theta_put', 'iv_crush_straddle', 'general'

        Returns:
            ConsensusResult — check `.agreed` before placing order.
        """
        # If council is disabled or no API key, auto-approve (pass-through)
        if not self.enabled or not self._available:
            return ConsensusResult(
                action="buy", net_score=1.0, agreed=True,
                threshold=self.threshold,
                conviction_tier="strong", size_multiplier=1.0,
                votes=[], dissenting_models=[],
            )

        context_str = json.dumps(market_context, indent=2)
        votes = self._gather_votes(symbol, context_str, strategy)
        return self._tally_votes(votes, strategy)

    # ──────────────────────────────────────────
    # Internal: parallel model querying
    # ──────────────────────────────────────────

    def _gather_votes(
        self, symbol: str, context_str: str, strategy: str
    ) -> list[ModelVote]:
        """Query all models in parallel, collect votes with timeout."""
        votes: list[ModelVote] = []

        with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
            futures = {
                pool.submit(self._query_model, model, context_str, strategy): model
                for model in self.models
            }
            for future in as_completed(futures, timeout=self.timeout + 2):
                model = futures[future]
                try:
                    vote = future.result(timeout=self.timeout)
                    votes.append(vote)
                    log.debug(
                        f"[{symbol}] {model}: {vote.action} "
                        f"(conf={vote.confidence:.2f})"
                    )
                except TimeoutError:
                    log.warning(f"[{symbol}] {model} timed out — casting hold")
                    votes.append(self._timeout_vote(model))
                except Exception as exc:
                    log.warning(f"[{symbol}] {model} error ({exc}) — casting hold")
                    votes.append(self._error_vote(model, str(exc)))

        return votes

    def _query_model(
        self, model_id: str, context_str: str, strategy: str
    ) -> ModelVote:
        """Query one LLM model and parse its JSON response."""
        prompt_template = _STRATEGY_PROMPTS.get(strategy, _STRATEGY_PROMPTS["general"])
        user_prompt = prompt_template.format(context=context_str)

        response = self._client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # low temp for deterministic trading decisions
            max_tokens=500,
        )

        raw = response.choices[0].message.content or ""
        return self._parse_vote(model_id, raw)

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
        if not votes:
            return ConsensusResult(
                action="hold", net_score=0.0, agreed=False,
                threshold=self.threshold,
                conviction_tier="veto", size_multiplier=0.0,
                votes=[], dissenting_models=[],
            )

        # Get credibility weights (default 1.0 for all models if no tracker)
        cred_weights = {}
        if self._credibility_tracker is not None:
            try:
                cred_weights = self._credibility_tracker.get_weights()
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
        regime_thresholds = self._REGIME_THRESHOLDS.get(
            self._current_regime,
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
