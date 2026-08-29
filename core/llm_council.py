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
    votes: list[ModelVote] = field(default_factory=list)
    dissenting_models: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Council verdict: {self.action.upper()} | "
            f"score={self.net_score:+.3f} | agreed={self.agreed}",
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

    All three models are queried **in parallel** (ThreadPoolExecutor) to keep
    latency similar to a single model call. If a model times out or errors,
    it casts a 'hold' vote at zero confidence (neutral, does not block trade).
    """

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
                    f"[bold green]LLMCouncil initialised | "
                    f"{len(self.models)} models | threshold={self.threshold:.2f}[/bold green]"
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
            max_tokens=200,
        )

        raw = response.choices[0].message.content or ""
        return self._parse_vote(model_id, raw)

    def _parse_vote(self, model_id: str, raw: str) -> ModelVote:
        """Parse a model's JSON response into a ModelVote."""
        try:
            # Find JSON object in response
            if "{" in raw:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                data = json.loads(raw[start:end])

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
        """Apply confidence-weighted majority vote formula."""
        if not votes:
            return ConsensusResult(
                action="hold", net_score=0.0, agreed=False,
                threshold=self.threshold, votes=[], dissenting_models=[],
            )

        # Weighted sum: buy=+1, hold=0, sell=-1, each scaled by confidence
        total_score = sum(v.score for v in votes)
        net_score = total_score / len(votes)  # normalised to [-1, +1]

        # Determine winning action
        if net_score >= self.threshold:
            action = "buy"
        elif net_score <= -self.threshold:
            action = "sell"
        else:
            action = "hold"

        agreed = abs(net_score) >= self.threshold

        # Find dissenting models
        dissenting = [
            v.model for v in votes
            if v.action != action and v.action != "hold"
        ]

        result = ConsensusResult(
            action=action,
            net_score=net_score,
            agreed=agreed,
            threshold=self.threshold,
            votes=votes,
            dissenting_models=dissenting,
        )

        # Pretty-print to console
        score_color = "green" if agreed else "yellow"
        console.print(
            f"[{score_color}]Council: {action.upper()} | "
            f"score={net_score:+.3f} | agreed={agreed}[/{score_color}]"
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
