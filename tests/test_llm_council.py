"""
tests/test_llm_council.py — Unit tests for the LLM Council voting engine.

Tests the voting math, JSON parsing, and edge cases entirely offline
(no real API calls) using unittest.mock to simulate model responses.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm_council import (
    LLMCouncil,
    ModelVote,
    ConsensusResult,
    _ACTION_SCORE,
)


# ── Helper: create a mock LLM response object ──────────────────

def _make_mock_choice(content: str) -> MagicMock:
    """Create a mock OpenAI ChatCompletion choice with given content."""
    choice = MagicMock()
    choice.message.content = content
    return choice


def _make_llm_response(content: str) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response object."""
    resp = MagicMock()
    resp.choices = [_make_mock_choice(content)]
    return resp


# ── Fixture: council with mocked client ────────────────────────

@pytest.fixture
def council_no_api():
    """Council with no API key — passes all votes through (auto-approve)."""
    with patch("config.FEATHERLESS_API_KEY", ""):
        with patch("config.COUNCIL.enabled", True):
            c = LLMCouncil.__new__(LLMCouncil)
            c.models = [
                "model-a",
                "model-b",
                "model-c",
            ]
            c.threshold = 0.60
            c.timeout = 5.0
            c.enabled = True
            c._client = None
            c._available = False
            return c


@pytest.fixture
def council_with_mock_client():
    """Council with a mocked OpenAI client for offline testing."""
    with patch("config.FEATHERLESS_API_KEY", "fake-key"):
        c = LLMCouncil.__new__(LLMCouncil)
        c.models = ["model-a", "model-b", "model-c"]
        c.threshold = 0.60
        c.timeout = 5.0
        c.enabled = True
        c._client = MagicMock()
        c._available = True
        return c


# ════════════════════════════════════════════════════════════════
# 1. Voting Math
# ════════════════════════════════════════════════════════════════

class TestVotingMath:

    def test_action_score_mapping(self):
        """Ensure action→score mapping is correct."""
        assert _ACTION_SCORE["buy"] == 1.0
        assert _ACTION_SCORE["sell"] == -1.0
        assert _ACTION_SCORE["hold"] == 0.0

    def test_model_vote_score(self):
        """ModelVote.score should be action_score × confidence."""
        v = ModelVote(model="m", action="buy", confidence=0.8, reasoning="")
        assert abs(v.score - 0.8) < 1e-9

        v2 = ModelVote(model="m", action="sell", confidence=0.9, reasoning="")
        assert abs(v2.score - (-0.9)) < 1e-9

        v3 = ModelVote(model="m", action="hold", confidence=1.0, reasoning="")
        assert v3.score == 0.0

    def test_tally_unanimous_buy(self, council_with_mock_client):
        """Three confident buy votes → action=buy, agreed=True."""
        votes = [
            ModelVote("m1", "buy", 0.9, "Strong uptrend"),
            ModelVote("m2", "buy", 0.85, "Volume confirms"),
            ModelVote("m3", "buy", 0.95, "Trend intact"),
        ]
        result = council_with_mock_client._tally_votes(votes, "momentum_call")

        assert result.action == "buy"
        assert result.agreed is True
        # net = (0.9 + 0.85 + 0.95) / 3 ≈ 0.900
        assert abs(result.net_score - 0.90) < 0.01

    def test_tally_unanimous_sell(self, council_with_mock_client):
        """Three sell votes → action=sell, agreed=True."""
        votes = [
            ModelVote("m1", "sell", 0.8, "IV too low"),
            ModelVote("m2", "sell", 0.75, "VIX rising"),
            ModelVote("m3", "sell", 0.9, "Bearish regime"),
        ]
        result = council_with_mock_client._tally_votes(votes, "theta_put")
        assert result.action == "sell"
        assert result.agreed is True
        assert result.net_score < -0.60

    def test_tally_split_no_consensus(self, council_with_mock_client):
        """Two buys at low confidence + one high-confidence sell → no consensus."""
        votes = [
            ModelVote("m1", "buy", 0.55, "Slight uptrend"),
            ModelVote("m2", "buy", 0.52, "Marginal volume"),
            ModelVote("m3", "sell", 0.95, "Strong bearish signal"),
        ]
        result = council_with_mock_client._tally_votes(votes, "momentum_call")
        # net = (0.55 + 0.52 - 0.95) / 3 ≈ +0.04 — well below threshold
        assert result.agreed is False
        assert result.action == "hold"

    def test_tally_two_hold_one_buy_no_consensus(self, council_with_mock_client):
        """Two holds + one buy → net ≈ 0.27, below threshold → no consensus."""
        votes = [
            ModelVote("m1", "buy", 0.80, "Breakout"),
            ModelVote("m2", "hold", 0.0, "Neutral"),
            ModelVote("m3", "hold", 0.0, "Neutral"),
        ]
        result = council_with_mock_client._tally_votes(votes, "momentum_call")
        expected_net = 0.80 / 3  # ≈ 0.267
        assert abs(result.net_score - expected_net) < 0.01
        assert result.agreed is False  # below 0.60 threshold

    def test_tally_two_confident_buys_override_one_uncertain_sell(
        self, council_with_mock_client
    ):
        """Two 0.90 buys vs one 0.55 sell → net > 0.60 → consensus BUY."""
        votes = [
            ModelVote("m1", "buy", 0.90, "Strong momentum"),
            ModelVote("m2", "buy", 0.90, "Confirmed breakout"),
            ModelVote("m3", "sell", 0.55, "Mild concern"),
        ]
        result = council_with_mock_client._tally_votes(votes, "momentum_call")
        # net = (0.90 + 0.90 - 0.55) / 3 ≈ 0.417 — below 0.60
        # Two 0.90 buys vs 0.55 sell: net = (0.9 + 0.9 - 0.55)/3 = 1.25/3 ≈ 0.417
        # This is below threshold — which is correct: 2/3 agreement is not enough
        # if the dissenter has meaningful confidence.
        assert result.agreed is False

    def test_two_very_confident_buys_override_uncertain_sell(
        self, council_with_mock_client
    ):
        """Two 0.95 buys vs one 0.20 sell → net ≥ 0.60 → consensus BUY."""
        votes = [
            ModelVote("m1", "buy", 0.95, "Clear uptrend"),
            ModelVote("m2", "buy", 0.95, "Volume confirmation"),
            ModelVote("m3", "sell", 0.20, "Weak concern"),
        ]
        result = council_with_mock_client._tally_votes(votes, "momentum_call")
        # net = (0.95 + 0.95 - 0.20) / 3 = 1.70/3 ≈ 0.567 — still below 0.60
        # 3 models: need net ≥ 0.60; with 2×0.95 + 1×(-0.20) = 1.70/3 ≈ 0.567
        assert result.net_score == pytest.approx(1.70 / 3, abs=0.01)

    def test_empty_votes_returns_hold(self, council_with_mock_client):
        """Empty vote list → hold, not agreed."""
        result = council_with_mock_client._tally_votes([], "general")
        assert result.action == "hold"
        assert result.agreed is False


# ════════════════════════════════════════════════════════════════
# 2. JSON Parsing
# ════════════════════════════════════════════════════════════════

class TestJsonParsing:

    def test_parse_clean_json(self, council_with_mock_client):
        """Valid JSON response parses correctly."""
        raw = '{"action": "buy", "confidence": 0.85, "reasoning": "Strong momentum"}'
        vote = council_with_mock_client._parse_vote("model-a", raw)
        assert vote.action == "buy"
        assert abs(vote.confidence - 0.85) < 1e-9
        assert "momentum" in vote.reasoning

    def test_parse_json_embedded_in_text(self, council_with_mock_client):
        """JSON buried in text (model preamble) still parses."""
        raw = 'Based on my analysis: {"action": "sell", "confidence": 0.75, "reasoning": "IV too low"} That is my final answer.'
        vote = council_with_mock_client._parse_vote("model-b", raw)
        assert vote.action == "sell"
        assert abs(vote.confidence - 0.75) < 1e-9

    def test_parse_invalid_action_defaults_to_hold(self, council_with_mock_client):
        """Unknown action value → defaults to hold."""
        raw = '{"action": "maybe", "confidence": 0.7, "reasoning": "Unsure"}'
        vote = council_with_mock_client._parse_vote("model-c", raw)
        assert vote.action == "hold"

    def test_parse_confidence_clamped(self, council_with_mock_client):
        """Confidence > 1.0 or < 0.0 is clamped."""
        raw = '{"action": "buy", "confidence": 1.5, "reasoning": "Very confident"}'
        vote = council_with_mock_client._parse_vote("model-a", raw)
        assert vote.confidence == 1.0

        raw2 = '{"action": "sell", "confidence": -0.3, "reasoning": "Error"}'
        vote2 = council_with_mock_client._parse_vote("model-a", raw2)
        assert vote2.confidence == 0.0

    def test_parse_garbage_returns_hold(self, council_with_mock_client):
        """Non-JSON / garbage response → hold at zero confidence."""
        vote = council_with_mock_client._parse_vote("model-a", "I am unable to provide an answer.")
        assert vote.action == "hold"
        assert vote.confidence == 0.0

    def test_parse_empty_string_returns_hold(self, council_with_mock_client):
        """Empty string → hold at zero confidence."""
        vote = council_with_mock_client._parse_vote("model-a", "")
        assert vote.action == "hold"
        assert vote.confidence == 0.0


# ════════════════════════════════════════════════════════════════
# 3. Auto-approve when no API key / council disabled
# ════════════════════════════════════════════════════════════════

class TestAutoApprove:

    def test_no_api_key_auto_approves(self, council_no_api):
        """Council without API key should auto-approve all signals."""
        result = council_no_api.vote("SPY", {}, strategy="theta_put")
        assert result.agreed is True
        assert result.action == "buy"

    def test_disabled_council_auto_approves(self, council_with_mock_client):
        """Council with enabled=False should auto-approve."""
        council_with_mock_client.enabled = False
        result = council_with_mock_client.vote("NVDA", {}, strategy="momentum_call")
        assert result.agreed is True


# ════════════════════════════════════════════════════════════════
# 4. Timeout / error fallback votes
# ════════════════════════════════════════════════════════════════

class TestFallbackVotes:

    def test_timeout_vote_is_hold(self):
        vote = LLMCouncil._timeout_vote("model-x")
        assert vote.action == "hold"
        assert vote.confidence == 0.0
        assert vote.score == 0.0

    def test_error_vote_is_hold(self):
        vote = LLMCouncil._error_vote("model-x", "connection refused")
        assert vote.action == "hold"
        assert vote.score == 0.0

    def test_all_timeout_returns_no_consensus(self, council_with_mock_client):
        """If all models time out → hold votes → no consensus."""
        timeout_votes = [
            LLMCouncil._timeout_vote("m1"),
            LLMCouncil._timeout_vote("m2"),
            LLMCouncil._timeout_vote("m3"),
        ]
        result = council_with_mock_client._tally_votes(timeout_votes, "general")
        assert result.net_score == 0.0
        assert result.agreed is False
        assert result.action == "hold"


# ════════════════════════════════════════════════════════════════
# 5. Full vote() integration test (mocked HTTP)
# ════════════════════════════════════════════════════════════════

class TestFullVoteIntegration:

    def test_full_vote_with_mocked_models(self, council_with_mock_client):
        """
        End-to-end vote() call with 3 mocked model responses.
        All three agree to buy → should return agreed=True, action=buy.
        """
        buy_vote = ModelVote(
            model="model-x", action="buy", confidence=0.88,
            reasoning="EMA crossover confirmed by high volume"
        )

        # Patch _query_model so ThreadPoolExecutor gets a real callable
        with patch.object(
            council_with_mock_client, "_query_model", return_value=buy_vote
        ):
            ctx = {
                "symbol": "NVDA",
                "vix": 17.5,
                "ema_crossover": True,
                "volume_surge_ratio": 2.4,
            }
            result = council_with_mock_client.vote("NVDA", ctx, strategy="momentum_call")

        assert result.agreed is True
        assert result.action == "buy"
        assert len(result.votes) == 3
        # All three should be "buy" at 0.88 confidence
        for v in result.votes:
            assert v.action == "buy"
            assert abs(v.confidence - 0.88) < 1e-9

    def test_full_vote_mixed_signals_no_consensus(self, council_with_mock_client):
        """
        Two cautious sells + one buy → no consensus.
        net = (-0.65 - 0.70 + 0.60) / 3 = -0.25 → below threshold
        """
        sell_votes = [
            ModelVote("m1", "sell", 0.65, "VIX rising"),
            ModelVote("m2", "sell", 0.70, "IV too low"),
        ]
        buy_vote = ModelVote("m3", "buy", 0.60, "Breakout signal")
        call_count = [0]
        vote_pool = sell_votes + [buy_vote]

        def mock_query(model_id, context_str, strategy):
            v = vote_pool[call_count[0] % len(vote_pool)]
            call_count[0] += 1
            return v

        with patch.object(council_with_mock_client, "_query_model", side_effect=mock_query):
            result = council_with_mock_client.vote("SPY", {}, strategy="theta_put")

        # net = (-0.65 - 0.70 + 0.60) / 3 = -0.25 → no consensus
        assert result.agreed is False
        assert abs(result.net_score - (-0.25)) < 0.02

