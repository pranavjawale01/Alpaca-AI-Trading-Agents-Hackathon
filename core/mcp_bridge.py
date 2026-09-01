"""
core/mcp_bridge.py — Alpaca MCP Server integration.

Connects the AI agent to Alpaca's MCP (Model Context Protocol) server,
which lets LLMs interact with Alpaca's APIs through structured tools.

The MCP server exposes Alpaca trading functions as "tools" that an LLM
can call directly — account queries, order placement, position checks, etc.

Usage:
    bridge = MCPBridge()
    result = bridge.query("What is my current portfolio value?")
    result = bridge.execute_tool("get_account", {})
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Optional

from openai import OpenAI
from rich.console import Console

import config
from core.model_discovery import discover_available_models

console = Console()
log = logging.getLogger(__name__)


# MCP Server tools available (mirrors Alpaca MCP server tool registry)
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account",
            "description": "Get current Alpaca paper account information including equity, cash, and buying power.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "Get all current open positions in the paper trading account.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a buy or sell order for stocks or options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "qty": {"type": "number", "description": "Number of shares/contracts"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "type": {"type": "string", "enum": ["market", "limit"]},
                    "limit_price": {"type": "number", "description": "Required for limit orders"},
                },
                "required": ["symbol", "qty", "side", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_option_chain",
            "description": "Get available option contracts for an underlying symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "expiration_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "option_type": {"type": "string", "enum": ["call", "put", "all"]},
                },
                "required": ["symbol"],
            },
        },
    },
]


class MCPBridge:
    """
    Bridge between AI agents and Alpaca's MCP Server.

    Uses Featherless AI (OpenAI-compatible API) to run an LLM that
    can invoke MCP-registered Alpaca tools for natural-language trading.
    """

    def __init__(self, alpaca_client=None) -> None:
        """
        Args:
            alpaca_client: Optional AlpacaClient instance for tool execution.
                           If None, tool calls will be simulated/logged.
        """
        self.alpaca_client = alpaca_client
        self.model = config.FEATHERLESS_MODEL

        # Featherless AI client (OpenAI-compatible)
        self.llm = None
        self._llm_available = False
        self._discovered_models: list[str] = []
        if config.FEATHERLESS_API_KEY:
            try:
                self.llm = OpenAI(
                    api_key=config.FEATHERLESS_API_KEY,
                    base_url=config.FEATHERLESS_BASE_URL,
                )
                self._llm_available = True

                # Discover available models
                self._discovered_models = discover_available_models(
                    client=self.llm,
                    base_url=config.FEATHERLESS_BASE_URL,
                    api_key=config.FEATHERLESS_API_KEY,
                )
                if not os.getenv("FEATHERLESS_MODEL") and self._discovered_models:
                    self.model = self._discovered_models[0]

                console.print(f"[green]OK MCPBridge initialised | model={self.model}[/green]")
            except Exception as e:
                log.warning(f"MCPBridge: LLM init failed ({e}) — running without LLM")
        else:
            console.print("[yellow]MCPBridge: No FEATHERLESS_API_KEY — LLM disabled, rules-based fallback active[/yellow]")

    def query(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        """
        Send a natural language query to the LLM with MCP tools available.

        Args:
            user_message: Natural language instruction
            system_prompt: Override default system prompt
            context: Additional context dict (account info, market data, etc.)

        Returns:
            LLM response text (may include tool call results)
        """
        if system_prompt is None:
            system_prompt = (
                "You are Cache Me, an autonomous AI trading agent specialising in "
                "options strategies on Alpaca's paper trading platform. "
                "You manage a $100,000 portfolio using theta decay, IV crush, and "
                "momentum strategies. Always apply risk gates before any trade. "
                "Be concise and precise in your trading decisions."
            )

        messages = [{"role": "system", "content": system_prompt}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{json.dumps(context, indent=2)}\n\n{user_message}",
            })
        else:
            messages.append({"role": "user", "content": user_message})

        # Fallback when LLM is not configured
        if not self._llm_available:
            log.debug("MCPBridge: LLM not available, returning empty response")
            return "[]"

        attempt_models = [self.model]
        if hasattr(self, "_discovered_models"):
            for m in self._discovered_models:
                if m not in attempt_models:
                    attempt_models.append(m)

        reply = None
        for current_model in attempt_models:
            try:
                response = self.llm.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=MCP_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,  # low temperature for trading decisions
                    max_tokens=1024,
                )

                if current_model != self.model:
                    self.model = current_model
                    console.print(f"[cyan][MCPBridge AUTO-HEAL] Swapped model -> '{current_model}'[/cyan]")

                reply = response.choices[0]
                break
            except Exception as exc:
                err_str = str(exc).lower()
                if any(err_kw in err_str for err_kw in ["not supported", "400", "404", "model_not_found", "does not exist"]):
                    continue
                # If tool choice is unsupported by smaller models, retry without tools
                if "tools" in err_str or "function" in err_str:
                    try:
                        response = self.llm.chat.completions.create(
                            model=current_model,
                            messages=messages,
                            temperature=0.1,
                            max_tokens=1024,
                        )
                        reply = response.choices[0]
                        break
                    except Exception:
                        continue
                log.warning(f"MCPBridge query failed on {current_model}: {exc}")
                return "[]"

        if reply is None:
            log.warning("MCPBridge: All models failed for query")
            return "[]"

        # Handle tool calls
        if reply.finish_reason == "tool_calls" and reply.message.tool_calls:
            tool_results = []
            for tool_call in reply.message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
                tool_results.append({
                    "tool": tool_call.function.name,
                    "result": result,
                })

            # Return formatted tool results
            return json.dumps(tool_results, indent=2)

        return reply.message.content or ""

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute an MCP tool call using the AlpacaClient."""
        log.info(f"MCP tool call: {tool_name}({args})")

        if self.alpaca_client is None:
            log.warning(f"No AlpacaClient — simulating tool: {tool_name}")
            return {"simulated": True, "tool": tool_name, "args": args}

        try:
            if tool_name == "get_account":
                return self.alpaca_client.get_account()
            elif tool_name == "get_positions":
                return self.alpaca_client.get_all_positions()
            elif tool_name == "place_order":
                side = args.get("side", "buy")
                order_type = args.get("type", "market")
                if order_type == "market":
                    return self.alpaca_client.place_market_order(
                        args["symbol"], args["qty"], side
                    )
                else:
                    return self.alpaca_client.place_limit_order(
                        args["symbol"], args["qty"], side, args.get("limit_price", 0)
                    )
            elif tool_name == "get_option_chain":
                return self.alpaca_client.get_option_contracts(
                    args["symbol"],
                    contract_type=args.get("option_type"),
                )
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            log.error(f"Tool execution failed [{tool_name}]: {e}")
            return {"error": str(e)}

    def get_trading_decision(
        self,
        symbol: str,
        market_context: dict,
        strategy: str = "theta",
    ) -> dict:
        """
        Ask the LLM for a trading decision given market context.

        Returns:
            {
                "action": "buy" | "sell" | "hold",
                "symbol": str,
                "option_type": "call" | "put" | None,
                "reasoning": str,
                "confidence": float  # 0-1
            }
        """
        prompt = (
            f"Given this market context for {symbol}, what is the best {strategy} "
            f"options trade action? Respond in JSON with keys: "
            f"action (buy/sell/hold), option_type (call/put/null), "
            f"reasoning (1 sentence), confidence (0.0-1.0)."
        )

        response_text = self.query(prompt, context=market_context)

        try:
            # Try to parse JSON response
            if "{" in response_text:
                start = response_text.index("{")
                end = response_text.rindex("}") + 1
                decision = json.loads(response_text[start:end])
                decision["symbol"] = symbol
                return decision
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: hold
        return {
            "action": "hold",
            "symbol": symbol,
            "option_type": None,
            "reasoning": "Could not parse LLM response",
            "confidence": 0.0,
        }
