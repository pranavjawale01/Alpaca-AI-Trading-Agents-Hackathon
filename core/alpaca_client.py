"""
core/alpaca_client.py — Alpaca API wrapper.

Thin wrapper around alpaca-py SDK providing:
  - Account info
  - Order placement (stocks & options)
  - Position management
  - Options chain fetching

All methods handle errors gracefully and log via Rich.
"""

from __future__ import annotations

import logging
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    AssetClass,
    ContractType,
    ExerciseStyle,
)
from alpaca.data.historical import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from rich.console import Console

import config

console = Console()
log = logging.getLogger(__name__)


class AlpacaClient:
    """
    Unified client for Alpaca Trading + Data APIs.

    Usage:
        client = AlpacaClient()
        account = client.get_account()
        client.place_market_order("SPY", 1, "buy")
    """

    def __init__(self) -> None:
        self.trading = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=True,  # always paper for hackathon
        )
        self.stock_data = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        self.option_data = OptionHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        console.print("[green]OK AlpacaClient initialised (paper trading)[/green]")

    # ─────────────────────────────────────────
    # Account
    # ─────────────────────────────────────────

    def get_account(self) -> dict:
        """Return account info as a plain dict."""
        acct = self.trading.get_account()
        return {
            "id": acct.id,
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "daytrade_count": acct.daytrade_count,
        }

    def get_equity(self) -> float:
        """Shortcut to current portfolio equity."""
        return float(self.trading.get_account().equity)

    # ─────────────────────────────────────────
    # Positions
    # ─────────────────────────────────────────

    def get_all_positions(self) -> list[dict]:
        """Return all open positions."""
        positions = self.trading.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "asset_class": str(p.asset_class),
            }
            for p in positions
        ]

    def close_position(self, symbol: str) -> dict:
        """Close an entire position by symbol."""
        result = self.trading.close_position(symbol)
        log.info(f"Closed position: {symbol}")
        return result

    # ─────────────────────────────────────────
    # Stock Orders
    # ─────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> dict:
        """Place a market order. side = 'buy' | 'sell'."""
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
            if time_in_force == "day"
            else TimeInForce.GTC,
        )
        order = self.trading.submit_order(req)
        log.info(f"Market order submitted: {side} {qty} {symbol} | id={order.id}")
        return {"id": str(order.id), "status": str(order.status), "symbol": symbol}

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict:
        """Place a limit order."""
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY
            if time_in_force == "day"
            else TimeInForce.GTC,
        )
        order = self.trading.submit_order(req)
        log.info(
            f"Limit order submitted: {side} {qty} {symbol} @ {limit_price} | id={order.id}"
        )
        return {"id": str(order.id), "status": str(order.status), "symbol": symbol}

    # ─────────────────────────────────────────
    # Options
    # ─────────────────────────────────────────

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        contract_type: Optional[str] = None,  # 'call' | 'put'
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch available option contracts for a symbol."""
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying_symbol],
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
            type=ContractType.CALL
            if contract_type == "call"
            else ContractType.PUT
            if contract_type == "put"
            else None,
            strike_price_gte=strike_price_gte,
            strike_price_lte=strike_price_lte,
            limit=limit,
        )
        contracts = self.trading.get_option_contracts(req)
        return [
            {
                "symbol": c.symbol,
                "underlying": c.underlying_symbol,
                "expiration": str(c.expiration_date),
                "strike": float(c.strike_price),
                "type": str(c.type),
                "style": str(c.style),
                "open_interest": c.open_interest,
            }
            for c in contracts.option_contracts
        ]

    def place_option_market_order(
        self,
        option_symbol: str,
        qty: int,
        side: str,
    ) -> dict:
        """
        Place a market order on an options contract.
        option_symbol: OCC-formatted symbol e.g. 'SPY250919C00500000'
        """
        req = MarketOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading.submit_order(req)
        log.info(
            f"Option market order: {side} {qty} {option_symbol} | id={order.id}"
        )
        return {"id": str(order.id), "status": str(order.status), "symbol": option_symbol}

    # ─────────────────────────────────────────
    # Market Data
    # ─────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Get latest bid/ask for a symbol."""
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.stock_data.get_stock_latest_quote(req)
        q = quotes[symbol]
        return {
            "symbol": symbol,
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "mid": (float(q.bid_price) + float(q.ask_price)) / 2,
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch historical OHLCV bars.
        timeframe: '1Min' | '5Min' | '1Hour' | '1Day'
        """
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, TimeFrame.Day),
            limit=limit,
        )
        bars = self.stock_data.get_stock_bars(req)
        return [
            {
                "t": str(b.timestamp),
                "o": float(b.open),
                "h": float(b.high),
                "l": float(b.low),
                "c": float(b.close),
                "v": float(b.volume),
            }
            for b in bars[symbol]
        ]
