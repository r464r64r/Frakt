"""
Hyperliquid exchange API wrapper.

Generic wrapper for Hyperliquid exchange. Provides basic order placement
and market data fetching. Does NOT include business logic.

Usage:
    from frakt.exchanges import HyperliquidExchange

    exchange = HyperliquidExchange(
        private_key="0x...",
        network="testnet"
    )

    # Place order
    result = exchange.place_order(
        symbol="BTC",
        side="LONG",
        size=0.1,
        price=42000,
        order_type="limit"
    )
"""

import logging
from typing import Literal, Optional

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

logger = logging.getLogger(__name__)


class HyperliquidExchange:
    """Generic Hyperliquid API client (no business logic)."""

    def __init__(
        self,
        private_key: str,
        network: Literal["mainnet", "testnet"] = "testnet",
    ):
        """
        Initialize exchange client.

        Args:
            private_key: Ethereum private key (0x...)
            network: "mainnet" or "testnet"

        Raises:
            ValueError: If network is invalid
        """
        if network not in ("mainnet", "testnet"):
            raise ValueError(f"Invalid network: {network}")

        self.network = network
        self.wallet = Account.from_key(private_key)

        # Initialize Hyperliquid clients
        api_url = constants.TESTNET_API_URL if network == "testnet" else constants.MAINNET_API_URL
        self.info = Info(api_url, skip_ws=True)
        self.exchange = Exchange(self.wallet, api_url)

        logger.info(f"HyperliquidExchange initialized (network={network})")

    def get_account_state(self) -> dict:
        """
        Get account state (balances, positions, etc.).

        Returns:
            dict: Account state from Hyperliquid API
        """
        try:
            state = self.info.user_state(self.wallet.address)
            return state
        except Exception as e:
            logger.error(f"Failed to get account state: {e}")
            raise

    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker for symbol.

        Args:
            symbol: Symbol (e.g., "BTC")

        Returns:
            dict: Ticker data (price, volume, etc.)
        """
        try:
            all_mids = self.info.all_mids()
            # all_mids is dict like {"BTC": "42000.5", "ETH": "2200.3"}
            if symbol not in all_mids:
                raise ValueError(f"Symbol not found: {symbol}")

            return {
                "symbol": symbol,
                "price": float(all_mids[symbol]),
            }
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    def place_order(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        size: float,
        price: Optional[float] = None,
        order_type: Literal["limit", "market"] = "limit",
        reduce_only: bool = False,
    ) -> dict:
        """
        Place order on Hyperliquid.

        Args:
            symbol: Symbol (e.g., "BTC")
            side: "LONG" or "SHORT"
            size: Size in base currency (e.g., 0.1 BTC)
            price: Limit price (required for limit orders)
            order_type: "limit" or "market"
            reduce_only: If True, order can only reduce position

        Returns:
            dict: Order result from exchange

        Raises:
            ValueError: If parameters are invalid
        """
        if side not in ("LONG", "SHORT"):
            raise ValueError(f"Invalid side: {side}")
        if size <= 0:
            raise ValueError(f"Invalid size: {size}")
        if order_type == "limit" and price is None:
            raise ValueError("Limit orders require price")

        is_buy = side == "LONG"

        try:
            # Hyperliquid order structure
            order = {
                "coin": symbol,
                "is_buy": is_buy,
                "sz": size,
                "limit_px": price if order_type == "limit" else None,
                "order_type": {"limit": price} if order_type == "limit" else {"market": {}},
                "reduce_only": reduce_only,
            }

            result = self.exchange.order(order)
            logger.info(f"Order placed: {symbol} {side} {size} @ {price}")
            return result

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancel order.

        Args:
            symbol: Symbol (e.g., "BTC")
            order_id: Order ID to cancel

        Returns:
            dict: Cancel result
        """
        try:
            result = self.exchange.cancel(symbol, order_id)
            logger.info(f"Order canceled: {symbol} #{order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Get open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            list[dict]: Open orders
        """
        try:
            state = self.get_account_state()
            orders = state.get("assetPositions", [])

            if symbol:
                orders = [o for o in orders if o.get("position", {}).get("coin") == symbol]

            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[dict]:
        """
        Get current position for symbol.

        Args:
            symbol: Symbol (e.g., "BTC")

        Returns:
            dict or None: Position info (size, entry price, etc.)
        """
        try:
            state = self.get_account_state()
            positions = state.get("assetPositions", [])

            for pos in positions:
                if pos.get("position", {}).get("coin") == symbol:
                    position_data = pos.get("position", {})
                    size = float(position_data.get("szi", "0"))

                    if size != 0:
                        return {
                            "symbol": symbol,
                            "size": abs(size),
                            "side": "LONG" if size > 0 else "SHORT",
                            "entry_price": float(position_data.get("entryPx", "0")),
                            "unrealized_pnl": float(position_data.get("unrealizedPnl", "0")),
                        }

            return None  # No position
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            raise

    def get_all_positions(self) -> list[dict]:
        """
        Get all open positions.

        Returns:
            list[dict]: All positions
        """
        try:
            state = self.get_account_state()
            positions = []

            for pos in state.get("assetPositions", []):
                position_data = pos.get("position", {})
                size = float(position_data.get("szi", "0"))

                if size != 0:
                    symbol = position_data.get("coin")
                    positions.append(
                        {
                            "symbol": symbol,
                            "size": abs(size),
                            "side": "LONG" if size > 0 else "SHORT",
                            "entry_price": float(position_data.get("entryPx", "0")),
                            "unrealized_pnl": float(position_data.get("unrealizedPnl", "0")),
                        }
                    )

            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise
