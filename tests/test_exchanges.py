"""Tests for exchange wrappers (no real API calls)."""

import pytest
from unittest.mock import Mock, patch

from exchanges.hyperliquid import HyperliquidExchange


class TestHyperliquidExchange:
    """Test Hyperliquid exchange wrapper."""

    def test_init_testnet(self):
        """Test initialization with testnet."""
        with patch("exchanges.hyperliquid.Info"), patch("exchanges.hyperliquid.Exchange"):
            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="testnet"
            )
            assert exchange.network == "testnet"
            assert exchange.wallet is not None

    def test_init_mainnet(self):
        """Test initialization with mainnet."""
        with patch("exchanges.hyperliquid.Info"), patch("exchanges.hyperliquid.Exchange"):
            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="mainnet"
            )
            assert exchange.network == "mainnet"

    def test_init_invalid_network(self):
        """Test initialization with invalid network."""
        with pytest.raises(ValueError, match="Invalid network"):
            HyperliquidExchange(
                private_key="0x" + "1" * 64, network="invalid"  # type: ignore
            )

    def test_place_order_validation(self):
        """Test order placement validation."""
        with patch("exchanges.hyperliquid.Info"), patch("exchanges.hyperliquid.Exchange"):
            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="testnet"
            )

            # Invalid side
            with pytest.raises(ValueError, match="Invalid side"):
                exchange.place_order(
                    symbol="BTC", side="INVALID", size=0.1, price=42000  # type: ignore
                )

            # Invalid size
            with pytest.raises(ValueError, match="Invalid size"):
                exchange.place_order(symbol="BTC", side="LONG", size=-0.1, price=42000)

            # Limit order without price
            with pytest.raises(ValueError, match="require price"):
                exchange.place_order(
                    symbol="BTC", side="LONG", size=0.1, order_type="limit"
                )

    def test_get_ticker_mock(self):
        """Test ticker retrieval (mocked)."""
        with patch("exchanges.hyperliquid.Info") as mock_info, patch(
            "exchanges.hyperliquid.Exchange"
        ):
            # Mock all_mids response
            mock_info_instance = Mock()
            mock_info_instance.all_mids.return_value = {"BTC": "42000.5", "ETH": "2200.3"}
            mock_info.return_value = mock_info_instance

            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="testnet"
            )

            ticker = exchange.get_ticker("BTC")
            assert ticker["symbol"] == "BTC"
            assert ticker["price"] == 42000.5

    def test_get_position_mock(self):
        """Test position retrieval (mocked)."""
        with patch("exchanges.hyperliquid.Info") as mock_info, patch(
            "exchanges.hyperliquid.Exchange"
        ):
            # Mock user_state response
            mock_info_instance = Mock()
            mock_info_instance.user_state.return_value = {
                "assetPositions": [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": "0.5",  # Long position
                            "entryPx": "41000.0",
                            "unrealizedPnl": "500.0",
                        }
                    }
                ]
            }
            mock_info.return_value = mock_info_instance

            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="testnet"
            )

            position = exchange.get_position("BTC")
            assert position is not None
            assert position["symbol"] == "BTC"
            assert position["size"] == 0.5
            assert position["side"] == "LONG"
            assert position["entry_price"] == 41000.0

    def test_no_position_returns_none(self):
        """Test get_position returns None when no position exists."""
        with patch("exchanges.hyperliquid.Info") as mock_info, patch(
            "exchanges.hyperliquid.Exchange"
        ):
            # Mock empty positions
            mock_info_instance = Mock()
            mock_info_instance.user_state.return_value = {"assetPositions": []}
            mock_info.return_value = mock_info_instance

            exchange = HyperliquidExchange(
                private_key="0x" + "1" * 64, network="testnet"
            )

            position = exchange.get_position("BTC")
            assert position is None
