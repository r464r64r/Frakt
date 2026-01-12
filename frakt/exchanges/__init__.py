"""
Generic exchange interfaces.

Frakt provides wrappers for common exchanges, but does NOT include:
- API credentials
- Order execution logic
- Position management

These belong in your application layer (e.g., FraktAl).
"""

from frakt.exchanges.hyperliquid import HyperliquidExchange

__all__ = ["HyperliquidExchange"]
