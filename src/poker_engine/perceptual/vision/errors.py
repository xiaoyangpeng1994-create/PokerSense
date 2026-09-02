"""Perceptual-layer exceptions."""

from __future__ import annotations

from poker_engine.core.errors import PokerEngineError


class TableMapError(PokerEngineError):
    """Base error for TableMap / mapping validation problems."""


class TableMapMismatchError(TableMapError):
    """Raised when actual frame dimensions/ratio are incompatible with a TableMap."""


__all__ = ["TableMapError", "TableMapMismatchError"]
