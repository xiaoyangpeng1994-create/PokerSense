"""Nested deep-freeze helpers for core immutability.

The frozen dataclasses guarantee that their OWN attributes cannot be
reassigned, but Python containers (list/dict/set) held inside those
attributes remain mutable unless they are also made immutable recursively.

These helpers convert mutable containers into immutable ones, recursively:

    list  -> tuple
    set   -> frozenset
    dict  -> MappingProxyType (with each value deep-frozen)

Everything is standard library only (no third-party deps).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Used as the default_factory for timestamp fields so the default value
    is already timezone-aware (never naive).
    """
    return datetime.now(timezone.utc)


def _require_aware_dt(dt: datetime) -> None:
    """Raise TypeError if datetime is naive (no tzinfo).

    All timestamps in the domain MUST be timezone-aware.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise TypeError("timestamp must be timezone-aware")


def deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers to immutable equivalents.

    - ``dict`` -> ``MappingProxyType`` (nested values deep-frozen)
    - ``list`` -> ``tuple`` (nested items deep-frozen)
    - ``set``  -> ``frozenset`` (nested items deep-frozen)
    - ``tuple`` -> new tuple with items deep-frozen
    - other objects are returned unchanged (assumed immutable, e.g. int,
      str, Decimal, Card, ChipAmount, enum members).
    """
    if isinstance(value, dict):
        return MappingProxyType(
            {k: deep_freeze(v) for k, v in value.items()}
        )
    if isinstance(value, list):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(deep_freeze(v) for v in value)
    return value


def freeze_mapping(value: Mapping[Any, Any], /) -> Mapping[Any, Any]:
    """Deep-freeze a mapping and return a read-only Mapping view.

    Convenience wrapper: accepts any mapping, returns a MappingProxyType
    whose values are recursively deep-frozen.
    """
    return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})


__all__ = ["deep_freeze", "freeze_mapping", "utc_now", "_require_aware_dt"]
