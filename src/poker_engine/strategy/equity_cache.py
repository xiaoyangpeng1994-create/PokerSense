"""Canonical, versioned in-memory cache for strategy equity results."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from threading import RLock

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.value_objects import Card

from .contracts import PotState, RangeDistribution
from .multiway_equity import MultiwayEquityResult


class EquityMethod(str, Enum):
    EXACT = "exact"
    MONTE_CARLO = "monte_carlo"


class EquityCacheState(str, Enum):
    HIT = "HIT"
    NOT_FOUND = "NOT_FOUND"
    STALE = "STALE"


@dataclass(frozen=True)
class EquityCacheQuery:
    hero_seat: int
    hero_cards: tuple[Card, Card]
    board_cards: tuple[Card, ...]
    villain_ranges: tuple[RangeDistribution, ...]
    pots: tuple[PotState, ...]
    method: EquityMethod
    engine_version: str
    trials: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hero_seat, int) or isinstance(self.hero_seat, bool):
            raise TypeError("hero_seat must be an int")
        hero = tuple(self.hero_cards)
        board = tuple(self.board_cards)
        ranges = tuple(self.villain_ranges)
        pots = tuple(self.pots)
        if len(hero) != 2 or not all(isinstance(card, Card) for card in hero):
            raise TypeError("hero_cards must contain exactly two Cards")
        if len(board) not in (0, 3, 4, 5) or not all(
            isinstance(card, Card) for card in board
        ):
            raise ValueError("board_cards must contain 0, 3, 4, or 5 Cards")
        if len(set(hero + board)) != len(hero + board):
            raise ValueError("hero and board cards must be distinct")
        if not ranges or not all(
            isinstance(item, RangeDistribution) for item in ranges
        ):
            raise TypeError("villain_ranges must contain values")
        if len({item.seat_id for item in ranges}) != len(ranges):
            raise ValueError("villain_ranges must have unique seat IDs")
        if self.hero_seat in {item.seat_id for item in ranges}:
            raise ValueError("hero_seat cannot also have a villain range")
        if not pots or not all(isinstance(item, PotState) for item in pots):
            raise TypeError("pots must contain values")
        if len({item.pot_id for item in pots}) != len(pots):
            raise ValueError("pots must have unique pot IDs")
        known_seats = {self.hero_seat} | {item.seat_id for item in ranges}
        if not all(set(item.eligible_seats) <= known_seats for item in pots):
            raise ValueError("pot eligibility references a seat without a range")
        if not isinstance(self.method, EquityMethod):
            raise TypeError("method must be an EquityMethod")
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version must be a non-empty str")
        if self.method is EquityMethod.EXACT:
            if self.trials is not None or self.seed is not None:
                raise ValueError("exact queries cannot specify trials or seed")
        else:
            if not isinstance(self.trials, int) or isinstance(self.trials, bool):
                raise TypeError("Monte Carlo trials must be an int")
            if self.trials <= 0:
                raise ValueError("Monte Carlo trials must be > 0")
            if not isinstance(self.seed, int) or isinstance(self.seed, bool):
                raise TypeError("cacheable Monte Carlo seed must be an int")
        object.__setattr__(self, "hero_cards", hero)
        object.__setattr__(self, "board_cards", board)
        object.__setattr__(self, "villain_ranges", ranges)
        object.__setattr__(self, "pots", pots)

    @property
    def key(self) -> str:
        payload = {
            "schema": 1,
            "hero_seat": self.hero_seat,
            "hero_cards": sorted(str(card) for card in self.hero_cards),
            "board_cards": sorted(str(card) for card in self.board_cards),
            "villain_ranges": [
                {
                    "seat_id": item.seat_id,
                    "source": item.source,
                    "source_version": item.source_version,
                    "combo_weights": sorted(
                        _canonical_weights(item.combo_weights).items()
                    ),
                }
                for item in sorted(self.villain_ranges, key=lambda value: value.seat_id)
            ],
            "pots": [
                {
                    "pot_id": item.pot_id,
                    "amount": _canonical_decimal(item.amount.value),
                    "eligible_seats": sorted(item.eligible_seats),
                }
                for item in sorted(self.pots, key=lambda value: value.pot_id)
            ],
            "method": self.method.value,
            "engine_version": self.engine_version,
            "trials": self.trials,
            "seed": self.seed,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EquityCacheEntry:
    key: str
    result: MultiwayEquityResult
    created_at: datetime
    expires_at: datetime | None
    evidence: tuple[str, ...]
    confidence_low: Decimal | None = None
    confidence_high: Decimal | None = None
    confidence_level: Decimal | None = None
    numerical_confidence: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or len(self.key) != 64:
            raise ValueError("key must be a SHA-256 hex digest")
        if not isinstance(self.result, MultiwayEquityResult):
            raise TypeError("result must be a MultiwayEquityResult")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        _require_aware_dt(self.created_at)
        if self.expires_at is not None:
            if not isinstance(self.expires_at, datetime):
                raise TypeError("expires_at must be a datetime or None")
            _require_aware_dt(self.expires_at)
            if self.expires_at <= self.created_at:
                raise ValueError("expires_at must be after created_at")
        evidence = tuple(self.evidence)
        if not evidence or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError("evidence must contain non-empty references")
        object.__setattr__(self, "evidence", evidence)
        interval = (
            self.confidence_low,
            self.confidence_high,
            self.confidence_level,
            self.numerical_confidence,
        )
        if any(value is not None for value in interval):
            if not all(isinstance(value, Decimal) for value in interval):
                raise TypeError("confidence metadata must be all Decimal or all None")
            low, high, level, confidence = interval
            if not all(value.is_finite() for value in interval):
                raise ValueError("confidence metadata must be finite")
            if not Decimal("0") <= low <= high <= Decimal("1"):
                raise ValueError("confidence interval must be in [0, 1]")
            if not Decimal("0") < level < Decimal("1"):
                raise ValueError("confidence_level must be in (0, 1)")
            if not Decimal("0") <= confidence <= Decimal("1"):
                raise ValueError("numerical_confidence must be in [0, 1]")


@dataclass(frozen=True)
class EquityCacheLookup:
    state: EquityCacheState
    key: str
    entry: EquityCacheEntry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, EquityCacheState):
            raise TypeError("state must be an EquityCacheState")
        if not isinstance(self.key, str) or len(self.key) != 64:
            raise ValueError("key must be a SHA-256 hex digest")
        if self.state is EquityCacheState.HIT and self.entry is None:
            raise ValueError("HIT lookup requires an entry")
        if self.state is not EquityCacheState.HIT and self.entry is not None:
            raise ValueError("non-HIT lookup cannot expose an entry")


class EquityCache:
    """Bounded LRU cache with explicit stale results and no silent reuse."""

    def __init__(self, max_entries: int = 256):
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise TypeError("max_entries must be an int")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, EquityCacheEntry] = OrderedDict()
        self._lock = RLock()

    def put(
        self,
        query: EquityCacheQuery,
        result: MultiwayEquityResult,
        *,
        created_at: datetime,
        expires_at: datetime | None,
        evidence: tuple[str, ...],
        confidence_low: Decimal | None = None,
        confidence_high: Decimal | None = None,
        confidence_level: Decimal | None = None,
        numerical_confidence: Decimal | None = None,
    ) -> EquityCacheEntry:
        if not isinstance(query, EquityCacheQuery):
            raise TypeError("query must be an EquityCacheQuery")
        if not isinstance(result, MultiwayEquityResult):
            raise TypeError("result must be a MultiwayEquityResult")
        if result.hero_seat != query.hero_seat:
            raise ValueError("result hero_seat does not match query")
        query_total = sum((pot.amount.value for pot in query.pots), Decimal("0"))
        if result.total_pot.value != query_total:
            raise ValueError("result total_pot does not match query")
        entry = EquityCacheEntry(
            query.key,
            result,
            created_at,
            expires_at,
            evidence,
            confidence_low,
            confidence_high,
            confidence_level,
            numerical_confidence,
        )
        with self._lock:
            self._entries[entry.key] = entry
            self._entries.move_to_end(entry.key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return entry

    def get(
        self,
        query: EquityCacheQuery,
        *,
        now: datetime,
    ) -> EquityCacheLookup:
        if not isinstance(query, EquityCacheQuery):
            raise TypeError("query must be an EquityCacheQuery")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        _require_aware_dt(now)
        key = query.key
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return EquityCacheLookup(EquityCacheState.NOT_FOUND, key)
            if entry.expires_at is not None and now >= entry.expires_at:
                del self._entries[key]
                return EquityCacheLookup(EquityCacheState.STALE, key)
            self._entries.move_to_end(key)
            return EquityCacheLookup(EquityCacheState.HIT, key, entry)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _canonical_weights(weights) -> dict[str, str]:
    values = {combo: value for combo, value in weights.items() if value > 0}
    total = sum(values.values(), Decimal("0"))
    if total <= 0:
        return {}
    return {
        combo: _canonical_decimal(value / total)
        for combo, value in sorted(values.items())
    }


def _canonical_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")


__all__ = [
    "EquityCache",
    "EquityCacheEntry",
    "EquityCacheLookup",
    "EquityCacheQuery",
    "EquityCacheState",
    "EquityMethod",
]
