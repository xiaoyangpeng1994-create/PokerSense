"""Transparent multiplayer preflop heuristic Provider.

The bundled ranges are not a GTO solution.  This adapter intentionally exposes
only the upstream source's explicit 6-handed and 9-handed open-raise lists and
returns a HEURISTIC candidate with no invented bet size or EV.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from poker_engine.core.enums import ActionType, Position, Street
from poker_engine.core.value_objects import ChipAmount

from .blueprint_provider import hand_class
from .contracts import DecisionContext, GameType
from .provider import (
    ActionOption,
    LookupState,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)


PROVIDER_ID = "preflopr-explicit-rfi-heuristic"
ASSET_NAME = "preflopr-explicit-rfi-ranges.json"
_HAND_PATTERN = re.compile(r"(?:[AKQJT2-9]{2}|[AKQJT2-9]{2}[so])\Z")
_POSITION_KEYS = {
    6: {
        Position.UTG: "UTG",
        Position.HJ: "HJ",
        Position.CO: "CO",
        Position.BTN: "BTN",
        Position.SB: "SB",
    },
    9: {
        Position.UTG: "UTG",
        Position.UTG1: "UTG+1",
        Position.UTG2: "UTG+2",
        Position.LJ: "MP",
        Position.HJ: "HJ",
        Position.CO: "CO",
        Position.BTN: "BTN",
        Position.SB: "SB",
    },
}


class PreflopRfiHeuristicProvider:
    """Binary raise/fold fallback for reviewed 6-max and 9-max RFI spots."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        asset_sha256: str,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if not isinstance(asset_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", asset_sha256
        ):
            raise ValueError("asset_sha256 must be a 64-character hex digest")
        self._payload = dict(payload)
        self._asset_sha256 = asset_sha256.lower()
        self._ranges = self._validate_asset(self._payload)
        self._capability = ProviderCapability(
            player_counts=frozenset({6, 9}),
            streets=frozenset({Street.PREFLOP}),
            game_types=frozenset({GameType.CASH}),
            stack_buckets_bb=(Decimal("100"),),
            ante_values=(ChipAmount("0"),),
            rake_percent_values=(Decimal("0"),),
            action_lines=frozenset({"unopened"}),
            base_match_kind=MatchKind.HEURISTIC,
            allow_stack_interpolation=False,
            priority=200,
            hero_positions=frozenset(
                position
                for positions in _POSITION_KEYS.values()
                for position in positions
            ),
        )

    @classmethod
    def from_asset_file(cls, path: str | Path) -> "PreflopRfiHeuristicProvider":
        asset_bytes = Path(path).read_bytes()
        payload = json.loads(asset_bytes)
        return cls(
            payload,
            asset_sha256=hashlib.sha256(asset_bytes).hexdigest(),
        )

    @classmethod
    def from_builtin(cls) -> "PreflopRfiHeuristicProvider":
        asset = resources.files("poker_engine.strategy.assets").joinpath(ASSET_NAME)
        asset_bytes = asset.read_bytes()
        payload = json.loads(asset_bytes)
        return cls(
            payload,
            asset_sha256=hashlib.sha256(asset_bytes).hexdigest(),
        )

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    @property
    def source_version(self) -> str:
        return (
            f"preflopR/{self._payload['source_revision']}"
            f":asset/{self._payload['asset_version']}@{self._asset_sha256}"
        )

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def asset_sha256(self) -> str:
        return self._asset_sha256

    @property
    def source_url(self) -> str:
        return str(self._payload["source_url"])

    @property
    def source_revision(self) -> str:
        return str(self._payload["source_revision"])

    def explicit_range(
        self,
        player_count: int,
        position: Position,
    ) -> frozenset[str] | None:
        """Return only an explicitly imported range; never run a fallback."""
        positions = _POSITION_KEYS.get(player_count)
        if positions is None or position not in positions:
            return None
        return self._ranges[f"{player_count}_{positions[position]}"]

    def query(self, context: DecisionContext) -> ProviderResult:
        match = self.capability.match(context)
        if not match.applicable:
            reasons = match.reasons
            if reasons == ("unsupported_hero_position",):
                reasons = ("unsupported_open_raise_position",)
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=reasons,
            )
        reasons = self._validate_context(context)
        if reasons:
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=reasons,
            )
        hero = next(seat for seat in context.seats if seat.seat_id == context.hero_seat)
        source_position = _POSITION_KEYS[context.strategy_player_count][hero.position]
        range_key = f"{context.strategy_player_count}_{source_position}"
        hero_class = hand_class(context.hero_cards)
        action = (
            ActionType.RAISE
            if hero_class in self._ranges[range_key]
            else ActionType.FOLD
        )
        probabilities = {
            ActionType.FOLD: (
                Decimal("1") if action is ActionType.FOLD else Decimal("0")
            ),
            ActionType.RAISE: (
                Decimal("1") if action is ActionType.RAISE else Decimal("0")
            ),
        }
        options = tuple(
            ActionOption(candidate, probability, source_label=candidate.value)
            for candidate, probability in probabilities.items()
        )
        candidate = StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            match_kind=MatchKind.HEURISTIC,
            state_match_score=1.0,
            action_probabilities=probabilities,
            action_options=options,
            confidence=0.4,
            evidence=(
                f"{self._payload['source_url']}/tree/"
                f"{self._payload['source_revision']}",
                f"source_sha256:{self._payload['source_sha256']}",
                f"asset_sha256:{self._asset_sha256}",
                f"explicit_range:{range_key}:{hero_class}",
            ),
            assumptions=tuple(self._payload["limitations"]),
            expires_at=context.request.expires_at,
        )
        return ProviderResult(
            LookupState.HIT_APPROXIMATE,
            self.provider_id,
            candidate,
        )

    def _validate_context(self, context: DecisionContext) -> tuple[str, ...]:
        reasons = []
        if context.game_config.variant.upper() != "NLHE":
            reasons.append("unsupported_variant")
        if len(context.hero_cards) != 2:
            reasons.append("missing_hero_cards")
        if context.actor_seat != context.hero_seat:
            reasons.append("hero_not_actor")
        hero = next(
            (seat for seat in context.seats if seat.seat_id == context.hero_seat),
            None,
        )
        positions = _POSITION_KEYS.get(context.strategy_player_count, {})
        if hero is None or hero.position not in positions:
            reasons.append("unsupported_open_raise_position")
        required = {ActionType.FOLD, ActionType.RAISE}
        if not required <= context.legal_action_types:
            reasons.append("legal_actions_do_not_support_raise_fold")
        return tuple(reasons)

    @staticmethod
    def _validate_asset(payload: Mapping[str, Any]) -> dict[str, frozenset[str]]:
        required_metadata = {
            "schema_version": 1,
            "asset_id": "preflopr-explicit-rfi-ranges",
            "asset_version": "1",
            "license": "MIT",
        }
        for field, expected in required_metadata.items():
            if payload.get(field) != expected:
                raise ValueError(f"invalid asset {field}")
        for field in ("source_url", "source_revision", "source_sha256"):
            value = payload.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"invalid asset {field}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", payload["source_sha256"]):
            raise ValueError("invalid asset source_sha256")
        limitations = payload.get("limitations")
        if not isinstance(limitations, list) or not limitations or not all(
            isinstance(value, str) and value for value in limitations
        ):
            raise ValueError("invalid asset limitations")
        expected_keys = {
            f"{players}_{position}"
            for players, positions in _POSITION_KEYS.items()
            for position in positions.values()
        }
        raw_ranges = payload.get("ranges")
        if not isinstance(raw_ranges, Mapping) or set(raw_ranges) != expected_keys:
            raise ValueError("asset range keys do not match reviewed capability")
        ranges: dict[str, frozenset[str]] = {}
        for key, raw_hands in raw_ranges.items():
            if not isinstance(raw_hands, list) or not raw_hands:
                raise ValueError(f"invalid asset range {key}")
            if len(raw_hands) != len(set(raw_hands)) or not all(
                isinstance(hand, str) and _HAND_PATTERN.fullmatch(hand)
                for hand in raw_hands
            ):
                raise ValueError(f"invalid hand list for asset range {key}")
            ranges[key] = frozenset(raw_hands)
        return ranges


__all__ = ["ASSET_NAME", "PROVIDER_ID", "PreflopRfiHeuristicProvider"]
