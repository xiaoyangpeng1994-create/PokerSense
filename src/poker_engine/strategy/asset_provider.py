"""Hash-pinned JSON adapter for licensed precomputed strategy assets."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from .contracts import DecisionContext
from .provider import (
    ActionOption,
    LookupState,
    MatchDimension,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)
from .strategy_cache import canonical_context_digest


STRATEGY_ASSET_SCHEMA_VERSION = 1


class JsonStrategyAssetProvider:
    """Read exact or disclosed approximate nodes from one immutable asset."""

    def __init__(
        self,
        asset_path: str | Path,
        *,
        expected_sha256: str,
        capability: ProviderCapability,
        capability_id: str,
    ) -> None:
        path = Path(asset_path)
        if not isinstance(expected_sha256, str) or (
            len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            raise ValueError("expected_sha256 must be lowercase SHA-256 hex")
        if not isinstance(capability, ProviderCapability):
            raise TypeError("capability must be ProviderCapability")
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("capability_id must be a non-empty str")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise ValueError("strategy asset SHA-256 mismatch")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("strategy asset must be valid UTF-8 JSON") from exc
        metadata = _validate_asset(
            payload, capability_id, provider_capability_digest(capability)
        )
        self._path = path
        self._asset_sha256 = digest
        self._payload = payload
        self._metadata = metadata
        self._capability = capability

    @property
    def provider_id(self) -> str:
        return self._metadata["provider_id"]

    @property
    def source_version(self) -> str:
        return self._metadata["source_version"]

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    @property
    def asset_sha256(self) -> str:
        return self._asset_sha256

    def query(self, context: DecisionContext) -> ProviderResult:
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be DecisionContext")
        match = self.capability.match(context)
        if not match.applicable:
            return ProviderResult(
                LookupState.NOT_APPLICABLE,
                self.provider_id,
                reasons=match.reasons,
            )
        key = canonical_context_digest(context)
        node = self._payload["nodes"].get(key)
        if node is None:
            return ProviderResult(
                LookupState.NOT_FOUND,
                self.provider_id,
                reasons=("strategy_asset_node_not_found",),
            )
        try:
            candidate = self._candidate(context, key, node)
        except (KeyError, TypeError, ValueError) as exc:
            return ProviderResult(
                LookupState.REJECTED,
                self.provider_id,
                reasons=(f"invalid_strategy_asset_node:{type(exc).__name__}",),
            )
        state = (
            LookupState.HIT_EXACT
            if candidate.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE
        )
        return ProviderResult(state, self.provider_id, candidate)

    def _candidate(
        self,
        context: DecisionContext,
        key: str,
        node: Any,
    ) -> StrategyCandidate:
        if not isinstance(node, Mapping):
            raise TypeError("node must be an object")
        probabilities = {
            ActionType(action): Decimal(value)
            for action, value in _mapping(
                node["action_probabilities"], "action_probabilities"
            ).items()
        }
        sizes = {
            ActionType(action): tuple(ChipAmount(value) for value in values)
            for action, values in _mapping(
                node.get("recommended_sizes", {}), "recommended_sizes"
            ).items()
        }
        action_ev = {
            ActionType(action): ChipDelta(value)
            for action, value in _mapping(
                node.get("action_ev", {}), "action_ev"
            ).items()
        }
        options = tuple(
            ActionOption(
                ActionType(item["action"]),
                Decimal(item["probability"]),
                ChipAmount(item["amount"])
                if item.get("amount") is not None else None,
                item.get("source_label"),
            )
            for item in _array(node.get("action_options", []), "action_options")
        )
        dimensions = tuple(
            MatchDimension(
                item["name"],
                item["requested"],
                item["matched"],
                Decimal(item["distance"]),
                Decimal(item["maximum_distance"]),
            )
            for item in _array(
                node.get("match_dimensions", []), "match_dimensions"
            )
        )
        assumptions = tuple(self._payload.get("limitations", [])) + tuple(
            _array(node.get("assumptions", []), "assumptions")
        )
        return StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            match_kind=MatchKind(node["match_kind"]),
            state_match_score=float(node["state_match_score"]),
            match_dimensions=dimensions,
            action_probabilities=probabilities,
            recommended_sizes=sizes,
            action_options=options,
            action_ev=action_ev,
            confidence=float(node["confidence"]),
            evidence=(
                f"{self._metadata['source_url']}/tree/"
                f"{self._metadata['source_revision']}",
                f"strategy_asset_sha256:{self._asset_sha256}",
                f"strategy_asset_node:{key}",
                f"strategy_asset_license:{self._metadata['license_spdx']}",
            ),
            assumptions=assumptions,
            expires_at=context.request.expires_at,
        )


def provider_capability_digest(capability: ProviderCapability) -> str:
    if not isinstance(capability, ProviderCapability):
        raise TypeError("capability must be ProviderCapability")
    payload = {
        "player_counts": sorted(capability.player_counts),
        "streets": sorted(value.value for value in capability.streets),
        "game_types": sorted(value.value for value in capability.game_types),
        "stack_buckets_bb": [str(value) for value in capability.stack_buckets_bb],
        "ante_values": [str(value.value) for value in capability.ante_values],
        "rake_percent_values": [
            str(value) for value in capability.rake_percent_values
        ],
        "action_lines": sorted(capability.action_lines),
        "base_match_kind": capability.base_match_kind.value,
        "allow_stack_interpolation": capability.allow_stack_interpolation,
        "max_stack_distance_bb": str(capability.max_stack_distance_bb),
        "ante_values_are_bb": capability.ante_values_are_bb,
        "stack_ante_pairs_bb": [
            [str(stack), str(ante)]
            for stack, ante in capability.stack_ante_pairs_bb
        ],
        "hero_positions": sorted(value.value for value in capability.hero_positions),
        "pot_buckets_bb": [str(value) for value in capability.pot_buckets_bb],
        "allow_pot_interpolation": capability.allow_pot_interpolation,
        "max_pot_distance_bb": str(capability.max_pot_distance_bb),
        "aggressive_size_buckets_bb": [
            str(value) for value in capability.aggressive_size_buckets_bb
        ],
        "allow_aggressive_size_interpolation": (
            capability.allow_aggressive_size_interpolation
        ),
        "max_aggressive_size_distance_bb": str(
            capability.max_aggressive_size_distance_bb
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_asset(
    payload: Any,
    capability_id: str,
    capability_sha256: str,
) -> Mapping[str, str]:
    if not isinstance(payload, Mapping):
        raise TypeError("strategy asset must be an object")
    if payload.get("schema_version") != STRATEGY_ASSET_SCHEMA_VERSION:
        raise ValueError("unsupported strategy asset schema_version")
    if payload.get("capability_id") != capability_id:
        raise ValueError("strategy asset capability_id mismatch")
    if payload.get("capability_sha256") != capability_sha256:
        raise ValueError("strategy asset capability SHA-256 mismatch")
    metadata = payload.get("provider")
    if not isinstance(metadata, Mapping):
        raise TypeError("provider metadata must be an object")
    required = (
        "provider_id",
        "source_version",
        "source_url",
        "source_revision",
        "license_spdx",
    )
    if any(
        not isinstance(metadata.get(name), str) or not metadata[name]
        for name in required
    ):
        raise ValueError("provider metadata fields must be non-empty strings")
    if not isinstance(payload.get("nodes"), Mapping):
        raise TypeError("nodes must be an object")
    limitations = payload.get("limitations", [])
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise TypeError("limitations must contain non-empty strings")
    return {name: metadata[name] for name in required}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


__all__ = [
    "JsonStrategyAssetProvider",
    "STRATEGY_ASSET_SCHEMA_VERSION",
    "provider_capability_digest",
]
