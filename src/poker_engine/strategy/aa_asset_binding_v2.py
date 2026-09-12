"""Versioned AA strategy-asset binding, separate from strategy node content."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .asset_provider import provider_capability_digest
from .provider import StrategyProvider


_ASSET_STATUS = {"test_only", "shadow_reviewed", "live_approved"}


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be SHA-256") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class AAStrategyAssetBindingV2:
    asset_sha256: str
    capability_sha256: str
    rule_fingerprint: str
    provider_id: str
    source_version: str
    player_counts: tuple[int, ...]
    streets: tuple[str, ...]
    asset_status: str
    source_url: str
    source_revision: str
    license_spdx: str
    limitations: tuple[str, ...]

    def __post_init__(self):
        for name in ("asset_sha256", "capability_sha256", "rule_fingerprint"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in (
            "provider_id", "source_version", "source_url", "source_revision",
            "license_spdx",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        counts = tuple(self.player_counts)
        if counts != tuple(sorted(set(counts))) or any(
            type(value) is not int or value not in (6, 7, 8) for value in counts
        ):
            raise ValueError("player_counts must be sorted unique values from 6,7,8")
        object.__setattr__(self, "player_counts", counts)
        streets = tuple(self.streets)
        if (streets != tuple(sorted(set(streets))) or not streets
                or any(value not in ("preflop", "flop", "turn", "river")
                       for value in streets)):
            raise ValueError("invalid sorted unique streets")
        object.__setattr__(self, "streets", streets)
        if self.asset_status not in _ASSET_STATUS:
            raise ValueError("invalid asset_status")
        limitations = tuple(self.limitations)
        if not limitations or not all(
            isinstance(value, str) and value for value in limitations
        ):
            raise ValueError("limitations must contain non-empty disclosures")
        object.__setattr__(self, "limitations", limitations)

    @classmethod
    def from_dict(cls, value):
        required = {
            "schema_version", "asset_sha256", "capability_sha256",
            "rule_fingerprint", "provider_id", "source_version", "player_counts",
            "streets", "asset_status", "source_url", "source_revision",
            "license_spdx", "limitations",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("AA asset binding requires exact schema fields")
        if value["schema_version"] != 2:
            raise ValueError("unsupported AA asset binding schema")
        return cls(
            value["asset_sha256"], value["capability_sha256"],
            value["rule_fingerprint"], value["provider_id"],
            value["source_version"], tuple(value["player_counts"]),
            tuple(value["streets"]), value["asset_status"], value["source_url"],
            value["source_revision"], value["license_spdx"],
            tuple(value["limitations"]),
        )

    def to_dict(self):
        return {
            "schema_version": 2,
            "asset_sha256": self.asset_sha256,
            "capability_sha256": self.capability_sha256,
            "rule_fingerprint": self.rule_fingerprint,
            "provider_id": self.provider_id,
            "source_version": self.source_version,
            "player_counts": list(self.player_counts),
            "streets": list(self.streets),
            "asset_status": self.asset_status,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "license_spdx": self.license_spdx,
            "limitations": list(self.limitations),
        }

    @property
    def binding_fingerprint(self):
        raw = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @property
    def shadow_permitted(self):
        return self.asset_status in ("shadow_reviewed", "live_approved")

    @property
    def live_permitted(self):
        return self.asset_status == "live_approved"

    def validate_provider(self, provider):
        if not isinstance(provider, StrategyProvider):
            raise TypeError("provider must implement StrategyProvider")
        reasons = []
        if provider.provider_id != self.provider_id:
            reasons.append("binding_provider_id_mismatch")
        if provider.source_version != self.source_version:
            reasons.append("binding_source_version_mismatch")
        if getattr(provider, "asset_sha256", None) != self.asset_sha256:
            reasons.append("binding_asset_sha256_mismatch")
        if provider_capability_digest(provider.capability) != self.capability_sha256:
            reasons.append("binding_capability_sha256_mismatch")
        if set(provider.capability.player_counts) != set(self.player_counts):
            reasons.append("binding_player_counts_mismatch")
        provider_streets = {street.value for street in provider.capability.streets}
        if provider_streets != set(self.streets):
            reasons.append("binding_streets_mismatch")
        return tuple(reasons)


__all__ = ["AAStrategyAssetBindingV2"]
