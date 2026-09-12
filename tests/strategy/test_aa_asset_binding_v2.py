import json
from pathlib import Path

import pytest

from poker_engine.strategy.aa_asset_binding_v2 import AAStrategyAssetBindingV2
from poker_engine.strategy.asset_provider import provider_capability_digest

from .helpers import capability


ROOT = Path(__file__).resolve().parents[2]


class Provider:
    provider_id = "aa-test"
    source_version = "v1"
    asset_sha256 = "a" * 64
    capability = capability((6, 7, 8))

    def query(self, context):
        raise AssertionError("binding validation must not query provider")


def payload(**changes):
    provider = Provider()
    value = {
        "schema_version": 2,
        "asset_sha256": provider.asset_sha256,
        "capability_sha256": provider_capability_digest(provider.capability),
        "rule_fingerprint": "b" * 64,
        "provider_id": provider.provider_id,
        "source_version": provider.source_version,
        "player_counts": [6, 7, 8],
        "streets": ["preflop"],
        "asset_status": "shadow_reviewed",
        "source_url": "https://example.invalid/aa-asset",
        "source_revision": "test-revision",
        "license_spdx": "LicenseRef-Test-Only",
        "limitations": ["synthetic test binding, not poker strategy"],
    }
    value.update(changes)
    return value


def test_binding_hashes_provider_capability_rules_and_source():
    binding = AAStrategyAssetBindingV2.from_dict(payload())
    assert binding.validate_provider(Provider()) == ()
    assert len(binding.binding_fingerprint) == 64
    assert binding.shadow_permitted
    assert not binding.live_permitted
    assert binding.to_dict() == payload()


@pytest.mark.parametrize(("field", "value", "reason"), (
    ("provider_id", "wrong", "binding_provider_id_mismatch"),
    ("source_version", "v2", "binding_source_version_mismatch"),
    ("asset_sha256", "c" * 64, "binding_asset_sha256_mismatch"),
    ("capability_sha256", "d" * 64, "binding_capability_sha256_mismatch"),
))
def test_provider_mismatch_is_explicit(field, value, reason):
    binding = AAStrategyAssetBindingV2.from_dict(payload(**{field: value}))
    assert reason in binding.validate_provider(Provider())


def test_status_separates_test_shadow_and_live():
    test = AAStrategyAssetBindingV2.from_dict(payload(asset_status="test_only"))
    live = AAStrategyAssetBindingV2.from_dict(payload(asset_status="live_approved"))
    assert not test.shadow_permitted and not test.live_permitted
    assert live.shadow_permitted and live.live_permitted
    assert test.binding_fingerprint != live.binding_fingerprint


@pytest.mark.parametrize("changes", (
    {"schema_version": 1}, {"asset_sha256": "A" * 64},
    {"player_counts": [8, 7]}, {"player_counts": [5]},
    {"streets": ["river", "flop"]}, {"streets": ["showdown"]},
    {"limitations": []}, {"extra": True},
))
def test_invalid_or_ambiguous_bindings_rejected(changes):
    with pytest.raises(ValueError):
        AAStrategyAssetBindingV2.from_dict(payload(**changes))


def test_checked_in_schema_is_strict_and_covers_runtime_fields():
    schema = json.loads((
        ROOT / "configs/strategy/aa-asset-binding-v2.schema.json"
    ).read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload())
    assert schema["properties"]["asset_status"]["enum"] == [
        "test_only", "shadow_reviewed", "live_approved",
    ]
