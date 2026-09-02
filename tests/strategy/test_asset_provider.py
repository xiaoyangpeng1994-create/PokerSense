"""Hash, schema, capability, and strategy-node tests for JSON assets."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import ActionType, Street
from poker_engine.strategy.advice import AdviceStatus, build_advice
from poker_engine.strategy.asset_provider import (
    JsonStrategyAssetProvider,
    provider_capability_digest,
)
from poker_engine.strategy.provider import LookupState, MatchKind
from poker_engine.strategy.router import StrategyRouter
from poker_engine.strategy.strategy_cache import canonical_context_digest

from .helpers import NOW, capability, context


FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _node(*, match_kind="exact", score=1.0, dimensions=None):
    return {
        "match_kind": match_kind,
        "state_match_score": score,
        "match_dimensions": dimensions or [],
        "action_probabilities": {"check": "0.4", "raise": "0.6"},
        "recommended_sizes": {"raise": ["2.5"]},
        "action_options": [
            {"action": "check", "probability": "0.4", "amount": None,
             "source_label": "check"},
            {"action": "raise", "probability": "0.6", "amount": "2.5",
             "source_label": "raise_2.5"},
        ],
        "action_ev": {"check": "0", "raise": "1.2"},
        "confidence": 0.8,
        "assumptions": ["synthetic_test_node"],
    }


def _write_asset(tmp_path, ctx, cap, *, node=None, mutate=None):
    payload = {
        "schema_version": 1,
        "capability_id": "test-cap-v1",
        "capability_sha256": provider_capability_digest(cap),
        "provider": {
            "provider_id": "licensed-asset",
            "source_version": "asset-v7",
            "source_url": "https://example.invalid/licensed-asset",
            "source_revision": "revision-7",
            "license_spdx": "LicenseRef-Commercial-Test",
        },
        "limitations": ["synthetic_test_asset_not_release_evidence"],
        "nodes": {
            canonical_context_digest(ctx): node or _node(),
        },
    }
    if mutate is not None:
        mutate(payload)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    path = tmp_path / "strategy.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), payload


def _provider(tmp_path, ctx, cap, **kwargs):
    path, digest, _ = _write_asset(tmp_path, ctx, cap, **kwargs)
    return JsonStrategyAssetProvider(
        path,
        expected_sha256=digest,
        capability=cap,
        capability_id="test-cap-v1",
    )


@pytest.mark.parametrize(
    ("ctx", "cap"),
    (
        (context(3), capability((3,))),
        (context(6, street=Street.FLOP, active_count=3),
         capability((3,), streets=(Street.FLOP,))),
        (context(9, street=Street.TURN, active_count=4),
         capability((4,), streets=(Street.TURN,))),
    ),
)
def test_asset_adapter_supports_explicit_multiplayer_scenarios(
    tmp_path, ctx, cap,
):
    provider = _provider(tmp_path, ctx, cap)

    result = provider.query(ctx)

    assert result.state is LookupState.HIT_EXACT
    assert result.candidate.match_kind is MatchKind.EXACT
    assert result.candidate.action_probabilities[ActionType.RAISE] == Decimal("0.6")
    assert result.candidate.action_options[1].amount.value == Decimal("2.5")
    assert result.candidate.provider_version == "asset-v7"
    assert f"strategy_asset_sha256:{provider.asset_sha256}" in (
        result.candidate.evidence
    )


def test_asset_candidate_routes_and_builds_ready_advice(tmp_path):
    ctx = context(3)
    provider = _provider(tmp_path, ctx, capability((3,)))

    route = StrategyRouter((provider,)).route(ctx, now=NOW)
    advice = build_advice(ctx, route, now=NOW)

    assert advice.status is AdviceStatus.READY
    assert advice.strategy_source == "licensed-asset"
    assert advice.strategy_version == "asset-v7"
    assert "synthetic_test_asset_not_release_evidence" in advice.assumptions


def test_missing_context_node_is_not_found(tmp_path):
    ctx = context(3)
    provider = _provider(tmp_path, ctx, capability((3,)))

    result = provider.query(context(3, action_line="raise"))

    assert result.state is LookupState.NOT_FOUND
    assert result.reasons == ("strategy_asset_node_not_found",)


def test_capability_mismatch_is_not_applicable_before_lookup(tmp_path):
    ctx = context(3)
    provider = _provider(tmp_path, ctx, capability((3,)))

    result = provider.query(context(4))

    assert result.state is LookupState.NOT_APPLICABLE
    assert result.reasons == ("unsupported_player_count",)


@pytest.mark.parametrize(
    "node",
    (
        {**_node(), "action_probabilities": {"check": "0.9", "raise": "0.9"}},
        _node(match_kind="interpolated", score=0.5),
        {**_node(), "confidence": "not-a-number"},
    ),
)
def test_malformed_or_unexplained_node_is_rejected(tmp_path, node):
    ctx = context(3)
    provider = _provider(tmp_path, ctx, capability((3,)), node=node)

    result = provider.query(ctx)

    assert result.state is LookupState.REJECTED
    assert result.reasons[0].startswith("invalid_strategy_asset_node:")


def test_hash_mismatch_fails_before_asset_can_register(tmp_path):
    ctx = context(3)
    cap = capability((3,))
    path, _, _ = _write_asset(tmp_path, ctx, cap)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        JsonStrategyAssetProvider(
            path,
            expected_sha256="0" * 64,
            capability=cap,
            capability_id="test-cap-v1",
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(schema_version=99),
        lambda payload: payload.update(capability_id="wrong"),
        lambda payload: payload.update(capability_sha256="0" * 64),
        lambda payload: payload["provider"].update(license_spdx=""),
        lambda payload: payload.update(nodes=[]),
    ),
)
def test_invalid_manifest_metadata_fails_at_registration(tmp_path, mutate):
    ctx = context(3)
    cap = capability((3,))
    path, digest, _ = _write_asset(tmp_path, ctx, cap, mutate=mutate)

    with pytest.raises((TypeError, ValueError)):
        JsonStrategyAssetProvider(
            path,
            expected_sha256=digest,
            capability=cap,
            capability_id="test-cap-v1",
        )


@pytest.mark.parametrize(
    "fixture",
    [
        item for item in (
            json.loads(line) for line in FIXTURES.read_text().splitlines()
        )
        if item["fixture_id"].startswith("MOCK-STRATEGY-ASSET-")
    ],
)
def test_generated_strategy_asset_fixtures_execute_through_adapter(
    tmp_path, fixture,
):
    expected = fixture["expected"]["strategy_asset_adapter"]
    street = Street(expected["street"])
    dealt = fixture["input"]["state"]["dealt_player_count"]
    active = expected["player_count"]
    ctx = context(dealt, street=street, active_count=active)
    cap = capability((active,), streets=(street,))
    asset_state = fixture["input"]["strategy_asset"]["asset_state"]
    node = (
        {**_node(), "action_probabilities": {"check": "0.9", "raise": "0.9"}}
        if asset_state == "malformed" else None
    )
    mutate = (
        (lambda payload: payload.update(nodes={}))
        if asset_state == "missing" else None
    )
    path, digest, _ = _write_asset(
        tmp_path, ctx, cap, node=node, mutate=mutate
    )
    if asset_state == "bad_hash":
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            JsonStrategyAssetProvider(
                path,
                expected_sha256="0" * 64,
                capability=cap,
                capability_id="test-cap-v1",
            )
        assert expected["outcome"] == "REGISTRATION_ERROR"
        return
    provider = JsonStrategyAssetProvider(
        path,
        expected_sha256=digest,
        capability=cap,
        capability_id="test-cap-v1",
    )

    result = provider.query(ctx)

    assert result.state.value == expected["outcome"]
    if result.candidate is not None:
        assert any(
            item.startswith("strategy_asset_sha256:")
            for item in result.candidate.evidence
        )
        assert any(
            item.startswith("strategy_asset_license:")
            for item in result.candidate.evidence
        )
