import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import Position, Rank, Suit
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.strategy.aa_range_assets_v2 import (
    AAConcreteRangeAssetV2,
    AARangeLookupState,
    AARangeQueryV2,
    AARangeShadowTrackerV2,
    assess_aa_range_readiness,
)
from poker_engine.strategy.contracts import PotState


RULES = "b" * 64
ROOT = Path(__file__).resolve().parents[2]


def payload(**changes):
    value = {
        "schema_version": 2,
        "asset_id": "aa-range-test",
        "asset_version": "1",
        "asset_status": "shadow_reviewed",
        "rule_fingerprint": RULES,
        "source": {
            "url": "https://example.invalid/ranges",
            "revision": "test",
            "license_spdx": "LicenseRef-Test",
        },
        "limitations": ["synthetic fixture, not a poker range"],
        "nodes": [{
            "node_id": "6-utg-unopened-100",
            "player_count": 6,
            "position": "UTG",
            "stack_bb": "100",
            "action_line": "unopened",
            "prior": [
                {"combo": "AcKc", "weight": "0.25"},
                {"combo": "AdKd", "weight": "0.25"},
                {"combo": "AhKh", "weight": "0.25"},
                {"combo": "AsKs", "weight": "0.25"},
            ],
            "action_likelihoods": {
                "call": [
                    {"combo": "AcKc", "likelihood": "0.1"},
                    {"combo": "AdKd", "likelihood": "0.2"},
                ],
                "aggressive": [
                    {"combo": "AcKc", "likelihood": "0.8"},
                    {"combo": "AdKd", "likelihood": "0.7"},
                    {"combo": "AhKh", "likelihood": "0.6"},
                    {"combo": "AsKs", "likelihood": "0.5"},
                ],
            },
            "confidence": 0.8,
            "effective_sample_size": 100,
            "evidence": ["synthetic://range-test"],
        }],
    }
    value.update(changes)
    return value


def write_asset(tmp_path, value=None):
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(value or payload()), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def query(seat=1, *, known=(), count=6, action_line="unopened"):
    return AARangeQueryV2(
        seat, count, Position.UTG, Decimal("100"), action_line, known,
    )


def test_exact_node_loads_concrete_range_and_filters_blockers(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    result = asset.lookup(query())
    assert result.state is AARangeLookupState.HIT
    assert len(result.distribution.combo_weights) == 4
    assert result.distribution.confidence == .8
    assert result.distribution.source_version.startswith(f"aa-ranges-v2:{RULES}:")
    blocked = asset.lookup(query(known=(Card(Rank.ACE, Suit.CLUBS),)))
    assert len(blocked.distribution.combo_weights) == 3
    assert sum(blocked.distribution.combo_weights.values()) == 1


def test_partial_action_likelihood_updates_and_degrades_confidence(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    lookup = asset.lookup(query())
    updated = asset.update(lookup.distribution, query(), "call")
    assert updated.applied
    assert updated.likelihood_coverage == Decimal("0.50")
    assert updated.distribution.confidence == pytest.approx(.4)
    assert updated.missing_likelihood_combos == ("AhKh", "AsKs")
    assert sum(updated.distribution.combo_weights.values()) == 1


def test_missing_action_likelihood_preserves_prior_but_is_not_applied(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    prior = asset.lookup(query()).distribution
    result = asset.update(prior, query(), "fold")
    assert not result.applied and result.distribution is prior
    assert result.likelihood_coverage == 0
    with pytest.raises(ValueError, match="not bound"):
        asset.update(
            replace(prior, seat_id=2), query(1), "call"
        )


def test_range_quality_requires_exact_pot_seats_rules_and_confidence(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    first = asset.lookup(query(1)).distribution
    second = asset.lookup(query(2)).distribution
    pots = (PotState("main", ChipAmount("10"), (0, 1, 2)),)
    ready = assess_aa_range_readiness(
        (first, second), hero_seat=0, pots=pots, rule_fingerprint=RULES
    )
    assert ready.permitted
    assert ready.required_seats == ready.supplied_seats == (1, 2)
    wrong = assess_aa_range_readiness(
        (first,), hero_seat=0, pots=pots, rule_fingerprint="c" * 64
    )
    assert not wrong.permitted
    assert "villain_range_seats_do_not_match_pot_eligibility" in wrong.reasons
    assert "villain_range_rule_fingerprint_mismatch" in wrong.reasons
    low = assess_aa_range_readiness(
        (first, second), hero_seat=0, pots=pots, rule_fingerprint=RULES,
        minimum_confidence=.9,
    )
    assert "villain_range_confidence_below_threshold" in low.reasons


def test_test_only_asset_and_missing_exact_node_do_not_fallback(tmp_path):
    value = payload(asset_status="test_only")
    path, digest = write_asset(tmp_path, value)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    assert asset.lookup(query()).state is AARangeLookupState.NOT_APPLICABLE
    value["asset_status"] = "shadow_reviewed"
    path.write_text(json.dumps(value), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    missing = asset.lookup(query(count=7))
    assert missing.state is AARangeLookupState.NOT_APPLICABLE
    assert missing.reasons == ("exact_range_node_not_found",)


def test_asset_hash_and_rule_fingerprint_are_pinned(tmp_path):
    path, digest = write_asset(tmp_path)
    with pytest.raises(ValueError, match="asset SHA"):
        AAConcreteRangeAssetV2(
            path, expected_sha256="a" * 64, expected_rule_fingerprint=RULES
        )
    with pytest.raises(ValueError, match="rule fingerprint mismatch"):
        AAConcreteRangeAssetV2(
            path, expected_sha256=digest, expected_rule_fingerprint="c" * 64
        )


def test_duplicate_json_keys_are_rejected_before_asset_use(tmp_path):
    original = json.dumps(payload())
    raw = original.replace(
        '"asset_id": "aa-range-test",',
        '"asset_id": "first", "asset_id": "aa-range-test",',
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        AAConcreteRangeAssetV2(
            path, expected_sha256=digest, expected_rule_fingerprint=RULES
        )


@pytest.mark.parametrize("mutation", (
    "reversed_combo", "bad_sum", "unknown_likelihood", "bad_action",
    "duplicate_node", "bad_count", "unsorted_combo",
))
def test_invalid_range_assets_rejected(tmp_path, mutation):
    value = payload()
    node = value["nodes"][0]
    if mutation == "reversed_combo":
        node["prior"][-1] = {"combo": "KcAc", "weight": "0.25"}
    elif mutation == "bad_sum":
        node["prior"][-1]["weight"] = "0.2"
    elif mutation == "unknown_likelihood":
        node["action_likelihoods"]["call"].append(
            {"combo": "QcQd", "likelihood": "0.5"}
        )
    elif mutation == "bad_action":
        node["action_likelihoods"]["raise"] = []
    elif mutation == "duplicate_node":
        value["nodes"].append(dict(node))
    elif mutation == "bad_count":
        node["player_count"] = 5
    else:
        node["prior"][0], node["prior"][1] = node["prior"][1], node["prior"][0]
    path, digest = write_asset(tmp_path, value)
    with pytest.raises((ValueError, TypeError)):
        AAConcreteRangeAssetV2(
            path, expected_sha256=digest, expected_rule_fingerprint=RULES
        )


@pytest.mark.parametrize("changes", (
    {"seat": 8}, {"count": 5}, {"position": Position.UNKNOWN},
    {"stack": Decimal("NaN")}, {"action_line": ""},
))
def test_invalid_queries_rejected(changes):
    values = {
        "seat_id": 1, "player_count": 6, "position": Position.UTG,
        "stack_bb": Decimal("100"), "action_line": "unopened",
    }
    mapping = {
        "seat": "seat_id", "count": "player_count", "position": "position",
        "stack": "stack_bb", "action_line": "action_line",
    }
    key, value = next(iter(changes.items()))
    values[mapping[key]] = value
    with pytest.raises(ValueError):
        AARangeQueryV2(**values)


def test_checked_in_schema_rejects_extra_fields_and_covers_runtime_asset():
    schema = json.loads((
        ROOT / "configs/strategy/aa-range-asset-v2.schema.json"
    ).read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(payload())
    node = schema["properties"]["nodes"]["items"]
    assert node["additionalProperties"] is False
    assert set(node["required"]) == set(payload()["nodes"][0])


def test_shadow_tracker_seeds_all_pot_opponents_and_full_update_stays_ready(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    tracker = AARangeShadowTrackerV2(asset)
    assert tracker.seed((query(1), query(2))) == ()
    event = tracker.observe_action(10, 1, "aggressive")
    assert event["applied"] and event["likelihood_coverage"] == "1.00"
    pots = (PotState("main", ChipAmount("10"), (0, 1, 2)),)
    snapshot = tracker.snapshot(hero_seat=0, pots=pots)
    assert snapshot.equity_permitted
    assert snapshot.version == 2
    assert not snapshot.strategy_eligible and not snapshot.advice_emitted


def test_partial_or_missing_likelihood_taints_tracker(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    pots = (PotState("main", ChipAmount("10"), (0, 1, 2)),)
    tracker = AARangeShadowTrackerV2(asset)
    tracker.seed((query(1), query(2)))
    partial = tracker.observe_action(10, 1, "call")
    assert partial["likelihood_coverage"] == "0.50"
    snapshot = tracker.snapshot(hero_seat=0, pots=pots)
    assert not snapshot.equity_permitted
    assert "range_action_likelihood_coverage_low:1:call" in snapshot.blockers

    tracker.seed((query(1), query(2)))
    missing = tracker.observe_action(11, 2, "fold")
    assert not missing["applied"]
    assert "range_action_likelihood_missing:2:fold" in tracker.snapshot(
        hero_seat=0, pots=pots
    ).blockers


def test_tracker_rejects_reordered_events_and_unseeded_actor(tmp_path):
    path, digest = write_asset(tmp_path)
    asset = AAConcreteRangeAssetV2(
        path, expected_sha256=digest, expected_rule_fingerprint=RULES
    )
    tracker = AARangeShadowTrackerV2(asset)
    tracker.seed((query(1),))
    event = tracker.observe_action(10, 2, "call")
    assert event["blocker"] == "range_action_without_seed:2"
    assert tracker.version == 2 and tracker.events == [event]
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.observe_action(10, 1, "call")
