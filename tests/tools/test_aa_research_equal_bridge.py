"""Bounded offline bridge checks: retained synthetic policies, real spawn worker.

These tests never train, fill coverage, grant live eligibility, or promote the
second game's stopped batch to a completed cross-validation experiment.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest

from poker_engine.desktop.aa_turn_runtime import (
    AATurnWindow, TurnEvidence, TurnIdentity,
)
from tools.aa_research_equal_bridge import (
    ResearchArtifact, ResearchEqualBridge, load_artifact,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "equal_memory_bridge"
INDEX = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))
ENTRIES = INDEX["artifacts"]
IDENTITY = TurnIdentity("synthetic-research-bridge", 1, "hand-1", "turn-1")


def artifact_path(entry):
    return FIXTURES / entry["path"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def altered_artifact(tmp_path, original, mutate):
    document = deepcopy(original.document)
    mutate(document)
    data = canonical(document)
    path = tmp_path / "altered.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def assert_refusal(result, reason=None):
    assert result["status"] == "ABSTAIN"
    assert result["action"] is None
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False
    if reason is not None:
        assert result["reason"] == reason


def observation_for(artifact, index=0):
    catalog = artifact.document["sources"]["catalog"]
    key = sorted(catalog)[index]
    return key, deepcopy(catalog[key]["visible_observation"])


def runtime(observation, pin, *, identity=IDENTITY, now=None):
    now = time.monotonic() if now is None else now
    evidence = TurnEvidence("synthetic-research-onset", "verified_onset", now, 10)
    window = AATurnWindow(identity, evidence, now=now)
    context = {"observation": deepcopy(observation), "identity": identity,
               "artifact_sha256": pin, "source_at": now}
    return dict(artifact_sha256=pin, simulation=True, identity=identity,
                window=window, source_at=now,
                current_context=lambda: deepcopy(context))


@pytest.fixture(scope="module", params=ENTRIES,
                ids=lambda entry: entry["name"])
def loaded(request):
    entry = request.param
    artifact = load_artifact(artifact_path(entry),
                             expected_sha256=entry["sha256"])
    return entry, artifact


@pytest.fixture(scope="module")
def real_bridge(loaded):
    entry, artifact = loaded
    bridge = ResearchEqualBridge(artifact, enabled=True)
    bridge.preload()
    try:
        yield entry, artifact, bridge
    finally:
        bridge.close()


def test_four_retained_sources_have_explicit_nonproduct_semantics():
    assert len(ENTRIES) == 4
    assert len({entry["sha256"] for entry in ENTRIES}) == 4
    assert sum(entry["source_batch_status"].startswith("STOP")
               for entry in ENTRIES) == 1


def test_source_bytes_match_external_pin_and_equal_semantics(loaded):
    entry, artifact = loaded
    assert hashlib.sha256(artifact_path(entry).read_bytes()).hexdigest() == (
        entry["sha256"])
    assert artifact.sha256 == entry["sha256"]
    document = artifact.document
    assert document["kind"] == "POKERSENSE_EQUAL_MEMORY_RESEARCH_BRIDGE_V1"
    assert document["schema_version"] == 1
    assert document["training_algorithm"] == "external_sampling_simple_linear_v1"
    assert document["average_statistic"] == (
        "equal_time_premerge_linear_delta_over_iteration_v1")
    assert document["information_encoding"] == (
        "test-river-v2-exact-plus-own-trace-v1")
    assert document["simulation_only"] is True
    assert document["strategy_eligible"] is False
    assert document["advice_emitted"] is False
    assert document["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert set(document["sources"]) == {
        "checkpoint", "equal", "catalog", "fixture", "quality", "result"}


def test_document_exposure_is_a_deep_copy(loaded):
    _, artifact = loaded
    original = artifact.document
    changed = artifact.document
    changed["sources"]["checkpoint"] = {"forged": True}
    changed["strategy_eligible"] = True
    assert artifact.document == original


def test_artifact_constructor_cannot_bypass_byte_authentication(loaded):
    _, artifact = loaded
    with pytest.raises(ValueError):
        ResearchArtifact(artifact.document, artifact.sha256)


def test_public_identity_is_read_only_and_private_damage_is_rechecked(loaded):
    entry, artifact = loaded
    with pytest.raises(AttributeError):
        artifact.sha256 = "0" * 64
    damaged = ResearchArtifact(artifact_path(entry).read_bytes(), artifact.sha256)
    damaged._sha256 = "0" * 64
    with pytest.raises(ValueError):
        ResearchEqualBridge(damaged, enabled=True)


def test_external_artifact_pin_must_match(loaded):
    entry, _ = loaded
    with pytest.raises(ValueError):
        load_artifact(artifact_path(entry), expected_sha256="0" * 64)


@pytest.mark.parametrize("field,value", [
    ("kind", "AA_FROZEN_POLICY_V2"),
    ("schema_version", True),
    ("training_algorithm", "external_sampling_equal_v1"),
    ("average_statistic", "linear_weighted_average_v1"),
    ("information_encoding", "v2"),
    ("simulation_only", False),
    ("strategy_eligible", True),
    ("advice_emitted", True),
    ("qualification", "PRODUCT_QUALIFIED"),
])
def test_repinning_does_not_authorize_semantic_laundering(
        loaded, tmp_path, field, value):
    _, artifact = loaded
    path, pin = altered_artifact(
        tmp_path, artifact, lambda document: document.__setitem__(field, value))
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=pin)


@pytest.mark.parametrize("source", [
    "checkpoint", "equal", "catalog", "fixture", "quality", "result",
])
def test_missing_source_cannot_be_repaired_by_new_outer_pin(
        loaded, tmp_path, source):
    _, artifact = loaded
    path, pin = altered_artifact(
        tmp_path, artifact, lambda document: document["sources"].pop(source))
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=pin)


@pytest.mark.parametrize("row_index", range(16))
def test_all_retained_rows_use_real_worker_and_preserve_research_gate(
        real_bridge, row_index):
    entry, artifact, bridge = real_bridge
    key, observation = observation_for(artifact, row_index)
    result = bridge.lookup(observation, **runtime(observation, artifact.sha256))
    assert result["status"] == "SHADOW_RESULT"
    assert result["binding"]["information"] == key
    record = artifact.document["sources"]["catalog"][key]
    assert result["action"] in record["menu"]
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False
    assert result["binding"]["turn"]["instance_id"] == IDENTITY.instance_id
    assert result["source_batch_status"] == entry["source_batch_status"]
    assert result["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert result["average_statistic"] == (
        "equal_time_premerge_linear_delta_over_iteration_v1")


def test_same_complete_decision_is_stable_under_worker_retry(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256)
    first = bridge.lookup(deepcopy(observation), **args)
    second = bridge.lookup(deepcopy(observation), **args)
    assert first["status"] == second["status"] == "SHADOW_RESULT"
    assert first["action"] == second["action"]
    assert first["request_key"] == second["request_key"]


def test_bridge_is_disabled_without_explicit_opt_in(loaded):
    _, artifact = loaded
    _, observation = observation_for(artifact)
    bridge = ResearchEqualBridge(artifact)
    try:
        result = bridge.lookup(observation, **runtime(observation, artifact.sha256))
        assert_refusal(result, "DISABLED")
    finally:
        bridge.close()


def test_real_worker_requires_explicit_simulation(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256)
    args["simulation"] = False
    assert_refusal(bridge.lookup(observation, **args), "SIMULATION_ONLY")


def test_enabled_bridge_does_not_autostart_a_worker_in_a_turn(loaded):
    _, artifact = loaded
    _, observation = observation_for(artifact)
    bridge = ResearchEqualBridge(artifact, enabled=True)
    try:
        assert_refusal(
            bridge.lookup(observation, **runtime(observation, artifact.sha256)),
            "WORKER_NOT_PRELOADED")
    finally:
        bridge.close()


def test_call_cannot_substitute_another_artifact_identity(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256)
    args["artifact_sha256"] = "0" * 64
    assert_refusal(bridge.lookup(observation, **args),
                   "ARTIFACT_IDENTITY_MISMATCH")


@pytest.mark.parametrize("change", [
    "observation", "identity", "artifact_sha256", "source_at", "bool",
])
def test_full_current_context_is_rechecked_after_real_lookup(real_bridge, change):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256)
    context = args["current_context"]()
    if change == "bool":
        args["current_context"] = lambda: True
    else:
        if change == "observation":
            context[change]["unexpected_current_state"] = True
        elif change == "identity":
            context[change] = replace(IDENTITY, turn_id="different-turn")
        elif change == "artifact_sha256":
            context[change] = "0" * 64
        else:
            context[change] -= 0.01
        args["current_context"] = lambda: deepcopy(context)
    assert_refusal(bridge.lookup(observation, **args), "STATE_CHANGED")


def test_expired_source_is_refused_by_unchanged_runtime_window(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256, now=100)
    args["source_at"] = 98
    args["clock"] = lambda: 100
    assert_refusal(bridge.lookup(observation, **args), "SOURCE_STALE")


def test_stale_turn_identity_is_refused_by_unchanged_runtime_window(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256, now=100)
    args["identity"] = replace(IDENTITY, turn_id="different-turn")
    args["clock"] = lambda: 100
    assert_refusal(bridge.lookup(observation, **args), "TURN_IDENTITY_CHANGED")


def test_elapsed_first_result_window_is_not_replenished(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256, now=100)
    args["source_at"] = 102.01
    args["clock"] = lambda: 102.01
    assert_refusal(bridge.lookup(observation, **args), "FIRST_RESULT_DEADLINE")


@pytest.mark.parametrize("change", [
    "board", "own_range", "opponent_range", "history", "menu", "rules",
    "forgotten_memory", "changed_memory", "encoding", "hidden_cards",
])
def test_scope_and_visibility_changes_refuse_before_worker_dispatch(
        real_bridge, monkeypatch, change):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    if change == "board":
        observation["board"][0] = "3h"
    elif change == "own_range":
        observation["own_hole"] = ["4h", "4s"]
    elif change == "opponent_range":
        observation["opponent_range"] = {"4h4s": 1.0}
    elif change == "history":
        observation["public_history"] = observation["public_history"][1:]
    elif change == "menu":
        observation["legal_actions"] = observation["legal_actions"][:-1]
    elif change == "rules":
        observation["rules"]["rake_percent"] = "0.04"
    elif change == "forgotten_memory":
        observation["own_memory"] = observation["own_memory"][1:]
    elif change == "changed_memory":
        observation["own_memory"][0]["visible_observation_sha256"] = "0" * 64
    elif change == "encoding":
        observation["information_encoding"] = "v2"
    else:
        observation["opponent_hole"] = ["4h", "4s"]

    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("out-of-scope observation reached the policy worker")

    monkeypatch.setattr(bridge._worker, "lookup", unexpected_dispatch)
    assert_refusal(
        bridge.lookup(observation, **runtime(observation, artifact.sha256)),
        "OUTSIDE_RESEARCH_SCOPE")


@pytest.mark.parametrize("change", [
    "native_digest", "equal_binding", "equal_seed", "negative_weight",
    "catalog_memory", "quality_variant", "policy_row",
])
def test_internal_source_damage_refuses_even_with_new_container_pin(
        loaded, tmp_path, change):
    _, artifact = loaded

    def damage(document):
        sources = document["sources"]
        if change == "native_digest":
            sources["checkpoint"]["sha256"] = "0" * 64
        elif change == "equal_binding":
            sources["equal"]["native_checkpoint_sha256"] = "0" * 64
        elif change == "equal_seed":
            sources["equal"]["seed"] = True
        elif change == "negative_weight":
            row = next(iter(sources["equal"]["average"].values()))
            row[next(iter(row))] = -1
        elif change == "catalog_memory":
            record = next(iter(sources["catalog"].values()))
            record["visible_observation"]["own_memory"] = []
        elif change == "quality_variant":
            sources["quality"]["variant"] = "linear"
        else:
            row = next(iter(document["policy"].values()))
            row[next(iter(row))] = 2.0
        # This does not restore the authenticated original source-file bytes.
        # It ensures rejection also follows the inner semantic contracts.
        for key, value in sources.items():
            document["source_canonical_sha256"][key] = (
                hashlib.sha256(canonical(value)).hexdigest())

    path, pin = altered_artifact(tmp_path, artifact, damage)
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=pin)


def test_boolean_probability_cannot_equal_a_valid_one_under_repinning(
        loaded, tmp_path):
    _, artifact = loaded

    def damage(document):
        candidates = [(row, action)
                      for row in document["policy"].values()
                      for action, probability in row.items()
                      if type(probability) is float and probability == 1.0]
        assert candidates, "retained fixture must contain a pure-action row"
        row, action = candidates[0]
        row[action] = True

    path, pin = altered_artifact(tmp_path, artifact, damage)
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=pin)


def test_unbound_equal_history_is_not_a_committed_checkpoint(
        loaded, tmp_path):
    _, artifact = loaded

    def damage(document):
        equal = document["sources"]["equal"]
        del equal["native_checkpoint_sha256"]
        document["source_canonical_sha256"]["equal"] = (
            hashlib.sha256(canonical(equal)).hexdigest())

    path, pin = altered_artifact(tmp_path, artifact, damage)
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=pin)


def test_current_context_failure_is_a_state_refusal(real_bridge):
    _, artifact, bridge = real_bridge
    _, observation = observation_for(artifact)
    args = runtime(observation, artifact.sha256)

    def unavailable_context():
        raise RuntimeError("synthetic current state temporarily unavailable")

    args["current_context"] = unavailable_context
    assert_refusal(bridge.lookup(observation, **args), "STATE_CHANGED")


@pytest.mark.parametrize("raw", [b'{"kind":1,"kind":2}', b'{"value":NaN}'])
def test_ambiguous_or_nonfinite_json_is_not_a_research_artifact(tmp_path, raw):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_artifact(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_retained_partial_history_keeps_real_missing_key_without_fallback():
    entry = INDEX["coverage_miss"]
    path = artifact_path(entry)
    original_bytes = path.read_bytes()
    artifact = load_artifact(path, expected_sha256=entry["sha256"])
    document = artifact.document
    catalog = document["sources"]["catalog"]
    policy = document["policy"]
    assert len(catalog) == 16
    assert len(policy) == 15
    assert len(set(catalog) - set(policy)) == 1
    bridge = ResearchEqualBridge(artifact, enabled=True)
    bridge.preload()
    try:
        results = []
        for key, record in sorted(catalog.items()):
            observation = deepcopy(record["visible_observation"])
            result = bridge.lookup(
                observation, **runtime(observation, artifact.sha256))
            results.append(result)
            if key not in policy:
                assert_refusal(result, "POLICY_COVERAGE_MISS")
            else:
                assert result["status"] == "SHADOW_RESULT"
                assert result["action"] in record["menu"]
                assert result["strategy_eligible"] is False
                assert result["advice_emitted"] is False
        assert sum(row["status"] == "SHADOW_RESULT" for row in results) == 15
        assert sum(row["status"] == "ABSTAIN" for row in results) == 1
    finally:
        bridge.close()
    assert path.read_bytes() == original_bytes
