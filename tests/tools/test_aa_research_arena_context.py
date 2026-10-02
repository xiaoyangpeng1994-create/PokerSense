"""60 fixed cases: native synthetic arena receipts, never training or capture."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from threading import Thread

import pytest

from poker_engine.strategy.aa_full_hand_arena import AAFullHandArena
from tools.aa_research_arena_context import ArenaContextSource
from tools.aa_research_equal_bridge import load_artifact


FIXTURES = Path(__file__).parents[1] / "fixtures" / "equal_memory_bridge"
ENTRY = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))[
    "artifacts"][0]
DEALS = (("AA", "KK"), ("AA", "88"), ("TT", "KK"), ("TT", "88"))
C, B2, B4 = "check_call", "raise_to:2", "raise_to:4"
HISTORIES = ((), (C,), (B2,), (B4,), (C, B2), (C, B4),
             (B2, B4), (C, B2, B4))


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds=10.0):
        self.value += seconds


@pytest.fixture(scope="module")
def artifact():
    return load_artifact(FIXTURES / ENTRY["path"],
                         expected_sha256=ENTRY["sha256"])


@pytest.fixture(scope="module")
def live_source(artifact):
    clock = Clock()
    source = ArenaContextSource(artifact, enabled=True, clock=clock)
    source.preload()
    try:
        yield source, clock
    finally:
        source.close()


@pytest.fixture
def ready_source(live_source):
    source, clock = live_source
    clock.advance()
    source.reset(DEALS[0], seed=0)
    return source, clock


def assert_refusal(result, reason):
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == reason
    assert result["action"] is None
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False


def receipt(snapshot):
    return (snapshot.source_id, snapshot.epoch, snapshot.revision,
            snapshot.identity, snapshot.source_at, snapshot.observation)


def matching_key(artifact, observation):
    matches = [key for key, record in
               artifact.document["sources"]["catalog"].items()
               if record["visible_observation"] == observation]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("deal", DEALS)
@pytest.mark.parametrize("history", HISTORIES)
def test_all_native_deals_and_histories_match_catalog_and_real_worker(
        live_source, artifact, deal, history):
    source, clock = live_source
    clock.advance()
    source.reset(deal, seed=17)
    for action in history:
        source.step(action)
    snapshot = source.snapshot()
    observation = snapshot.observation
    assert isinstance(source._arena, AAFullHandArena)
    assert observation == source.observe(source.actor)
    assert observation["actor"] == source.actor
    assert observation["street"] == "river"
    assert tuple(row["id"] for row in observation["public_history"][10:]) == (
        history)
    key = matching_key(artifact, observation)
    result = source.lookup(snapshot)
    assert result["status"] == "SHADOW_RESULT"
    assert result["binding"]["information"] == key
    assert result["action"] in [row["id"] for row in observation["legal_actions"]]
    assert result["research_artifact_sha256"] == artifact.sha256
    assert result["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False


def test_default_is_disabled_for_reset_and_preload(artifact):
    source = ArenaContextSource(artifact)
    try:
        with pytest.raises(ValueError, match="DISABLED"):
            source.reset(DEALS[0])
        with pytest.raises(ValueError, match="DISABLED"):
            source.preload()
    finally:
        source.close()


def test_no_snapshot_or_observation_before_reset(artifact):
    source = ArenaContextSource(artifact, enabled=True)
    try:
        with pytest.raises(ValueError, match="NOT_RESET"):
            source.snapshot()
        with pytest.raises(ValueError, match="NOT_RESET"):
            source.observe(1)
    finally:
        source.close()


@pytest.mark.parametrize("operation", ["snapshot", "step"])
def test_nonactor_cannot_request_snapshot_or_apply_action(ready_source, operation):
    source, _ = ready_source
    before = source.snapshot()
    with pytest.raises(ValueError, match="NON_ACTOR"):
        if operation == "snapshot":
            source.snapshot(seat=2)
        else:
            source.step(C, seat=2)
    assert receipt(source.snapshot()) == receipt(before)


def test_snapshot_and_observe_do_not_expose_mutable_state_or_window(ready_source):
    source, _ = ready_source
    snapshot = source.snapshot()
    expected = snapshot.observation
    changed = snapshot.observation
    changed["own_memory"].clear()
    changed["board"][0] = "3h"
    observed = source.observe(source.actor)
    observed["public_history"].clear()
    assert snapshot.observation == expected
    assert source.observe(source.actor) == expected
    assert not hasattr(snapshot, "window")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        snapshot.source_at = 0


def test_successful_step_invalidates_old_state_receipt(ready_source):
    source, clock = ready_source
    before = source.snapshot()
    clock.advance(0.01)
    source.step(C)
    after = source.snapshot()
    assert after.epoch == before.epoch
    assert after.revision > before.revision
    assert after.source_at > before.source_at
    assert_refusal(source.lookup(before), "STATE_CHANGED")
    assert source.lookup(after)["status"] == "SHADOW_RESULT"


def test_successful_reset_invalidates_old_arena_epoch(ready_source):
    source, clock = ready_source
    before = source.snapshot()
    clock.advance(0.01)
    source.reset(DEALS[0], seed=0)
    after = source.snapshot()
    assert after.epoch > before.epoch
    assert after.source_at > before.source_at
    assert_refusal(source.lookup(before), "STALE_ARENA_EPOCH")
    assert source.lookup(after)["status"] == "SHADOW_RESULT"


@pytest.mark.parametrize("failure", ["illegal", "mutate_then_raise"])
def test_failed_step_keeps_native_state_memory_and_receipt(
        ready_source, monkeypatch, failure):
    source, clock = ready_source
    before = source.snapshot()
    memory = deepcopy(source._memory)
    journal = deepcopy(source._journal)
    clock.advance(0.01)
    with monkeypatch.context() as patch:
        if failure == "mutate_then_raise":
            clone = source._arena.clone()
            original_step = clone.step

            def failed_step(action):
                original_step(action)
                raise RuntimeError("synthetic native step failure after mutation")

            patch.setattr(clone, "step", failed_step)
            patch.setattr(source._arena, "clone", lambda: clone)
        with pytest.raises(ValueError) as raised:
            source.step("raise_to:999" if failure == "illegal" else C)
        if failure == "mutate_then_raise":
            assert str(raised.value) == "ARENA_TRANSACTION_FAILED"
            assert isinstance(raised.value.__cause__, RuntimeError)
    assert receipt(source.snapshot()) == receipt(before)
    assert source._memory == memory
    assert source._journal == journal
    assert source.lookup(before)["status"] == "SHADOW_RESULT"


@pytest.mark.parametrize("failure", ["unsupported_deal", "native_reset"])
def test_failed_reset_keeps_epoch_state_memory_and_receipt(
        ready_source, monkeypatch, failure):
    source, clock = ready_source
    before = source.snapshot()
    memory = deepcopy(source._memory)
    journal = deepcopy(source._journal)
    clock.advance(0.01)
    with monkeypatch.context() as patch:
        if failure == "native_reset":
            original_reset = AAFullHandArena.reset

            def failed_reset(arena, *args, **kwargs):
                original_reset(arena, *args, **kwargs)
                raise RuntimeError("synthetic native reset failure after mutation")

            patch.setattr(AAFullHandArena, "reset", failed_reset)
        with pytest.raises(ValueError) as raised:
            source.reset(("unsupported", "KK") if failure == "unsupported_deal"
                         else DEALS[0], seed=0)
        if failure == "native_reset":
            assert str(raised.value) == "ARENA_TRANSACTION_FAILED"
            assert isinstance(raised.value.__cause__, RuntimeError)
    assert receipt(source.snapshot()) == receipt(before)
    assert source._memory == memory
    assert source._journal == journal
    assert source.lookup(before)["status"] == "SHADOW_RESULT"


def test_terminal_state_cannot_mint_another_decision(ready_source):
    source, _ = ready_source
    before = source.snapshot()
    source.step(C)
    source.step(C)
    assert source.terminal is True
    with pytest.raises(ValueError, match="TERMINAL"):
        source.snapshot()
    assert_refusal(source.lookup(before), "STATE_CHANGED")


@pytest.mark.parametrize("actor", [1, 2])
def test_opponent_hidden_cards_do_not_change_actor_visible_state(
        live_source, artifact, actor):
    source, clock = live_source
    deals = (("AA", "KK"), ("AA", "88")) if actor == 1 else (
        ("AA", "KK"), ("TT", "KK"))
    observations = []
    keys = []
    for deal in deals:
        clock.advance()
        source.reset(deal, seed=7)
        if actor == 2:
            source.step(C)
        snapshot = source.snapshot(seat=actor)
        observations.append(snapshot.observation)
        keys.append(matching_key(artifact, snapshot.observation))
        assert source.lookup(snapshot)["status"] == "SHADOW_RESULT"
    assert observations[0] == observations[1]
    assert keys[0] == keys[1]
    assert "opponent_hole" not in observations[0]
    assert "deck" not in observations[0]
    assert "seed" not in observations[0]


def test_unused_deck_tail_seed_is_not_part_of_visible_decision(live_source):
    source, clock = live_source
    observations = []
    decks = []
    for seed in (0, 92837):
        clock.advance()
        source.reset(DEALS[0], seed=seed)
        observations.append(source.snapshot().observation)
        decks.append(tuple(map(repr, source._arena._deck)))
    assert observations[0] == observations[1]
    assert decks[0][:20] == decks[1][:20]
    assert decks[0][20:] != decks[1][20:]


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_memory_loss_or_edit_is_refused(ready_source, damage):
    source, _ = ready_source
    before = source.snapshot()
    if damage == "missing":
        source._memory[source.actor].clear()
    else:
        source._memory[source.actor][0]["visible_observation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="MEMORY_INTEGRITY_MISMATCH"):
        source.snapshot()
    assert_refusal(source.lookup(before), "MEMORY_INTEGRITY_MISMATCH")


@pytest.mark.parametrize("damage", ["board", "range", "rules"])
def test_unexpected_native_observation_is_refused(
        ready_source, monkeypatch, damage):
    source, _ = ready_source
    before = source.snapshot()
    original_observe = source._arena.observe

    def corrupted_observe(seat):
        observation = original_observe(seat)
        if damage == "board":
            observation["board"][0] = "3h"
        elif damage == "range":
            observation["own_hole"] = ["4h", "4s"]
        else:
            observation["rules"]["rake_percent"] = "0.04"
        return observation

    monkeypatch.setattr(source._arena, "observe", corrupted_observe)
    with pytest.raises(ValueError, match="OUTSIDE_RESEARCH_SCOPE"):
        source.snapshot()
    assert_refusal(source.lookup(before), "OUTSIDE_RESEARCH_SCOPE")


def test_copied_snapshot_is_not_a_minted_receipt(ready_source):
    source, _ = ready_source
    original = source.snapshot()
    forged = replace(original)
    assert receipt(forged) == receipt(original)
    assert forged is not original
    assert_refusal(source.lookup(forged), "INVALID_ARENA_SNAPSHOT")


def test_another_instance_cannot_use_a_valid_snapshot(ready_source, artifact):
    source, clock = ready_source
    original = source.snapshot()
    other = ArenaContextSource(artifact, enabled=True, clock=clock)
    try:
        other.reset(DEALS[0], seed=0)
        assert other.snapshot().observation == original.observation
        assert_refusal(other.lookup(original), "INVALID_ARENA_SNAPSHOT")
    finally:
        other.close()


def test_repeated_snapshot_cannot_refresh_source_ttl(ready_source):
    source, clock = ready_source
    original = source.snapshot()
    clock.advance(1.01)
    current = source.snapshot()
    assert receipt(current) == receipt(original)
    assert_refusal(source.lookup(current), "SOURCE_STALE")


def test_closed_source_does_not_reset_preload_or_issue_advice(artifact):
    source = ArenaContextSource(artifact, enabled=True, clock=Clock())
    source.reset(DEALS[0], seed=0)
    snapshot = source.snapshot()
    source.close()
    with pytest.raises(ValueError, match="CLOSED"):
        source.reset(DEALS[0])
    with pytest.raises(ValueError, match="CLOSED"):
        source.preload()
    assert_refusal(source.lookup(snapshot), "CLOSED")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")],
                         ids=["nan", "infinity"])
def test_nonfinite_clock_cannot_commit_reset_or_step(ready_source, invalid):
    source, clock = ready_source
    before = source.snapshot()
    memory = deepcopy(source._memory)
    journal = deepcopy(source._journal)
    clock.value = invalid
    try:
        with pytest.raises(ValueError, match="INVALID_SOURCE_CLOCK"):
            source.reset(DEALS[0], seed=0)
        assert receipt(source.snapshot()) == receipt(before)
        with pytest.raises(ValueError, match="INVALID_SOURCE_CLOCK"):
            source.step(C)
        assert receipt(source.snapshot()) == receipt(before)
        assert source._memory == memory
        assert source._journal == journal
    finally:
        clock.value = before.source_at + 0.01
    assert source.lookup(before)["status"] == "SHADOW_RESULT"


def test_clock_cannot_rewind_below_time_already_read_by_real_lookup(ready_source):
    source, clock = ready_source
    before = source.snapshot()
    memory = deepcopy(source._memory)
    journal = deepcopy(source._journal)
    clock.value = before.source_at + 0.6
    assert source.lookup(before)["status"] == "SHADOW_RESULT"
    clock.value = before.source_at + 0.2
    try:
        with pytest.raises(ValueError, match="CLOCK_REGRESSION"):
            source.step(C)
        assert receipt(source.snapshot()) == receipt(before)
        assert source._memory == memory
        assert source._journal == journal
    finally:
        clock.value = before.source_at + 0.61
    assert source.lookup(before)["status"] == "SHADOW_RESULT"


def test_actual_step_after_real_worker_result_invalidates_final_return(
        ready_source, monkeypatch):
    source, _ = ready_source
    before = source.snapshot()
    original_lookup = source._bridge.lookup
    threads, failures = [], []

    def step_in_another_thread():
        try:
            source.step(C)
        except Exception as error:
            failures.append(error)

    def real_lookup_then_change_state(*args, **kwargs):
        result = original_lookup(*args, **kwargs)
        assert result["status"] == "SHADOW_RESULT"
        thread = Thread(target=step_in_another_thread, daemon=True)
        threads.append(thread)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive(), "state change blocked during worker lookup"
        assert not failures
        return result

    monkeypatch.setattr(source._bridge, "lookup", real_lookup_then_change_state)
    try:
        assert_refusal(source.lookup(before), "STATE_CHANGED")
    finally:
        for thread in threads:
            thread.join(timeout=2)
    assert len(threads) == 1 and not threads[0].is_alive()
    assert not failures
    assert source.revision > before.revision
