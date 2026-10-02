"""105 native-source cases for three saved equal continuation endpoints.

Each endpoint exercises 32 real arena observations and three refusal cases.
The real spawn worker and monotonic clock remain unchanged. These are offline
synthetic interface checks, not live qualification or new quality evaluation.
"""

import json
import os
from pathlib import Path

import pytest

from poker_engine.strategy.aa_frozen_policy import action_ids
from poker_engine.strategy.aa_full_hand_arena import AAFullHandArena
from tools.aa_research_arena_context import ArenaContextSource
from tools.aa_research_equal_bridge import load_artifact


FIXTURES = Path(__file__).parents[1] / "fixtures" / "equal_memory_continuation"
ENTRIES = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))[
    "artifacts"]
SEEDS = (2026100301, 2026100302, 2026100303)
BOARD = ["Ts", "Jh", "Qd", "Kc", "2c"]
RANGES = {
    "1": {"STRAIGHT_A": ["As", "6h"], "SET_K": ["Kh", "Kd"]},
    "2": {"STRAIGHT_B": ["Ah", "7h"], "SET_Q": ["Qc", "Qh"]},
}
DEALS = (("STRAIGHT_A", "STRAIGHT_B"), ("STRAIGHT_A", "SET_Q"),
         ("SET_K", "STRAIGHT_B"), ("SET_K", "SET_Q"))
C, B2, B4 = "check_call", "raise_to:2", "raise_to:4"
HISTORIES = ((), (C,), (B2,), (B4,), (C, B2), (C, B4),
             (B2, B4), (C, B2, B4))


@pytest.fixture(scope="module", params=ENTRIES,
                ids=lambda entry: f"equal-{entry['seed']}-16384")
def live_source(request):
    assert len(ENTRIES) == 3
    assert {entry["seed"] for entry in ENTRIES} == set(SEEDS)
    assert len({entry["sha256"] for entry in ENTRIES}) == 3
    entry = request.param
    assert entry["iterations"] == 16384
    artifact = load_artifact(FIXTURES / entry["path"],
                             expected_sha256=entry["sha256"])
    document = artifact.document
    checkpoint = document["sources"]["checkpoint"]
    fixture = document["sources"]["fixture"]
    assert checkpoint["seed"] == entry["seed"]
    assert checkpoint["iterations"] == 16384
    assert document["average_statistic"] == (
        "equal_time_premerge_linear_delta_over_iteration_v1")
    assert document["strategy_eligible"] is False
    assert document["advice_emitted"] is False
    assert document["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert document["sources"]["result"]["status"] == (
        "COMPLETE_CONTINUATION_NO_PROMOTION")
    assert fixture["board"] == BOARD
    assert fixture["active_ranges"] == RANGES
    assert {tuple(deal) for deal in fixture["joint_deals"]} == set(DEALS)
    source = ArenaContextSource(artifact, enabled=True)
    try:
        source.preload()
        process = source._bridge._worker._process
        assert process is not None and process.is_alive()
        assert type(process.pid) is int and process.pid > 0
        assert process.pid != os.getpid()
        yield source, artifact, process, document["sources"]["catalog"]
    finally:
        source.close()
        process = source._bridge._worker._process
        assert process is None or not process.is_alive()


def assert_refusal(result, reason):
    assert result["status"] == "ABSTAIN"
    assert result["reason"] == reason
    assert result["action"] is None
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False


@pytest.mark.parametrize("deal", DEALS)
@pytest.mark.parametrize("history", HISTORIES)
def test_latest_equal_native_observation_and_real_worker(
        live_source, deal, history):
    source, artifact, process, catalog = live_source
    source.reset(deal, seed=17)
    for action in history:
        source.step(action)
    snapshot = source.snapshot()
    observation = snapshot.observation
    assert isinstance(source._arena, AAFullHandArena)
    assert observation == source.observe(source.actor)
    assert observation["actor"] == source.actor
    assert observation["street"] == "river"
    assert observation["board"] == BOARD
    actor = observation["actor"]
    assert sorted(observation["own_hole"]) == sorted(
        RANGES[str(actor)][deal[actor - 1]])
    assert tuple(row["id"] for row in observation["public_history"][10:]) == (
        history)
    assert len(observation["public_history"]) == 10 + len(history)
    matches = [key for key, row in catalog.items()
               if row["visible_observation"] == observation]
    assert len(matches) == 1
    assert source._bridge._worker._process is process
    assert process.is_alive() and process.pid != os.getpid()

    result = source.lookup(snapshot)

    assert result["status"] == "SHADOW_RESULT"
    assert result["binding"]["information"] == matches[0]
    assert result["action"] in action_ids(observation)
    assert result["research_artifact_sha256"] == artifact.sha256
    assert result["average_statistic"] == (
        "equal_time_premerge_linear_delta_over_iteration_v1")
    assert result["source_batch_status"] == "COMPLETE_CONTINUATION_NO_PROMOTION"
    assert result["source_parent_batch_status"] == "STOP_ERROR_OR_BUDGET"
    assert result["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False
    assert result["arena_epoch"] == snapshot.epoch
    assert result["arena_revision"] == snapshot.revision
    assert source._bridge._worker._process is process and process.is_alive()


def test_latest_equal_refuses_previous_native_epoch(live_source):
    source, _, _, _ = live_source
    source.reset(DEALS[0], seed=17)
    old = source.snapshot()
    source.reset(DEALS[0], seed=17)
    current = source.snapshot()
    assert current.epoch > old.epoch
    assert current.observation == old.observation
    assert_refusal(source.lookup(old), "STALE_ARENA_EPOCH")
    assert source.snapshot() is current


@pytest.mark.parametrize("fault,swaps", [
    ("wrong_range", ((2, "4h"), (3, "4s"))),
    ("unsupported_board", ((13, "8h"),)),
])
def test_latest_equal_refuses_actual_native_out_of_scope_deal(
        live_source, monkeypatch, fault, swaps):
    source, _, process, _ = live_source
    source.reset(DEALS[0], seed=17)
    before = source.snapshot()
    original_deck, original_publish = source._deck, source._publish
    captured = {}

    def swapped_native_deck(deal, seed):
        deck = original_deck(deal, seed)
        original_cards = set(deck)
        for index, card in swaps:
            other = deck.index(card)
            deck[index], deck[other] = deck[other], deck[index]
        assert len(deck) == len(set(deck)) == 52
        assert set(deck) == original_cards
        return deck

    def observe_native_publish(arena, memory, journal, epoch, revision):
        assert isinstance(arena, AAFullHandArena)
        observation = arena.observe(arena.actor)
        captured.update(actor=arena.actor, board=observation["board"],
                        own_hole=observation["own_hole"],
                        history=observation["public_history"])
        return original_publish(arena, memory, journal, epoch, revision)

    with monkeypatch.context() as patch:
        patch.setattr(source, "_deck", swapped_native_deck)
        patch.setattr(source, "_publish", observe_native_publish)
        with pytest.raises(ValueError, match="^OUTSIDE_RESEARCH_SCOPE$"):
            source.reset(DEALS[0], seed=17)

    assert source._deck == original_deck
    assert source._publish == original_publish
    assert captured["actor"] == 1
    assert len(captured["history"]) == 10
    if fault == "wrong_range":
        assert sorted(captured["own_hole"]) == ["4h", "4s"]
        assert captured["board"] == BOARD
    else:
        assert captured["board"] == ["8h", "Jh", "Qd", "Kc", "2c"]
        assert sorted(captured["own_hole"]) == sorted(RANGES["1"][DEALS[0][0]])
    assert source.snapshot() is before
    assert source.epoch == before.epoch
    assert source.revision == before.revision
    assert source.observe(source.actor) == before.observation
    assert source._bridge._worker._process is process and process.is_alive()
