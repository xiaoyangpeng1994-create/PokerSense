"""Desktop live hand boundary: a qualified opening must survive the rest of the hand.

Regression for the duplicate-epoch defect observed on the frozen development
recording. ``LiveStateAdapter``'s dealer-advance branch re-anchored the hand at
source frame 1278 -- 14 frames after frame 1264 had already produced a valid
``MULTI_POST_DEAL_CANDIDATE`` opening with seven debits -- and the replacement
epoch made ``LiveHandLedger`` re-taint a ledger that had just resolved.

``tests/fixtures/aa_reference_hands/opening_rows_1260_1290_v1.json`` holds the
pre-adapter rows captured verbatim from source frames 1260..1290 (1260-1262 come
from ``aa8_first_hand_entry_dense_v2``, 1263+ from ``aa8_first_hand_full_v1``), so
CI reproduces the defect without the private ``G:`` pools. Rows are observation
inputs, never legal state.

The opening is independently corroborated by the reviewed development fixture
``eight_session_boundary_v1.json`` for the same hand: ``initial_posting_slots``
``[0, 1, 2, 3, 4, 5, 7]`` with slot 6 empty, and hero ``200 -> 196``.
"""

import json
from pathlib import Path

from poker_engine.desktop.aa_live_context import LiveHandLedger, LiveStateAdapter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "aa_reference_hands"
OPENING = FIXTURES / "opening_rows_1260_1290_v1.json"
REVIEWED_BOUNDARY = FIXTURES / "eight_session_boundary_v1.json"


def replay_alive():
    """Feed the frozen rows through the desktop boundary path, in order."""
    document = json.loads(OPENING.read_text(encoding="utf-8"))
    adapter, ledger = LiveStateAdapter(), LiveHandLedger()
    results = []
    for source in document["rows"]:
        row = json.loads(json.dumps(source))
        row["observed_state_v2"] = adapter.observe(row)
        results.append((row["frame"], ledger.observe(row, adapter)))
    return document, adapter, results


def test_fixture_reproduces_the_reviewed_opening_and_nothing_is_invented():
    """The frozen rows must still carry the reviewed posting, or the test is empty."""
    document, adapter, results = replay_alive()
    reviewed = json.loads(REVIEWED_BOUNDARY.read_text(encoding="utf-8"))

    rows = {row["frame"]: row for row in document["rows"]}
    before = rows[reviewed["start_global_frame"] - 3]["stacks"]
    after = rows[reviewed["start_global_frame"]]["stacks"]
    assert int(before["4"]["value"]) == int(reviewed["hero_balance_before_start"])
    assert int(after["4"]["value"]) == int(reviewed["hero_balance_at_start"])
    posted = sorted(int(seat) for seat, item in before.items()
                    if item["value"] is not None
                    and after[seat]["value"] is not None
                    and int(after[seat]["value"]) < int(item["value"]))
    assert posted == reviewed["initial_posting_slots"]
    assert str(reviewed["empty_slot_before_start"]) not in map(str, posted)

    opening = [event for event in adapter.epoch_events
               if event["status"] == "MULTI_POST_DEAL_CANDIDATE"]
    assert len(opening) == 1
    assert opening[0]["posting_comparison"]["debits"] == (
        document["expected_opening_debits"])
    assert opening[0]["posting_comparison"]["excluded_na_slots"] == (
        document["expected_excluded_na_slots"])
    assert results  # the replay itself must not be empty


def test_qualified_opening_is_not_replaced_by_a_later_dealer_reading():
    """A dealer-seat reading change must not discard this hand's opening ledger."""
    document, adapter, results = replay_alive()

    duplicate = [event for event in adapter.epoch_events
                 if event["frame"] == document["duplicate_epoch_frame"]]
    assert duplicate == [], (
        "a second epoch was created at frame "
        f"{document['duplicate_epoch_frame']} even though frame "
        f"{document['reproduced_epoch_frame']} already opened the hand with "
        f"{len(document['expected_opening_debits'])} debits")

    statuses = [ledger["status"] for _, ledger in results]
    resolved = [i for i, status in enumerate(statuses)
                if status == "OBSERVED_HAND_COMMITMENTS_CANDIDATE"]
    assert resolved, f"the opening never resolved a ledger: {sorted(set(statuses))}"
    first = resolved[0]
    lost = [(results[i][0], statuses[i]) for i in range(first, len(statuses))
            if statuses[i] != "OBSERVED_HAND_COMMITMENTS_CANDIDATE"]
    assert lost == [], f"the ledger stopped being resolved at {lost}"
    tainted = [(results[i][0], results[i][1]["taint_reasons"])
               for i in range(first, len(results))
               if results[i][1]["taint_reasons"]]
    assert tainted == [], f"a resolved ledger was re-tainted at {tainted}"

    final = results[-1][1]
    assert final["status"] == "OBSERVED_HAND_COMMITMENTS_CANDIDATE"
    assert final["opening_evidence"]["debits"] == document["expected_opening_debits"]
    assert final["opening_evidence"]["frame"] == document["reproduced_epoch_frame"]


def test_dealer_advance_without_a_qualified_opening_still_reanchors():
    """The fix must not freeze the epoch forever: an unanchored hand still moves."""
    adapter = LiveStateAdapter()
    for frame, dealer in ((0, 1), (1, 1), (2, 2), (3, 2)):
        row = {"frame": frame, "scene_supported": True, "source_sha256": "a" * 64,
               "live_dealer_candidate": dealer, "board_count": 0,
               "pot": {"value": "0"}, "hand_transition": {"center_deal": {}},
               "special_modes": {}, "glyph_transitions": [],
               "continuous_context": None, "current_actor": None,
               "cards": {"hero": [None, None], "board_slots": [None] * 5},
               "stacks": {str(s): {"value": "100"} for s in range(8)},
               "participation": {"slots": {}}}
        adapter.observe(row)
    assert [event["status"] for event in adapter.epoch_events] == [
        "DEALER_ADVANCE_CONTEXT_CANDIDATE"]
