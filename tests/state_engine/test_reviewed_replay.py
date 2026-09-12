"""Review-driven ledger tests, distinct from raw-frame/live acceptance."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path

import pytest

pytest.importorskip("pokerkit")

from poker_engine.core.value_objects import ChipAmount  # noqa: E402
from poker_engine.state_engine.action_reconstruction import (  # noqa: E402
    ReconstructionStatus, reconstruct_action_event,
)
from poker_engine.state_engine.reviewed_replay import (  # noqa: E402
    audit_observed_settlement, decision_seats, replay_reviewed_hand,
    terminal_call_projection, verify_reviewed_checkpoints,
)
from poker_engine.strategy.state import calculate_side_pots  # noqa: E402


@pytest.fixture
def trace():
    path = (Path(__file__).resolve().parents[1]
            / "fixtures/wpk_reference_hands/aq_allin_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_review_reconciles_every_action_without_combining_refund(trace):
    original = deepcopy(trace)
    steps = replay_reviewed_hand(trace)
    actions = [step for step in steps if step.kind == "action"]
    assert len(actions) == 13
    assert [s.state.state_version for s in steps] == list(range(len(steps)))
    assert [actions[i].state.pot.value for i in (8, 10, 11, 12)] == [94, 218, 520, 584]
    assert actions[-1].action_event.payload["amount_additional"] == "64"
    assert actions[-1].action_event.payload["all_in"] is True
    assert steps[-1].kind == "uncalled_return"
    assert steps[-1].returned == ((6, Decimal(238)),)
    assert steps[-1].state.pot.value == 346
    assert next(p for p in steps[-1].state.players if p.seat == 6).stack.value == 238
    assert trace == original
    assert verify_reviewed_checkpoints(trace, steps) == 5
    # The old one-action reducer must still reject a multi-player money move.
    joined = reconstruct_action_event(
        actions[-2].state, steps[-1].state, actor_seat=0,
        observed_action=None, timestamp=actions[-1].action_event.timestamp)
    assert joined.status is ReconstructionStatus.INVALID


def test_antes_are_dead_money_not_live_street_bets(trace):
    first = replay_reviewed_hand(trace, through_order=1)[0].state
    hero = next(p for p in first.players if p.is_hero)
    small = next(p for p in first.players if p.seat == 2)
    assert hero.committed_this_hand.value == 2
    assert hero.committed_this_street.value == 0
    assert small.committed_this_hand.value == 4
    assert small.committed_this_street.value == 2
    assert first.pot.value == 22


def test_pre_call_projection_uses_only_current_state(trace):
    prefix = replay_reviewed_hand(trace, through_order=12)
    state = prefix[-1].state
    projection = terminal_call_projection(state)
    assert state.actor == 0 and state.to_call.value == 64
    assert projection.additional == 64
    assert projection.eligible_pot_after_call == 346
    assert projection.uncalled_returns == ((6, Decimal(238)),)
    assert projection.break_even_before_fees == Fraction(32, 173)
    poisoned = deepcopy(trace)
    poisoned["actions"][-1]["amount"] = 999999
    poisoned["actions"][-1]["order"] = "future order must not be consumed"
    poisoned["cards"][1]["cards"] = ["not", "cards"]
    poisoned["board"][-1]["cards"] = ["not", "a", "board"]
    poisoned["settlement"] = {"untrusted_future": True}
    poisoned["checkpoints"][-2]["raw_total_pot"] = "unobserved future value"
    for row in poisoned["source_seats"]:
        row["hand_contributed"] = 999999
        row["settled_stack"] = "not observed yet"
    assert replay_reviewed_hand(poisoned, through_order=12) == prefix


def test_unknown_fee_difference_stays_unallocated(trace):
    final = replay_reviewed_hand(trace)[-1].state
    audit = audit_observed_settlement(trace, final)
    assert audit["observed_positive_stack_deltas"][0] == 324
    assert audit["unallocated_outflow"] == 22
    assert audit["insurance_notice_not_cash_entry"] == 12
    assert not audit["rake_confirmed"]
    assert not audit["insurance_debit_confirmed"]
    assert not audit["capture_to_state_verified"]


def test_missing_insurance_notice_is_unknown_not_zero(trace):
    final = replay_reviewed_hand(trace)[-1].state
    trace.pop("settlement")
    audit = audit_observed_settlement(trace, final)
    assert audit["insurance_notice_not_cash_entry"] is None
    assert audit["unallocated_outflow"] == 22


@pytest.mark.parametrize("error", [
    "call", "actor", "semantics", "all_in", "order", "evidence", "straddle", "variant",
])
def test_bad_or_unsupported_review_cannot_silently_replay(trace, error):
    if error == "call":
        trace["actions"][0]["amount"] = 5
    elif error == "actor":
        trace["actions"][0]["source_seat"] = 0
    elif error == "semantics":
        trace["actions"][2]["amount_semantics"] = "additional"
    elif error == "all_in":
        trace["actions"][-1]["all_in"] = False
    elif error == "order":
        trace["actions"][0]["order"] = 2
    elif error == "evidence":
        trace["actions"][0]["evidence"] = []
    elif error == "straddle":
        trace["rules"]["observed_straddle"] = 8
    else:
        trace["rules"]["variant"] = "bomb_pot"
    with pytest.raises(ValueError):
        replay_reviewed_hand(trace)


def synthetic_side_pot_hand(count):
    positions = {6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
                 7: ["SB", "BB", "UTG", "LJ", "HJ", "CO", "BTN"],
                 8: ["SB", "BB", "UTG", "UTG1", "LJ", "HJ", "CO", "BTN"]}[count]
    rows = [{"seat": i, "position": position,
             "starting_stack": [100, 60, 20][i] if i < 3 else 400}
            for i, position in enumerate(positions)]
    moves = [(2, "raise_to", 20, True)]
    moves += [(seat, "fold", 0, False) for seat in range(3, count)]
    moves += [(0, "raise_to", 100, True), (1, "call", 56, True)]
    actions = [{"order": i + 1, "street": "preflop", "source_seat": seat,
                "kind": kind, "amount": amount, "all_in": all_in,
                "amount_semantics": "total_street" if kind == "raise_to"
                else "additional",
                "evidence": [f"a{i + 1}"]}
               for i, (seat, kind, amount, all_in) in enumerate(moves)]
    evidence = {f"a{i + 1}": {"source_frame": i + 10} for i in range(len(moves))}
    evidence["posts"] = {"source_frame": 0}
    return {"hand_id": f"synthetic-{count}", "rules": {
        "player_count": count, "small_blind": 2, "big_blind": 4, "ante_each": 0,
        "observed_straddle": 0, "hero_source_seat": 0, "dealer_source_seat": count - 1},
        "source_seats": rows, "pokerkit_index_to_source_seat": list(range(count)),
        "cards": [{"seat": 0, "cards": ["As", "Ah"],
                   "first_verified_visible_frame": 0}],
        "board": [], "actions": actions, "evidence": evidence,
        "checkpoints": [{"stage": "forced_posts", "evidence": "posts",
                         "raw_total_pot": 6}]}


@pytest.mark.parametrize("count", [6, 7, 8])
def test_multiway_main_side_pots_and_uncalled_money(count):
    steps = replay_reviewed_hand(synthetic_side_pot_hand(count))
    final = steps[-1]
    assert final.returned == ((0, Decimal(40)),)
    assert final.state.pot.value == 140
    pots = calculate_side_pots(decision_seats(final.state))
    assert [(p.amount.value, p.eligible_seats) for p in pots.pots] == [
        (Decimal(60), (0, 1, 2)), (Decimal(80), (0, 1))]


def test_fractional_stakes_are_exact_and_not_hardcoded():
    trace = synthetic_side_pot_hand(6)
    for key in ("small_blind", "big_blind", "ante_each"):
        trace["rules"][key] = str(Decimal(trace["rules"][key]) / 10)
    for row in trace["source_seats"]:
        row["starting_stack"] = str(Decimal(row["starting_stack"]) / 10)
    for action in trace["actions"]:
        action["amount"] = str(Decimal(action["amount"]) / 10)
    trace["checkpoints"][0]["raw_total_pot"] = "0.6"
    final = replay_reviewed_hand(trace)[-1]
    assert final.state.pot.value == Decimal("14")
    assert final.returned == ((0, Decimal("4")),)


def test_settlement_overpayment_needs_explicit_external_cashflow(trace):
    final = replay_reviewed_hand(trace)[-1].state
    trace["source_seats"][0]["settled_stack"] = 500
    with pytest.raises(ValueError, match="exceed"):
        audit_observed_settlement(trace, final)


def test_modified_final_money_is_not_accepted_for_settlement(trace):
    final = replay_reviewed_hand(trace)[-1].state
    with pytest.raises(ValueError, match="conserve"):
        audit_observed_settlement(trace, replace(final, pot=ChipAmount(347)))


def test_late_hero_visibility_is_a_separate_knowledge_step(trace):
    trace["cards"][0]["first_verified_visible_frame"] = 8460
    steps = replay_reviewed_hand(trace)
    reveal = next(step for step in steps if step.kind == "hero_cards_revealed")
    assert reveal.known_by_frame == 8460
    assert steps[0].state.hero_cards == ()
    assert len([s for s in steps if s.action_event is not None]) == 13


def test_binary_float_money_and_future_board_are_rejected(trace):
    trace["rules"]["small_blind"] = 2.0
    with pytest.raises(ValueError, match="floats"):
        replay_reviewed_hand(trace)
    trace["rules"]["small_blind"] = 2
    trace["board"][0]["first_verified_visible_frame"] = 999999
    with pytest.raises(ValueError, match="future board"):
        replay_reviewed_hand(trace)


def test_replay_is_deterministic(trace):
    assert replay_reviewed_hand(trace) == replay_reviewed_hand(trace)


@pytest.mark.parametrize("field", ["raw_total_pot", "hero_stack", "villain_stack"])
def test_visual_money_contradictions_reject_an_otherwise_legal_action_line(
    trace, field,
):
    trace["checkpoints"][1][field] += 1
    with pytest.raises(ValueError, match="contradict"):
        replay_reviewed_hand(trace)


def test_visual_refund_partition_must_match_rules(trace):
    trace["checkpoints"][-2]["displayed_pot_parts"] = [345, 239]
    with pytest.raises(ValueError, match="pot parts"):
        replay_reviewed_hand(trace)


@pytest.mark.parametrize("count", [6, 7, 8])
def test_explicit_utg_straddle_changes_posts_and_action_order(count):
    trace = synthetic_side_pot_hand(count)
    for row in trace["source_seats"]:
        row["starting_stack"] = 400
    trace["rules"].update(observed_straddle=8, straddle_source_seat=2, ante_each=2)
    trace["checkpoints"][0]["raw_total_pot"] = count * 2 + 14
    trace["actions"] = [{"order": 1, "street": "preflop", "source_seat": 3,
                         "kind": "call", "amount": 8, "all_in": False,
                         "amount_semantics": "additional", "evidence": ["a1"]}]
    steps = replay_reviewed_hand(trace)
    assert steps[0].state.actor == 3
    assert steps[0].state.current_bet.value == 8
    assert steps[-1].state.pot.value == count * 2 + 22
    assert steps[-1].action_event.payload["amount_additional"] == "8"
