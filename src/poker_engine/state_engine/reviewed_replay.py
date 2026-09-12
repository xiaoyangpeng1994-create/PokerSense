"""Offline bridge: reviewed actions -> PokerKit legality -> core state snapshots.

This is NOT an OCR adapter or a live authorization path. Source evidence gives
an upper knowledge bound, not exact action time. Forced posts/collections/
returns have distinct ledger entries; only player actions use StateEvent.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from poker_engine.core.enums import (
    ActionType, PlayerStatus, Position, Rank, Street, Suit,
)
from poker_engine.core.events import StateEvent
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.strategy.contracts import DecisionSeat
from poker_engine.strategy.state import calculate_side_pots
from .action_reconstruction import ReconstructionStatus, reconstruct_action_event


def money(value) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError("money must use exact integers/decimal strings, not floats")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("invalid exact monetary value") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("money must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class ReviewedStep:
    kind: str
    state: PokerState
    known_by_frame: int
    evidence: tuple[str, ...]
    action_order: int | None = None
    action_event: StateEvent | None = None
    returned: tuple[tuple[int, Decimal], ...] = ()
    source_kind: str = "reviewed_actions_and_rules"


@dataclass(frozen=True)
class CallProjection:
    actor_seat: int
    additional: Decimal
    eligible_pot_after_call: Decimal
    uncalled_returns: tuple[tuple[int, Decimal], ...]
    break_even_before_fees: Fraction


def decision_seats(state: PokerState) -> tuple[DecisionSeat, ...]:
    return tuple(DecisionSeat(
        seat_id=p.seat, player_id=p.player_id, position=p.position, stack=p.stack,
        street_committed=p.committed_this_street, hand_committed=p.committed_this_hand,
        status=p.status, is_hero=p.is_hero, occupied=True) for p in state.players)


def terminal_call_projection(state: PokerState) -> CallProjection | None:
    """Conditional final call only; no future actions/cards or inferred ranges."""
    active = [p for p in state.players
              if p.status is PlayerStatus.ACTIVE and p.has_cards]
    eligible = [p for p in state.players if p.has_cards
                and p.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)]
    if len(active) != 1 or len(eligible) < 2 or active[0].seat != state.actor:
        return None
    committed = sum((p.committed_this_hand.value for p in state.players), Decimal(0))
    if committed != state.pot.value:
        raise ValueError("call projection requires coherent pot/commitments")
    actor = active[0]
    owed = max(Decimal(0), state.current_bet.value - actor.committed_this_street.value)
    amount = min(actor.stack.value, owed)
    if amount == 0:
        return None
    after = replace(
        actor, stack=ChipAmount(actor.stack.value - amount),
        committed_this_street=ChipAmount(actor.committed_this_street.value + amount),
        committed_this_hand=ChipAmount(actor.committed_this_hand.value + amount),
        status=(PlayerStatus.ALL_IN if actor.stack.value == amount
                else PlayerStatus.ACTIVE))
    projected = replace(state, players=tuple(
        after if p.seat == actor.seat else p for p in state.players),
        pot=ChipAmount(state.pot.value + amount), to_call=ChipAmount(0), actor=None)
    pots = calculate_side_pots(decision_seats(projected), settle_uncalled=True)
    total = sum((p.amount.value for p in pots.pots
                 if actor.seat in p.eligible_seats), Decimal(0))
    if total <= 0:
        raise ValueError("call projection has no contestable pot")
    return CallProjection(actor.seat, amount, total,
                          tuple(sorted((seat, value.value) for seat, value
                                       in pots.uncalled_returns.items())),
                          Fraction(amount) / Fraction(total))


def replay_reviewed_hand(
    trace: dict, *, through_order: int | None = None,
) -> tuple[ReviewedStep, ...]:
    """Public betting-only API; later completion facts are not consumed."""
    return _replay_betting_session(trace, through_order=through_order)[0]


def _replay_betting_session(trace: dict, *, through_order: int | None = None):
    """Replay the betting ledger; unresolved settlement stays outside PokerState.

    V1 supports a reviewed 6-8-seat, per-player-ante, standard single-board hand.
    A single UTG straddle requires an explicit source seat and amount. Other
    positions/variants are rejected rather than silently approximated.
    """
    from pokerkit import Automation, Mode, NoLimitTexasHoldem

    rules = trace["rules"]
    straddle = money(rules["observed_straddle"])
    if rules.get("variant", "standard") != "standard":
        raise ValueError("special variants need a separately reviewed rules adapter")
    if through_order is not None and (
        type(through_order) is not int or through_order < 1
    ):
        raise ValueError("through_order must be a positive integer")
    order = tuple(trace["pokerkit_index_to_source_seat"])
    rows = {p["seat"]: p for p in trace["source_seats"]}
    if (len(rows) != len(trace["source_seats"]) or len(set(order)) != len(order)
            or set(order) != set(rows) or not 6 <= len(order) <= 8
            or rules["player_count"] != len(order)):
        raise ValueError("seat census/order mismatch")
    hero = rules["hero_source_seat"]
    dealer = rules["dealer_source_seat"]
    if hero not in rows or dealer not in rows:
        raise ValueError("hero/dealer not in seat census")
    initial = tuple(money(rows[seat]["starting_stack"]) for seat in order)
    starting_total = sum(initial, Decimal(0))
    sb, bb, ante = (money(rules[k]) for k in ("small_blind", "big_blind", "ante_each"))
    if not 0 < sb <= bb or any(stack <= 0 for stack in initial):
        raise ValueError("unsupported starting stacks/blinds")
    if rows[order[0]]["position"] != "SB" or rows[order[1]]["position"] != "BB":
        raise ValueError("PokerKit ordering must begin with source SB/BB")
    posts, minimum_bet = (sb, bb), bb
    if straddle:
        if rules.get("straddle_source_seat") != order[2] or straddle < bb * 2:
            raise ValueError("straddle needs explicit UTG seat and at least two BB")
        posts, minimum_bet = (sb, bb, straddle), straddle
    state = NoLimitTexasHoldem.create_state(
        (Automation.ANTE_POSTING, Automation.BLIND_OR_STRADDLE_POSTING),
        True, ante, posts, minimum_bet, initial, len(order), mode=Mode.CASH_GAME)
    if state.can_collect_bets():
        state.collect_bets()  # dead antes are not live street commitments
    for _ in order:
        state.deal_hole("????")  # no later revealed opponent cards enter this engine
    board, street, steps = (), Street.PREFLOP, []

    def knowledge(refs):
        if not refs or any(ref not in trace["evidence"] for ref in refs):
            raise ValueError("missing source evidence")
        values = [trace["evidence"][ref]["source_frame"] for ref in refs]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("invalid source frame")
        return max(values)

    def snapshot(frame):
        known = next((row["cards"] for row in trace["cards"] if row["seat"] == hero
                      and row["first_verified_visible_frame"] <= frame), ())
        hero_cards = tuple(Card(Rank(c[0]), Suit(c[1])) for c in known)
        players = []
        for i, seat in enumerate(order):
            stack = money(state.stacks[i])
            status = (PlayerStatus.FOLDED if not state.statuses[i] else
                      PlayerStatus.ALL_IN if stack == 0 else PlayerStatus.ACTIVE)
            players.append(PlayerState(
                f"review-seat-{seat}", seat, Position(rows[seat]["position"]),
                ChipAmount(stack), ChipAmount(state.bets[i]),
                ChipAmount(initial[i] - stack),
                status, bool(state.statuses[i]), seat == hero, seat == dealer))
        actor = state.actor_index
        pot = money(state.total_pot_amount)
        if sum((p.stack.value for p in players), Decimal(0)) + pot != starting_total:
            raise ValueError("chip conservation failed")
        if sum((p.committed_this_hand.value for p in players), Decimal(0)) != pot:
            raise ValueError("commitment/pot reconciliation failed")
        return PokerState(len(steps), trace["hand_id"], street, hero_cards, board,
                          tuple(sorted(players, key=lambda p: p.seat)), ChipAmount(pot),
                          ChipAmount(max(state.bets)),
                          ChipAmount(state.checking_or_calling_amount
                                     if actor is not None else 0),
                          order[actor] if actor is not None else None)

    ready = next(c for c in trace["checkpoints"] if c["stage"] == "forced_posts")
    refs = (ready["evidence"],)
    frame = knowledge(refs)
    steps.append(ReviewedStep("forced_posts", snapshot(frame), frame, refs))
    if steps[-1].state.pot.value != money(ready["raw_total_pot"]):
        raise ValueError("forced posts contradict reviewed pot")
    actions = (trace["actions"] if through_order is None
               else trace["actions"][:through_order])
    if [a["order"] for a in actions] != list(range(1, len(actions) + 1)):
        raise ValueError("action order must be consecutive")
    for action in actions:
        refs = tuple(action["evidence"])
        frame = knowledge(refs)
        if frame < steps[-1].known_by_frame:
            raise ValueError("evidence knowledge order regressed")
        next_street = Street(action["street"])
        if next_street is not street:
            board_row = next(row for row in trace["board"]
                             if row["street"] == next_street.value)
            if board_row["first_verified_visible_frame"] > frame:
                raise ValueError("future board cannot enter an earlier action")
            state.burn_card("??")
            cards = board_row["cards"]
            state.deal_board("".join(cards[len(board):]))
            board = tuple(Card(Rank(c[0]), Suit(c[1])) for c in cards)
            street = next_street
            steps.append(ReviewedStep("street_change", snapshot(frame), frame, refs))
        actor = state.actor_index
        if actor is None or order[actor] != action["source_seat"]:
            raise ValueError("reviewed actor contradicts legal action order")
        before = steps[-1].state
        newly_known = snapshot(frame)
        if newly_known.hero_cards != before.hero_cards:
            steps.append(ReviewedStep("hero_cards_revealed", newly_known,
                                      frame, refs, source_kind="reviewed_visibility"))
            before = newly_known
        amount = money(action["amount"])
        kind = action["kind"]
        semantics = "total_street" if kind in ("bet_to", "raise_to") else "additional"
        if action["amount_semantics"] != semantics:
            raise ValueError("action amount semantics mismatch")
        old_stack = money(state.stacks[actor])
        if kind == "fold":
            if amount != 0:
                raise ValueError("fold cannot spend chips")
            state.fold()
            observed = ActionType.FOLD
        elif kind == "call":
            if amount != money(state.checking_or_calling_amount):
                raise ValueError("reviewed call amount is not legal")
            state.check_or_call()
            observed = ActionType.CALL
        elif kind == "check":
            if amount != 0 or state.checking_or_calling_amount != 0:
                raise ValueError("check cannot spend or face a bet")
            state.check_or_call()
            observed = ActionType.CHECK
        elif kind in ("bet_to", "raise_to"):
            state.complete_bet_or_raise_to(amount)
            observed = ActionType.BET if kind == "bet_to" else ActionType.RAISE
        else:
            raise ValueError("unsupported reviewed action")
        paid = old_stack - money(state.stacks[actor])
        actual_all_in = state.stacks[actor] == 0 and paid > 0
        if type(action["all_in"]) is not bool or action["all_in"] != actual_all_in:
            raise ValueError("reviewed all-in flag contradicts remaining stack")
        current = snapshot(frame)
        # Logical ordering timestamp only, not a recovered wall-clock event time.
        timestamp = (datetime(2000, 1, 1, tzinfo=timezone.utc)
                     + timedelta(microseconds=len(steps)))
        reconstructed = reconstruct_action_event(
            before, current, actor_seat=order[actor], observed_action=observed,
            timestamp=timestamp, source="reviewed_hand_replay")
        if reconstructed.status is not ReconstructionStatus.EXACT:
            raise ValueError(
                f"project action reconciliation failed: {reconstructed.reasons}")
        steps.append(ReviewedStep("action", current, frame, refs, action["order"],
                                  reconstructed.event))
        if state.can_collect_bets():
            old_stacks = tuple(money(value) for value in state.stacks)
            state.collect_bets()  # separate operation; may refund a different player
            returned = tuple(
                (seat, money(state.stacks[i]) - old_stacks[i])
                for i, seat in enumerate(order) if state.stacks[i] > old_stacks[i])
            collected = snapshot(frame)
            calculated = calculate_side_pots(
                decision_seats(current), settle_uncalled=True)
            expected = tuple(sorted((seat, value.value) for seat, value
                                    in calculated.uncalled_returns.items()))
            if tuple(sorted(returned)) != expected:
                raise ValueError("project/PokerKit uncalled-return disagreement")
            project_pots = calculate_side_pots(
                decision_seats(collected), settle_uncalled=True)
            ours = [(p.amount.value, p.eligible_seats) for p in project_pots.pots]
            theirs = [(money(p.unraked_amount), tuple(sorted(
                order[i] for i in p.player_indices))) for p in state.pots]
            if ours != theirs:
                raise ValueError("project/PokerKit side-pot disagreement")
            steps.append(ReviewedStep(
                "uncalled_return" if returned else "bet_collection",
                collected, frame, refs, returned=returned,
                source_kind="rules_derived_not_visual_credit_time"))
    result = tuple(steps)
    verify_reviewed_checkpoints(
        trace, result, through_frame=result[-1].known_by_frame
        if through_order is not None else None)
    return result, state, order


def verify_reviewed_checkpoints(trace: dict, steps: tuple[ReviewedStep, ...],
                                *, through_frame: int | None = None) -> int:
    """Check reviewed logical targets against pre-collection action states.

    Legacy ``raw_total_pot`` is NOT guaranteed to be a same-frame OCR target:
    f8560's visible ribbon is156 while the reviewed post-call total is218.
    Screen-field parity must be checked separately (aq_observation_v1).
    """
    checked = 0
    for point in trace["checkpoints"]:
        if point["stage"] == "settled":
            continue  # settlement cash is audited separately
        frame = trace["evidence"][point["evidence"]]["source_frame"]
        if through_frame is not None and frame > through_frame:
            continue
        candidates = [step for step in steps if step.kind in ("forced_posts", "action")
                      and step.known_by_frame <= frame]
        if not candidates:
            raise ValueError("checkpoint has no preceding reviewed state")
        state = candidates[-1].state
        if ("raw_total_pot" in point
                and money(point["raw_total_pot"]) != state.pot.value):
            raise ValueError("reviewed pot checkpoint contradicts reconstructed state")
        hero = next(p for p in state.players if p.is_hero)
        if "hero_stack" in point and money(point["hero_stack"]) != hero.stack.value:
            raise ValueError("reviewed hero stack contradicts reconstructed state")
        if "villain_stack" in point:
            opponents = [p for p in state.players if p.has_cards and not p.is_hero]
            if len(opponents) != 1:
                raise ValueError("villain checkpoint needs an explicit unique seat")
            if money(point["villain_stack"]) != opponents[0].stack.value:
                raise ValueError("reviewed opponent stack contradicts ledger")
        if "displayed_pot_parts" in point:
            if state.actor is not None:
                raise ValueError("open-round pot parts need separate semantics")
            calculated = calculate_side_pots(decision_seats(state))
            parts = [p.amount.value for p in calculated.pots]
            parts += [value.value for value in calculated.uncalled_returns.values()]
            displayed = sorted(money(value) for value in point["displayed_pot_parts"])
            if sorted(parts) != displayed:
                raise ValueError("displayed pot parts contradict pots/returns")
        checked += 1
    return checked


def audit_observed_settlement(trace: dict, final: PokerState) -> dict:
    """Observe cash conservation without pretending unknown fees are rake."""
    if final.actor is not None:
        raise ValueError("betting is still open")
    if final.hand_id != trace["hand_id"]:
        raise ValueError("settlement belongs to another hand")
    before = {p.seat: p.stack.value for p in final.players}
    observed = {p["seat"]: money(p["settled_stack"]) for p in trace["source_seats"]}
    if set(before) != set(observed) or len(observed) != len(trace["source_seats"]):
        raise ValueError("settlement seat census mismatch")
    initial = sum((money(p["starting_stack"]) for p in trace["source_seats"]),
                  Decimal(0))
    if sum(before.values(), Decimal(0)) + final.pot.value != initial:
        raise ValueError("settlement input does not conserve chips")
    gains = {seat: observed[seat] - amount for seat, amount in before.items()}
    if any(amount < 0 for amount in gains.values()):
        raise ValueError("unmodeled post-hand debit requires separate evidence")
    awarded = sum(gains.values(), Decimal(0))
    unresolved = final.pot.value - awarded
    if unresolved < 0:
        raise ValueError("observed awards exceed contestable pot")
    insurance_notice = trace.get("settlement", {}).get("insurance_premium_displayed")
    return {"status": "PARTIAL", "observed_positive_stack_deltas": gains,
            "delta_basis": ("observed settled stacks minus "
                            "rules-derived refund balances"),
            "contestable_pot": final.pot.value, "unallocated_outflow": unresolved,
            "insurance_notice_not_cash_entry": money(insurance_notice)
            if insurance_notice is not None else None,
            "rake_confirmed": False, "insurance_debit_confirmed": False,
            "capture_to_state_verified": False}
