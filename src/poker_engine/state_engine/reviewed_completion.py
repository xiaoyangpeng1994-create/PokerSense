"""Reviewed single-board showdown, gross award and separate cash observations.

Never called by the live capture pipeline. Cash differences remain unallocated;
an insurance notice is not a transaction. Earlier betting replay is unchanged.
"""

from dataclasses import dataclass, replace
from decimal import Decimal

from poker_engine.core.enums import PlayerStatus, Rank, Street, Suit
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity.evaluator import evaluate
from poker_engine.strategy.state import calculate_side_pots
from .reviewed_replay import (
    ReviewedStep, _replay_betting_session, decision_seats, money,
)


@dataclass(frozen=True)
class CompletionStep:
    kind: str
    state: PokerState
    known_by_frame: int
    evidence: tuple[str, ...]
    source_kind: str
    revealed_holes: tuple[tuple[int, tuple[Card, ...]], ...] = ()
    gross_awards: tuple[tuple[int, Decimal], ...] = ()
    visible_balances: tuple[tuple[int, Decimal], ...] = ()
    cash_observed_seats: tuple[int, ...] = ()
    unallocated_outflows: tuple[tuple[int, Decimal], ...] = ()
    insurance_notice: tuple[str, Decimal | None] | None = None


@dataclass(frozen=True)
class CompletedReviewedHand:
    betting_steps: tuple[ReviewedStep, ...]
    completion_steps: tuple[CompletionStep, ...]
    gross_awards: tuple[tuple[int, Decimal], ...]
    unallocated_outflows: tuple[tuple[int, Decimal], ...]
    all_cash_balances_observed: bool
    pokerkit_hand_ended: bool
    fee_components_verified: bool = False
    capture_to_state_verified: bool = False


def independent_awards(base: PokerState, holes: dict, board: tuple,
                       chip_unit: Decimal) -> dict[int, Decimal]:
    """Cross-check pot eligibility/winners using the project's five-card evaluator.

    Odd-chip allocation is intentionally unsupported without a room-specific
    policy. Do not silently pick a winner for the remainder.
    """
    if len(board) != 5:
        raise ValueError("showdown requires a complete board")
    known = list(board) + [card for values in holes.values() for card in values]
    if len(set(known)) != len(known):
        raise ValueError("duplicate revealed cards")
    pots = calculate_side_pots(decision_seats(base))
    result = {p.seat: Decimal(0) for p in base.players}
    for pot in pots.pots:
        if any(seat not in holes for seat in pot.eligible_seats):
            raise ValueError("unknown contender cards cannot produce a winner")
        strength = {seat: evaluate(holes[seat] + board) for seat in pot.eligible_seats}
        best = max(strength.values())
        winners = [seat for seat, value in strength.items() if value == best]
        units = pot.amount.value / chip_unit
        if units % 1 or units % len(winners):
            raise ValueError("odd-chip allocation requires a supported room rule")
        for seat in winners:
            result[seat] += pot.amount.value / len(winners)
    return result


def complete_reviewed_hand(trace: dict, review: dict) -> CompletedReviewedHand:
    if review["hand_id"] != trace["hand_id"]:
        raise ValueError("completion review belongs to another hand")
    if type(review["runout_count"]) is not int or review["runout_count"] != 1:
        raise ValueError("only explicitly reviewed single-board runouts supported")
    if review.get("single_board_reviewed") is not True:
        raise ValueError("single-board review is required")
    chip_unit = money(review["chip_unit"])
    if chip_unit <= 0:
        raise ValueError("chip unit must be positive")
    betting, engine, order = _replay_betting_session(trace)
    base = betting[-1].state
    if base.actor is not None:
        raise ValueError("betting has not closed")
    contenders = {p.seat for p in base.players if p.has_cards}
    if len(contenders) < 2:
        raise ValueError("this completion adapter requires a contested showdown")
    total = sum((p.stack.value for p in base.players), base.pot.value)
    steps, holes, observed, differences = [], {}, {}, {}
    board, street = base.board_cards, base.street
    gross = {seat: Decimal(0) for seat in order}
    settled = False

    def frame_of(ref):
        if ref not in review["evidence"]:
            raise ValueError("completion evidence is missing")
        frame = review["evidence"][ref]["source_frame"]
        if type(frame) is not int or frame < 0:
            raise ValueError("invalid completion source frame")
        return frame

    cutoff = frame_of(review["next_hand_first_post_evidence"])
    interval = trace.get("next_hand_start_interval")
    if interval and not interval[0] <= cutoff <= interval[1]:
        raise ValueError("next-hand cutoff contradicts reviewed interval")

    def append(kind, frame, ref, source_kind, awards=(), balances=(), notice=None):
        players = []
        for player in base.players:
            gross_balance = money(engine.stacks[order.index(player.seat)])
            amount = observed.get(player.seat, gross_balance)
            status = (PlayerStatus.FOLDED if not player.has_cards else
                      PlayerStatus.ALL_IN if amount == 0 else PlayerStatus.ACTIVE)
            players.append(replace(player, stack=ChipAmount(amount), status=status,
                                   committed_this_street=ChipAmount(0)))
        pot = money(engine.total_pot_amount)
        accounted = (sum((p.stack.value for p in players), pot)
                     + sum(differences.values()))
        if accounted != total:
            raise ValueError("completion cash conservation failed")
        snapshot = replace(base, state_version=base.state_version + len(steps) + 1,
                           players=tuple(players), board_cards=board, street=street,
                           pot=ChipAmount(pot), current_bet=ChipAmount(0),
                           to_call=ChipAmount(0), actor=None)
        if frame < (steps[-1].known_by_frame if steps else betting[-1].known_by_frame):
            raise ValueError("completion knowledge time regressed")
        steps.append(CompletionStep(
            kind, snapshot, frame, (ref,), source_kind,
            tuple(sorted(holes.items())), tuple(awards), tuple(balances),
            tuple(sorted(observed)), tuple(sorted(differences.items())), notice))

    pending = []
    for seat in sorted(contenders):
        cards = [row for row in trace["cards"] if row["seat"] == seat]
        if len(cards) != 1:
            raise ValueError("each contender needs one reviewed hole-card record")
        ref = review["reveal_evidence"][str(seat)]
        frame = frame_of(ref)
        if frame < cards[0]["first_verified_visible_frame"]:
            raise ValueError("later hole cards cannot be revealed earlier")
        pending.append((frame, 0, "reveal", (seat, cards[0]["cards"], ref)))
    for row in trace["board"]:
        if len(row["cards"]) > len(base.board_cards):
            ref = review["board_evidence"][row["street"]]
            frame = frame_of(ref)
            if frame < row["first_verified_visible_frame"]:
                raise ValueError("future board cannot be used early")
            pending.append((frame, 1, "board", (row, ref)))
    for notice in review.get("insurance_notices", []):
        pending.append((frame_of(notice["evidence"]), 2, "notice", notice))
    for observation in review["cash_observations"]:
        frame = frame_of(observation["evidence"])
        if frame >= cutoff:
            raise ValueError("next-hand posts cannot be booked as this hand's fees")
        pending.append((frame, 3, "cash", observation))
    pending.sort(key=lambda entry: entry[:2])
    while engine.can_select_runout_count():
        engine.select_runout_count(1)  # offline reviewed rule, not a live user choice
    for frame, _, kind, item in pending:
        if frame < betting[-1].known_by_frame or frame >= cutoff:
            raise ValueError("completion event outside this hand's evidence window")
        if kind == "reveal":
            seat, codes, ref = item
            if len(codes) != 2:
                raise ValueError("two known cards required at showdown")
            cards = tuple(Card(Rank(c[0]), Suit(c[1])) for c in codes)
            prior_cards = list(board) + [c for values in holes.values() for c in values]
            hero_seat = trace["rules"]["hero_source_seat"]
            if seat != hero_seat and hero_seat not in holes:
                prior_cards.extend(base.hero_cards)
            if len(set(cards)) != 2 or set(cards) & set(prior_cards):
                raise ValueError("duplicate revealed cards")
            engine.show_or_muck_hole_cards("".join(codes), order.index(seat))
            holes[seat] = cards
            append("hole_cards_shown", frame, ref, "reviewed_visibility")
        elif kind == "board":
            row, ref = item
            cards = tuple(Card(Rank(c[0]), Suit(c[1])) for c in row["cards"])
            if cards[:len(board)] != board:
                raise ValueError("runout changed existing board cards")
            if len(set(cards)) != len(cards) or set(cards) & {
                c for values in holes.values() for c in values
            }:
                raise ValueError("duplicate board/revealed cards")
            engine.burn_card("??")
            engine.deal_board("".join(row["cards"][len(board):]))
            board, street = cards, Street(row["street"])
            append("runout_dealt", frame, ref, "reviewed_visibility")
        elif kind == "notice":
            ref = item["evidence"]
            amount = money(item["amount"]) if item.get("amount") is not None else None
            append("insurance_notice", frame, ref, "notice_not_cash_transaction",
                   notice=(item["status"], amount))
        else:
            if not settled:
                raise ValueError("cash observation precedes a resolved showdown")
            ref = item["evidence"]
            balances = {int(seat): money(amount)
                        for seat, amount in item["balances"].items()}
            if (len(balances) != len(item["balances"])
                    or not set(balances) <= set(order)):
                raise ValueError("cash observation seat mismatch")
            for seat, value in balances.items():
                expected = money(engine.stacks[order.index(seat)])
                if value > expected:
                    raise ValueError("external cash inflow needs separate evidence")
                observed[seat] = value
                if value == expected:
                    differences.pop(seat, None)
                else:
                    differences[seat] = expected - value
            append("cash_observed", frame, ref, "observed_balances",
                   balances=tuple(sorted(balances.items())))
        if not settled and len(board) == 5 and set(holes) == contenders:
            expected = independent_awards(base, holes, board, chip_unit)
            while engine.can_kill_hand():
                engine.kill_hand()
            street = Street.SHOWDOWN
            append("showdown_resolved", frame, ref, "rules_and_independent_evaluator")
            while engine.can_push_chips():
                engine.push_chips()
            while engine.can_pull_chips():
                operation = engine.pull_chips()
                seat, amount = order[operation.player_index], money(operation.amount)
                gross[seat] += amount
                append("gross_award", frame, ref, "rules_gross_not_observed_cash",
                       awards=((seat, amount),))
            if gross != expected or engine.status:
                raise ValueError("PokerKit/project gross-award disagreement")
            settled = True
    if not settled or not steps:
        raise ValueError("incomplete runout/showdown")
    append("reviewed_hand_end", steps[-1].known_by_frame, steps[-1].evidence[0],
           "review_completion_not_live_authorization")
    return CompletedReviewedHand(
        betting, tuple(steps), tuple(sorted(gross.items())),
        tuple(sorted(differences.items())), set(observed) == set(order),
        not engine.status)
