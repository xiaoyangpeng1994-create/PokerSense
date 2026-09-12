"""Candidate hand commitments and preflop action line for AA8 shadow logs."""

from decimal import Decimal, InvalidOperation


def _amount(value):
    if not isinstance(value, str):
        return None
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def candidate_action_line(actions, epoch, street):
    result = {
        "value": None,
        "reason": "postflop_action_line_not_released",
        "source_action_frames": [],
        "complete_and_canonical_verified": False,
        "strategy_eligible": False,
    }
    if not isinstance(epoch, str) or not epoch:
        return {**result, "reason": "hand_epoch_unknown"}
    if street != "preflop":
        return result
    selected = [
        action for action in actions
        if action.get("epoch") == epoch and action.get("street") == "preflop"
    ]
    sequence = []
    previous_frame = None
    for action in selected:
        kind = action.get("kind")
        seat = action.get("slot")
        frame = action.get("frame")
        if (kind not in ("fold", "check", "call", "aggressive", "all_in")
                or type(seat) is not int or not 0 <= seat < 8
                or type(frame) is not int
                or previous_frame is not None and frame <= previous_frame):
            return {**result, "reason": "invalid_preflop_action_sequence"}
        sequence.append((seat, kind))
        previous_frame = frame
    if any(kind == "all_in" for _, kind in sequence):
        value = "all_in"
    else:
        aggressive = [
            index for index, (_, kind) in enumerate(sequence)
            if kind == "aggressive"
        ]
        if not aggressive:
            limps = sum(kind == "call" for _, kind in sequence)
            value = "unopened" if limps == 0 else (
                "limp" if limps == 1 else "multi_limp"
            )
        elif len(aggressive) == 1:
            before = sequence[:aggressive[0]]
            value = "iso_raise" if any(
                kind == "call" for _, kind in before
            ) else "raise"
        elif len(aggressive) == 2:
            between = sequence[aggressive[0] + 1:aggressive[1]]
            value = "squeeze" if any(
                kind == "call" for _, kind in between
            ) else "three_bet"
        else:
            value = "four_bet"
    return {
        **result,
        "value": value,
        "reason": "observed_actions_candidate_not_complete_history",
        "source_action_frames": [action["frame"] for action in selected],
    }


class AA8HandLedgerCandidate:
    def __init__(self):
        self.last = None
        self.epoch = None
        self.totals = None
        self.applied = set()
        self.opening = None
        self.tainted = False
        self.taint_reasons = []

    def _reset(self, epoch, adapter):
        self.epoch = epoch
        self.totals = None
        self.applied.clear()
        self.opening = None
        self.tainted = False
        self.taint_reasons = []
        if not isinstance(epoch, str) or not epoch:
            return
        events = [
            event for event in adapter.epoch_events if event.get("epoch") == epoch
        ]
        if len(events) != 1:
            self.taint_reasons.append("unique_opening_event_missing")
            return
        event = events[0]
        posting = event.get("posting_comparison") or {}
        debits = posting.get("debits")
        if (event.get("status") != "MULTI_POST_DEAL_CANDIDATE"
                or not isinstance(debits, dict) or len(debits) < 3):
            self.taint_reasons.append("opening_post_vector_incomplete")
            return
        totals = {str(seat): Decimal("0") for seat in range(8)}
        for seat, value in debits.items():
            amount = _amount(value)
            if seat not in totals or amount is None:
                self.taint_reasons.append("opening_post_amount_invalid")
                return
            totals[seat] = amount
        self.totals = totals
        self.opening = {
            "frame": event["frame"],
            "first_frame": event.get("first_frame"),
            "debits": dict(debits),
            "excluded_na_slots": posting.get("excluded_na_slots") or [],
            "authoritative_boundary": event.get("authoritative_boundary") is True,
        }

    def observe(self, row, adapter):
        frame = row["frame"]
        if (type(frame) is not int or frame < 0
                or self.last is not None and frame <= self.last):
            raise ValueError("strictly increasing source frames required")
        gap = self.last is not None and frame != self.last + 1
        self.last = frame
        epoch = (row.get("observed_state_v2") or {}).get("observed_epoch")
        if gap or epoch != self.epoch:
            self._reset(epoch, adapter)
            if gap:
                self.tainted = True
                self.taint_reasons.append("source_frame_gap")
        modes = row.get("special_modes") or {}
        if modes.get("block_state_updates") or modes.get("insurance") == "VISIBLE":
            self.tainted = True
            self.taint_reasons.append("special_mode_or_overlay_observed")
        if self.totals is None:
            return self.snapshot(row, "HAND_COMMITMENTS_UNKNOWN")
        for action in adapter.actions:
            if action.get("epoch") != self.epoch:
                continue
            identity = (action.get("frame"), action.get("slot"), action.get("kind"))
            if identity in self.applied:
                continue
            amount = _amount(action.get("amount"))
            seat = str(action.get("slot"))
            action_frame = action.get("frame")
            if (amount is None or seat not in self.totals
                    or type(action_frame) is not int or action_frame > frame):
                self.tainted = True
                self.taint_reasons.append("action_amount_or_seat_unknown")
                continue
            self.totals[seat] += amount
            self.applied.add(identity)
        status = (
            "HAND_COMMITMENTS_SUSPENDED" if self.tainted
            else "OBSERVED_HAND_COMMITMENTS_CANDIDATE"
        )
        return self.snapshot(row, status)

    def snapshot(self, row, status):
        pot = _amount((row.get("pot") or {}).get("value"))
        total = sum(self.totals.values(), Decimal("0")) if self.totals else None
        difference = str(total - pot) if total is not None and pot is not None else None
        return {
            "frame": row["frame"],
            "epoch": self.epoch,
            "status": status,
            "hand_commitments": {
                seat: str(value) for seat, value in self.totals.items()
            } if self.totals is not None else None,
            "observed_total": str(total) if total is not None else None,
            "displayed_pot": str(pot) if pot is not None else None,
            "unallocated_difference": difference,
            "difference_semantics": (
                "OBSERVED_COMMITMENTS_MINUS_DISPLAYED_POT_NOT_RAKE_OR_FEE"
            ),
            "opening_evidence": self.opening,
            "applied_action_count": len(self.applied),
            "taint_reasons": list(dict.fromkeys(self.taint_reasons)),
            "complete_and_canonical_verified": False,
            "strategy_eligible": False,
        }


__all__ = ["AA8HandLedgerCandidate", "candidate_action_line"]
