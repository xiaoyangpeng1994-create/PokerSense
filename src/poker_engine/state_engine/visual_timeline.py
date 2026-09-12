"""Causal visual evidence timeline, not authoritative betting StateEvents.

Pairs current/previous observed badge and stack changes in a bounded window.
No reviewed action line, future frame, room rule or final payout enters consume.
"""

from collections import deque
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TimelineEvidence:
    kind: str
    first_frame: int
    confirmed_frame: int
    street: str | None = None
    seat: int | None = None
    action: str | None = None
    amount: str | None = None
    balance_before: str | None = None
    balance_after: str | None = None
    note: str = "observational_candidate_not_state_event"


class VisualTimeline:
    def __init__(self, *, confirmations=2, pair_window=12, slot_count=8, hero_slot=0):
        if type(confirmations) is not int or confirmations < 2:
            raise ValueError("at least two distinct source frames required")
        if type(pair_window) is not int or pair_window < confirmations:
            raise ValueError("invalid pairing window")
        if type(slot_count) is not int or not 2 <= slot_count <= 10:
            raise ValueError("slot_count must describe 2 to 10 physical slots")
        if type(hero_slot) is not int or not 0 <= hero_slot < slot_count:
            raise ValueError("hero_slot outside configured layout")
        self.slot_count, self.hero_slot = slot_count, hero_slot
        self.confirmations, self.pair_window = confirmations, pair_window
        self.events = []
        self.roster = None
        self.folded, self.departed = set(), set()
        self.street = None
        self.last_frame = None
        self._streaks, self._balances, self._presence = {}, {}, {}
        self._badges, self._pending = {}, {}
        self._debits = {seat: deque(maxlen=16) for seat in range(self.slot_count)}
        self._consumed_debits, self._emitted_keys = set(), set()
        self._in_gap = False

    def _confirm(self, key, value, frame):
        if value is None:
            self._streaks.pop(key, None)
            return None
        old = self._streaks.get(key)
        if old is None or old[0] != value or old[2] != frame - 1:
            row = (value, frame, frame, 1)
        else:
            row = (value, old[1], frame, old[3] + 1)
        self._streaks[key] = row
        return (row[1], value) if row[3] >= self.confirmations else None

    def _append(self, kind, first, frame, **kwargs):
        street = kwargs.pop("street", self.street)
        self.events.append(TimelineEvidence(kind, first, frame, street, **kwargs))

    def _clear_temporary(self, frame):
        for seat, pending in self._pending.items():
            self._append("unresolved_action", pending["first"], frame, seat=seat,
                         action=pending["action"], note="visual_gap_before_pairing")
        self._pending.clear()
        self._streaks.clear()
        self._balances.clear()
        self._badges.clear()
        for values in self._debits.values():
            values.clear()

    @staticmethod
    def _value(field):
        return field.get("value") if field.get("status") == "valid" else None

    def consume(self, row):
        frame = row["source_frame"]
        if type(frame) is not int or frame < 0:
            raise ValueError("invalid source frame")
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("duplicate/backwards frame cannot accumulate evidence")
        allowed = {str(seat) for seat in range(self.slot_count)}
        for field in ("stacks", "seat_presence", "actions"):
            values = row.get(field, {})
            if not isinstance(values, dict) or not set(values).issubset(allowed):
                raise ValueError("visual slots outside configured layout")
        skipped = self.last_frame is not None and frame != self.last_frame + 1
        supported = row.get("scene_reason") == "supported_layout_candidate"
        if skipped or not supported:
            if not self._in_gap:
                self._append("visual_gap", frame, frame)
            self._clear_temporary(frame)
            self._in_gap = True
        self.last_frame = frame
        if not supported:
            return
        self._in_gap = False

        raw_street = self._value(row.get("street_observation", {}))
        streets = ("preflop", "flop", "turn", "river")
        stage = raw_street if raw_street in streets else None
        ready = self._confirm("street", stage, frame)
        if ready and ready[1] != self.street:
            advance = (self.street is None
                       or streets.index(ready[1]) > streets.index(self.street))
            if advance:
                self.street = ready[1]
                self._append("street_observed", ready[0], frame)
                self._badges.clear()
            # A clearing board alone must not start an invented new hand.

        for seat in range(self.slot_count):
            field = row.get("stacks", {}).get(str(seat), {})
            value = self._value(field)
            amount = Decimal(str(value)) if value is not None else None
            if amount is not None and (not amount.is_finite() or amount < 0):
                raise ValueError("invalid observed balance")
            ready = self._confirm(("stack", seat), amount, frame)
            if ready:
                old = self._balances.get(seat)
                if old and old[0] != amount:
                    if frame - old[1] <= self.pair_window and not skipped:
                        change = amount - old[0]
                        kind = "balance_decrease" if change < 0 else "balance_increase"
                        self._append(kind, ready[0], frame, seat=seat,
                                     amount=str(abs(change)),
                                     balance_before=str(old[0]),
                                     balance_after=str(amount))
                        if change < 0:
                            self._debits[seat].append({
                                "first": ready[0], "confirmed": frame,
                                "before": old[0],
                                "after": amount, "street": self.street})
                    else:
                        self._append("balance_gap", ready[0], frame, seat=seat,
                                     balance_before=str(old[0]),
                                     balance_after=str(amount))
                self._balances[seat] = (amount, frame)

            present = self._value(row.get("seat_presence", {}).get(str(seat), {}))
            if present is not None and type(present) is not bool:
                raise ValueError("presence must be an observed boolean")
            ready = self._confirm(("presence", seat), present, frame)
            if ready:
                previous = self._presence.get(seat)
                self._presence[seat] = present
                changed = previous is not None and previous != present
                if self.roster is not None and changed:
                    if not present and seat in self.roster:
                        self.departed.add(seat)
                        self._append("opening_seat_became_empty", ready[0], frame,
                                     seat=seat, note="does_not_infer_a_turn_fold")
                    elif present and seat not in self.roster:
                        self._append("nonopening_seat_appeared", ready[0], frame,
                                     seat=seat, note="not_added_to_opening_hand")

        cards = row.get("hero", [])
        complete_pair = len(cards) == 2 and all(cards) and cards[0] != cards[1]
        pair = tuple(cards) if complete_pair else None
        ready = self._confirm("hero_pair", pair, frame)
        if self.roster is None and ready and len(self._presence) == self.slot_count:
            self.roster = tuple(seat for seat in range(self.slot_count)
                                if self._presence[seat])
            self._append("opening_roster_observed", ready[0], frame,
                         note=",".join(map(str, self.roster)))

        for seat in range(self.slot_count):
            if self.roster is None or seat not in self.roster or seat in self.departed:
                continue
            action = row.get("actions", {}).get(str(seat), {})
            label = action.get("value") if action.get("accepted_candidate") else None
            if label not in ("fold", "check", "call", "bet", "raise", "all_in"):
                label = None
            if label is None:
                self._confirm(("badge", seat), None, frame)
                self._badges[seat] = None
                continue
            ready = self._confirm(("badge", seat), label, frame)
            if ready and self._badges.get(seat) != label and self.street is not None:
                self._badges[seat] = label
                key = (self.street, seat, label)
                if seat in self.folded:
                    continue
                if label in ("fold", "check"):
                    if key not in self._emitted_keys:
                        self._append("action_evidence", ready[0], frame,
                                     seat=seat, action=label, amount="0")
                        self._emitted_keys.add(key)
                        if label == "fold":
                            self.folded.add(seat)
                elif seat not in self._pending:
                    self._pending[seat] = {"first": ready[0], "action": label,
                                           "street": self.street, "key": key}
        self._pair_actions(frame)

    def _pair_actions(self, frame):
        for seat, pending in list(self._pending.items()):
            candidates = [d for d in self._debits[seat]
                          if (seat, d["first"]) not in self._consumed_debits
                          and abs(d["first"] - pending["first"]) <= self.pair_window
                          and d["street"] == pending["street"]]
            if len(candidates) == 1:
                debit = candidates[0]
                if pending["key"] in self._emitted_keys:
                    self._append("unresolved_action", pending["first"], frame,
                                 seat=seat, action=pending["action"],
                                 note="repeat_badge_new_debit_requires_betting_context")
                    self._consumed_debits.add((seat, debit["first"]))
                    del self._pending[seat]
                    continue
                self._append("action_evidence", pending["first"], frame, seat=seat,
                             street=pending["street"],
                             action=pending["action"],
                             amount=str(debit["before"] - debit["after"]),
                             balance_before=str(debit["before"]),
                             balance_after=str(debit["after"]))
                self._consumed_debits.add((seat, debit["first"]))
                self._emitted_keys.add(pending["key"])
                del self._pending[seat]
            elif len(candidates) > 1 or frame - pending["first"] > self.pair_window:
                if pending["key"] not in self._emitted_keys:
                    self._append("unresolved_action", pending["first"], frame,
                                 seat=seat, action=pending["action"],
                                 note="no_unique_nearby_debit")
                del self._pending[seat]

    def summary(self):
        return {"opening_roster": self.roster, "folded": sorted(self.folded),
                "departed": sorted(self.departed), "last_street": self.street,
                "pending_actions": len(self._pending),
                "hero_action_eligible": (False if self.hero_slot in self.folded
                                         else None),
                "canonical_state_events_verified": False, "release_eligible": False}
