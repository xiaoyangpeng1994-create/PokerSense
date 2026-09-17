"""Causal interpretation of AA observations, never canonical state or Advice."""

from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re


def amount(value):
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        number = Decimal(value)
        return number if number.is_finite() and number >= 0 else None
    except InvalidOperation:
        return None


def blocked(row):
    modes = row.get("special_modes") or {}
    return (row.get("scene_supported") is not True
            or modes.get("block_state_updates") is True
            or modes.get("insurance") == "VISIBLE")


def identity(action):
    return tuple(action.get(k) for k in ("epoch", "frame", "slot", "kind", "amount"))


class AAObservationSemantics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = OrderedDict()
        self.actions = OrderedDict()
        self.source = self.last = self.epoch = None
        self.terminal_frame = self.last_ledger = None
        self.clear_streak = 0
        self.waiting = False

    def _interpret(self, action, row):
        result = deepcopy(action)
        confirmed = action.get("confirmed_at", action.get("frame"))
        observation = self.history.get(confirmed)
        result.update(semantic_kind=None, all_in=action.get("glyph") == "all_in",
                      semantic_street=None,
                      amount_semantics="additional_debit", target_total=None,
                      interpretation_status="UNKNOWN", interpretation_reason=None,
                      source_confirmation_frame=(observation or {}).get("source_frame"),
                      legal_action_verified=False, strategy_eligible=False)

        def reject(reason):
            result["interpretation_reason"] = reason
            return result

        state = row.get("observed_state_v2") or {}
        if (blocked(row) or action.get("epoch") is None
                or action.get("epoch") != state.get("observed_epoch")
                or type(confirmed) is not int or confirmed > row["frame"]):
            return reject("action_context_not_current")
        glyph = action.get("glyph")
        debit = amount(action.get("amount"))
        if glyph in ("fold", "check"):
            if debit != 0:
                return reject("zero_cost_label_has_debit")
            result.update(semantic_kind=glyph, all_in=False,
                          interpretation_status="VISIBLE_LABEL_CANDIDATE",
                          interpretation_reason="legality_not_verified")
            if action.get("street") == state.get("street_candidate"):
                result["semantic_street"] = action.get("street")
            else:
                result["interpretation_reason"] = "street_unresolved_at_confirmation"
            return result
        if glyph not in ("aggressive", "all_in", "call"):
            return reject("unsupported_visible_label")
        if (action.get("street") not in ("preflop", "flop", "turn", "river")
                or action["street"] != state.get("street_candidate")):
            return reject("action_street_unverified_at_confirmation")
        cash = action.get("cash_evidence") or {}
        first = cash.get("first_frame")
        cash_confirmed = cash.get("confirmed_frame")
        if (type(first) is not int or first >= confirmed + 1
                or type(cash_confirmed) is not int
                or not first <= cash_confirmed <= confirmed
                or cash.get("kind") != "balance_decrease"
                or cash.get("seat") != action.get("slot")
                or cash.get("street") != action.get("street") or debit is None):
            return reject("cash_onset_or_identity_missing")
        pre = self.history.get(first - 1)
        if (pre is None or pre["epoch"] != action.get("epoch")
                or pre["street"] != action.get("street") or pre["blocked"]):
            return reject("pre_debit_context_missing")
        interval = [self.history.get(f) for f in range(first, confirmed + 1)]
        if any(s is None or s["blocked"] or s["epoch"] != action["epoch"]
               or s["street"] != action["street"] for s in interval):
            return reject("intervening_action_context_invalid")
        wagers = pre["wagers"]
        if (wagers.get("status") != "OBSERVED_STREET_WAGERS_CANDIDATE"
                or wagers.get("title_center_ledger_reconciled") is not True):
            return reject("pre_debit_price_unknown")
        vector = wagers.get("wagers")
        if not isinstance(vector, dict) or set(vector) != set(map(str, range(8))):
            return reject("pre_debit_wagers_incomplete")
        values = []
        for value in vector.values():
            if value == {"status": "NOT_APPLICABLE"}:
                continue
            number = amount(value)
            if number is None:
                return reject("pre_debit_wagers_incomplete")
            values.append(number)
        seat = str(action.get("slot"))
        own = amount(vector.get(seat))
        price = amount(wagers.get("street_price"))
        before = amount(cash.get("balance_before"))
        after = amount(cash.get("balance_after"))
        observed_after = amount((row.get("stacks") or {}).get(seat))
        if (not values or price != max(values) or own is None or price < own
                or before is None or after is None or before - after != debit
                or amount(cash.get("amount")) != debit
                or amount(pre["stacks"].get(seat)) != before
                or observed_after != after or debit <= 0):
            return reject("price_or_cash_contradiction")
        all_in = after == 0
        if glyph == "all_in" and not all_in:
            return reject("all_in_without_zero_balance")
        owed, target = price - own, own + debit
        if target > price:
            kind = "bet" if price == 0 else "raise"
        elif owed > 0 and (debit == owed or all_in and debit < owed):
            kind = "call"
        else:
            return reject("debit_does_not_resolve_action")
        if ((glyph == "call" and kind != "call")
                or (glyph == "aggressive" and kind == "call")):
            return reject("visible_label_conflicts_with_price")
        result.update(
            semantic_kind=kind, semantic_street=action["street"],
            all_in=all_in, target_total=str(target),
            call_cost_before=str(owed),
            short_all_in_call=kind == "call" and debit < owed,
            interpretation_status="PRICE_DERIVED_CANDIDATE",
            interpretation_reason=(
                "full_history_raise_reopening_and_rules_not_verified"),
            predecision_evidence={"processed_frame": first - 1,
                                  "source_frame": pre["source_frame"],
                                  "source_sha256": pre["hash"],
                                  "street_price": str(price), "own_wager": str(own)},
        )
        return result

    def _phase(self, row):
        state = row.get("observed_state_v2") or {}
        epoch = state.get("observed_epoch")
        ledger = row.get("hand_ledger_v2") or {}
        if epoch != self.epoch:
            self.epoch = epoch
            self.terminal_frame = self.last_ledger = None
            self.clear_streak = 0
            self.waiting = False
        base = {"epoch": epoch, "canonical_verified": False,
                "strategy_eligible": False, "settlement_rules_verified": False,
                "historical_ledger": deepcopy(self.last_ledger),
                "current_ledger": deepcopy(ledger), "phase": "OBSERVING"}
        if epoch is None or blocked(row):
            self.clear_streak = 0
            self.terminal_frame = None
            self.waiting = False
            base.update(phase="SUSPENDED" if blocked(row) else "WAITING_OPENING",
                        current_ledger=None)
            return base
        participants = state.get("participants") or {}
        opening = (ledger.get("opening_evidence") or {}).get("debits") or {}
        states = [participants.get(s, {}) for s in opening]
        board = (row.get("cards") or {}).get("board_slots")
        full_river = (state.get("street_candidate") == "river"
                      and isinstance(board, list) and len(board) == 5
                      and all(isinstance(c, str) and re.fullmatch(
                          r"[2-9TJQKA][cdhs]", c) for c in board)
                      and len(set(board)) == 5)
        closed = (len(states) >= 2
                  and all(p.get("epoch") == epoch and p.get("state") in (
                      "all_in", "folded") for p in states)
                  and sum(p.get("state") == "all_in" for p in states) >= 2
                  and state.get("pending_actions") == 0
                  and row.get("current_actor") is None)
        if self.terminal_frame is not None and not closed:
            self.terminal_frame = None
            self.waiting = False
            self.clear_streak = 0
            base.update(phase="SUSPENDED", current_ledger=None)
            return base
        terminal = closed and full_river
        if terminal and self.terminal_frame is None:
            self.terminal_frame = row["frame"]
        credits = []
        for credit in state.get("unallocated_positive_cash", []):
            confirmed = credit.get("confirmed_frame")
            if (self.terminal_frame is not None and credit.get("epoch") == epoch
                    and type(confirmed) is int
                    and self.terminal_frame < confirmed <= row["frame"]
                    and amount(credit.get("amount")) not in (None, Decimal(0))):
                credits.append(deepcopy(credit))
        pot = amount(row.get("pot"))
        if pot is not None and pot > 0 and not self.waiting:
            self.last_ledger = deepcopy(ledger)
        base["historical_ledger"] = deepcopy(self.last_ledger)
        clear = (credits and pot == 0 and row.get("current_actor") is None
                 and (row.get("cards") or {}).get("hero") is None
                 and board == [None] * 5
                 and ((row.get("hand_transition") or {}).get("center_deal") or {}).get(
                     "visible") is not True)
        self.clear_streak = self.clear_streak + 1 if clear else 0
        if self.clear_streak >= 2:
            self.waiting = True
        if self.waiting or self.clear_streak:
            base.update(phase="WAITING_NEXT_HAND_CANDIDATE" if self.waiting
                        else "POT_CLEAR_PENDING", current_ledger=None)
        elif credits:
            base["phase"] = "SETTLEMENT_CANDIDATE"
        base.update(terminal_observation_frame=self.terminal_frame,
                    visible_credits=credits,
                    credit_semantics="UNALLOCATED_NOT_RAKE_OR_PROFIT")
        return base

    def observe(self, row):
        frame, source = row["frame"], row.get("source_id")
        if (type(frame) is not int or frame < 0 or not isinstance(source, str)
                or not source):
            raise ValueError("semantic_observation_identity_required")
        if self.last is not None and (source != self.source or frame != self.last + 1):
            self.reset()
        self.source, self.last = source, frame
        state = row.get("observed_state_v2") or {}
        self.history[frame] = {
            "epoch": state.get("observed_epoch"),
            "street": state.get("street_candidate"),
            "source_frame": row.get("source_frame"), "hash": row.get("source_sha256"),
            "blocked": blocked(row),
            "wagers": deepcopy(row.get("causal_street_wagers_v2") or {}),
            "stacks": deepcopy(row.get("stacks") or {}),
        }
        while len(self.history) > 64:
            self.history.popitem(last=False)
        new = []
        for action in row.get("observed_actions_v2", []):
            key = identity(action)
            if key not in self.actions:
                self.actions[key] = self._interpret(action, row)
                new.append(deepcopy(self.actions[key]))
        while len(self.actions) > 256:
            self.actions.popitem(last=False)
        return {"interpreted_actions": new,
                "interpreted_action_history": deepcopy(list(self.actions.values())),
                "hand_phase": self._phase(row)}
