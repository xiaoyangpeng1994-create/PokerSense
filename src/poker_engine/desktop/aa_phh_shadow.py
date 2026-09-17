"""PHH shadow export: confirmed AA facts -> PHH -> PokerKit, as an ORACLE only.

This is a SIDE CHANNEL. It reads PokerSense's confirmed facts, writes a PHH
(Poker Hand History) document, and replays that document through PokerKit to
produce a second, independent opinion about the same hand. It never feeds
anything back into the live state machine and it never replaces a line of
PokerSense's own rule code.

The rule that matters here: **a PHH required field is only ever written from a
confirmed PokerSense fact.** PHH's ``NT`` (no-limit Texas hold'em) variant
requires ``antes``, ``blinds_or_straddles``, ``min_bet``, ``starting_stacks``
and ``actions`` up front, and every one of them -- plus every action -- is
indexed by player number ``p1..pn``. PokerKit will happily accept
``antes = [0, 0, 0]`` and silently treat an unobserved ante as a zero ante,
which is exactly the failure mode this module exists to prevent: a downstream
consumer would read a made-up zero as a fact, and a hand that lost chips to
antes would look like a hand that did not.

So a field that cannot be traced to confirmed evidence is NEVER defaulted. The
export is REFUSED, the missing field is named, and the evidence that would close
it is reported alongside it. Two invariants are load-bearing:

* ``UNKNOWN -> ABSENT``: an unconfirmed required field yields
  ``status = NOT_EXPORTABLE``, never a default and never a placeholder.
* ``ORACLE -> SEPARATE``: the differential comparison only runs on a document
  whose required fields were all confirmed, and every comparison that cannot be
  made is reported as ``NOT_COMPARABLE`` with a reason -- never as a pass.

PokerKit is an OPTIONAL dependency (``pyproject.toml`` extra ``solver-tools``).
The import is lazy so everything outside this module keeps working on a runtime
where PokerKit is absent; absence is reported as its own status.
"""

from decimal import Decimal, InvalidOperation

SHADOW_SCHEMA = "aa-phh-shadow-v1"
VARIANT = "NT"
POKERKIT_PINNED = "0.7.5"

#: PokerKit's OWN required ``NT`` fields, less ``variant`` (a product constant
#: rather than a per-hand reading). A test asserts this against the installed
#: library, so the tuple cannot drift away from what PokerKit actually demands.
POKERKIT_NT_REQUIRED = ("antes", "blinds_or_straddles", "min_bet",
                        "starting_stacks", "actions")

#: This module's gate: PokerKit's own list, plus ``variant``, plus
#: ``seat_order``. ``seat_order`` is NOT a PHH header field -- there is no such
#: field in the notation. It is a PokerSense PRECONDITION, because a PHH
#: indexes antes, blinds, starting stacks and every action by player number
#: ``p1..pn``, so an unconfirmed button would silently move a seat's forced bet
#: onto a different seat.
PHH_REQUIRED = ("variant", "seat_order", "antes", "blinds_or_straddles",
                "min_bet", "starting_stacks", "actions")

#: The optional PHH header field that carries the physical seats. PokerKit
#: round-trips it byte-identically, and without it a PHH is only positionally
#: indexed -- ``p2`` cannot be mapped back onto a seat number, so the reverse
#: direction could never restore ``seat_order``.
SEATS_FIELD = "seats"

#: ``(PHH field, PokerSense evidence, what makes it confirmed)``. Single source
#: of truth for the field-mapping document, so prose and code cannot drift.
FIELD_MAP = (
    ("variant", "product scope: the AA desktop track only watches hold'em",
     "a product constant, not a screen reading"),
    ("seat_order",
     "the dealt seats in positional order, from "
     "``live_dealer_candidate`` + the seat geometry",
     "the button seat is CONFIRMED, not a stability-gated candidate"),
    ("antes", "table rules ``ante`` + ``ante_mode`` (manual declaration)",
     "``ante`` is a declared amount and ``ante_mode`` is not ``unknown``; a "
     "declared ``0`` with mode ``none`` is a real zero, not a missing value"),
    ("blinds_or_straddles",
     "table rules ``small_blind`` / ``big_blind`` / ``straddle_mode``, assigned "
     "positionally through ``seat_order``",
     "both blinds declared, the straddle mode known, and the positional order "
     "confirmed"),
    ("min_bet", "table rules ``big_blind``",
     "``big_blind`` declared -- that is the minimum bet in a no-limit game"),
    ("starting_stacks", "per-seat stack reading at the hand's opening row",
     "every seat's stack confirmed BEFORE any forced bet is posted"),
    ("actions", "``observed_actions_v2`` interpretations",
     "``interpretation_status`` is a confirmed status, not a "
     "``PRICE_DERIVED_CANDIDATE``"),
    ("seats", "the confirmed ``seat_order``",
     "same precondition as ``seat_order``: written so a reader can map ``p1..pn``"
     " back onto physical seats, which a PHH does not carry otherwise"),
)


class _PokerKitMissing(RuntimeError):
    """PokerKit is not importable on this runtime."""


def _pokerkit():
    """Import PokerKit lazily; the shadow path is optional at runtime."""
    try:
        from pokerkit import Automation, HandHistory
    except ImportError as error:  # pragma: no cover - depends on the runtime
        raise _PokerKitMissing(str(error)) from None
    return Automation, HandHistory


def _declared(document, key):
    """A user-declared table-rule amount as a ``Decimal``, or None if absent."""
    if not isinstance(document, dict):
        return None
    value = document.get(key)
    if not isinstance(value, str) or not value:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite() or number < 0:
        return None
    return number


def _chips(number):
    """PHH chip amounts are integral; a fractional forced bet is unmappable."""
    if number is None:
        return None
    if number != number.to_integral_value():
        return None
    return int(number)


def _entry(name, value, source, reason):
    confirmed = value is not None
    return {"field": name, "value": value, "source": source,
            "status": "CONFIRMED" if confirmed else "UNCONFIRMED",
            "unconfirmed_because": None if confirmed else reason}


def _seat_order(facts, seats):
    """PokerSense seats in PHH ``p1..pn`` order, only if confirmed."""
    order = facts.get("seat_order")
    if (not isinstance(order, list) or len(order) != seats
            or len({str(seat) for seat in order}) != seats
            or facts.get("seat_order_confirmed") is not True):
        return _entry(
            "seat_order", None,
            "live_dealer_candidate + seat geometry",
            "the positional order is not confirmed: PHH indexes antes, blinds, "
            "starting stacks and every action by player number, so an "
            "unconfirmed button would silently move a seat's forced bet onto "
            "another seat"), None
    return _entry("seat_order", [str(seat) for seat in order],
                  "live_dealer_candidate + seat geometry", None), [
        str(seat) for seat in order]


def _forced_bets(facts, seats):
    """Map declared table rules onto PHH's antes / blinds / min_bet."""
    rules = facts.get("table_rules")
    small = _chips(_declared(rules, "small_blind"))
    big = _chips(_declared(rules, "big_blind"))
    ante = _chips(_declared(rules, "ante"))
    ante_mode = (rules or {}).get("ante_mode")
    straddle_mode = (rules or {}).get("straddle_mode")
    straddle = _chips(_declared(rules, "straddle_amount"))
    fields = [_entry(
        "min_bet", big, "table_rules.big_blind",
        "big_blind is not declared, so the no-limit minimum bet is unknown")]
    # ``optional_explicit_utg`` is deliberately NOT mappable: the table rules
    # declare the straddle AMOUNT, but whether it was posted in this hand is an
    # observation, not a declaration. Writing the straddle as 0 because the
    # amount is known would silently restate a straddled hand as an unstraddled
    # one, so the field stays unconfirmed until the posting itself is observed.
    blind_values = None
    if (small is not None and big is not None
            and straddle_mode in ("none", "mandatory_utg")
            and (straddle_mode == "none" or straddle is not None)):
        blind_values = [small, big] + [0] * (seats - 2)
        if straddle_mode == "mandatory_utg" and seats >= 3:
            blind_values[2] = straddle
    fields.append(_entry(
        "blinds_or_straddles", blind_values, "table_rules blinds/straddle",
        "small_blind/big_blind are not both declared, or the straddle mode is "
        "unknown or per-hand (optional_explicit_utg), so the forced bets cannot "
        "be stated"))
    ante_values = None
    if ante is not None and ante_mode in ("none", "per_dealt_player"):
        ante_values = [0] * seats if ante == 0 else [ante] * seats
    fields.append(_entry(
        "antes", ante_values, "table_rules.ante + ante_mode",
        "the ante is not declared or its mode is unknown; PHH would otherwise "
        "record an unobserved ante as a zero ante"))
    return fields


def _starting_stacks(facts, seats, order):
    """Per-seat opening stacks in positional order, only when all confirmed."""
    declared = facts.get("starting_stacks")
    if order is None or not isinstance(declared, dict):
        return _entry(
            "starting_stacks", None,
            "per-seat stack reading at the hand opening",
            "the positional order is unconfirmed, or no per-seat opening "
            "stack was supplied")
    values = []
    for seat in order:
        value = _chips(_declared(declared, seat))
        if value is None:
            return _entry(
                "starting_stacks", None,
                "per-seat stack reading at the hand opening",
                f"seat {seat} has no confirmed opening stack")
        values.append(value)
    return _entry("starting_stacks", values,
                  "per-seat stack reading at the hand opening", None)


def _actions(facts):
    """The PHH action list, only from confirmed interpretations."""
    raw = facts.get("actions")
    if not isinstance(raw, list) or not raw:
        return _entry("actions", None, "observed_actions_v2",
                      "no action list was supplied for this hand"), []
    confirmed, rejected = [], []
    for item in raw:
        status = (item or {}).get("interpretation_status")
        text = (item or {}).get("phh")
        if status == "CONFIRMED" and isinstance(text, str) and text:
            confirmed.append(text)
        else:
            rejected.append({"seat": (item or {}).get("slot"),
                             "kind": (item or {}).get("kind"),
                             "reason": status or "no_status"})
    if rejected:
        return _entry(
            "actions", None, "observed_actions_v2",
            f"{len(rejected)} of {len(raw)} action interpretations are not "
            "confirmed; PHH would turn a candidate interpretation into a fact"
        ), rejected
    return _entry("actions", confirmed, "observed_actions_v2", None), []


def _pulled_by_player(final):
    """Chips PokerKit pushed, keyed by PLAYER NUMBER (0-based)."""
    pulled = {}
    for operation in getattr(final, "operations", None) or ():
        if type(operation).__name__ != "ChipsPulling":
            continue
        index = getattr(operation, "player_index", None)
        amount = getattr(operation, "amount", None)
        if isinstance(index, int) and isinstance(amount, int):
            pulled[index] = pulled.get(index, 0) + amount
    return pulled


def _pulled_by_seat(final, order):
    """Chips PokerKit actually pushed to each seat, keyed by physical seat.

    This -- not the net payoff -- is what PokerSense's ``unallocated_positive_
    cash`` records: the settlement-time *balance delta*. The two coincide only
    when the credited seat contributed nothing to the pot, so comparing credits
    against net payoffs would silently misreport every split pot. The chips
    pull is the operation-level fact both sides can be held to.
    """
    pulled = {}
    for operation in getattr(final, "operations", None) or ():
        if type(operation).__name__ != "ChipsPulling":
            continue
        index = getattr(operation, "player_index", None)
        amount = getattr(operation, "amount", None)
        if isinstance(index, int) and isinstance(amount, int) and index < len(order):
            seat = str(order[index])
            pulled[seat] = pulled.get(seat, 0) + amount
    return pulled


def _replay(history):
    """Replay a PHH and return ``(final_state, pot_split)``.

    ``State.pots`` is zeroed the moment the chips are pushed, so the split has
    to be sampled at the last moment it is still readable -- the last state
    that still had a non-zero total. Reading it off the finished state instead
    reports an empty hand, which is how a side-pot check silently becomes a
    tautology.
    """
    final, pots = None, []
    for state, _action in history.state_actions:
        final = state
        if state.total_pot_amount:
            pots = [(int(pot.raked_amount) + int(pot.unraked_amount),
                     [int(index) for index in pot.player_indices])
                    for pot in state.pots]
    return final, pots


def _folded_indices(final):
    """Player numbers that folded, read from the operation log.

    ``State.statuses`` cannot be used for this: it is ``False`` both for a
    player who folded and for a hand that was killed at showdown, and a killed
    hand was still eligible for the pot it lost. Only the ``Folding`` operation
    distinguishes the two.
    """
    folded = set()
    for operation in getattr(final, "operations", None) or ():
        if type(operation).__name__ != "Folding":
            continue
        index = getattr(operation, "player_index", None)
        if isinstance(index, int):
            folded.add(index)
    return folded


def _side_pot_check(pots, order, contributions, folded):
    """Slice the contributions by level and hold that against PokerKit's pots.

    This is an oracle self-consistency check, not a PokerSense comparison:
    PokerSense's ordinary terminal records ONE unallocated credit per seat and
    no pot levels at all, so there is nothing on the PokerSense side to hold a
    split against. What can be checked is that the split PokerKit computed is
    the split the money implies -- every part of the replay must agree on how
    the contributions were sliced.

    The slicing is done here from ``starting - final + pulled``, raised level by
    level: at each distinct positive contribution the payers are everyone who
    paid at least that much, and the *eligible* seats are those payers who did
    not fold. A player who called in full and then folded still paid into the
    pot but is not eligible for it, so fold status is load-bearing.

    **A first version of this check divided the pot by its eligible count to
    recover the level and reported a MISMATCH on a plain single-pot hand** -- the
    level is not ``amount / eligible``, and the mistake is recorded here rather
    than quietly deleted because it is the exact shape of a check that looks
    like it verifies something and does not.
    """
    seats = [str(seat) for seat in order]
    evidence = {"evidence": "none: the ordinary terminal records one "
                            "unallocated credit per seat and no pot levels"}
    if not pots:
        return {"check": "side_pot_split", "poker_kit": {"pots": []},
                "poker_sense": evidence,
                "comparable_against_poker_sense": False,
                "status": "NOT_COMPARABLE",
                "reason": "the replay produced no pot at all, so there is no "
                          "split to check"}
    expected, previous = {}, 0
    for level in sorted({paid for paid in contributions if paid > 0}):
        payers = [index for index, paid in enumerate(contributions)
                  if paid >= level]
        eligible = frozenset(index for index in payers
                             if index not in folded)
        expected[eligible] = (expected.get(eligible, 0)
                              + (level - previous) * len(payers))
        previous = level
    actual = {}
    for amount, indices in pots:
        key = frozenset(indices)
        actual[key] = actual.get(key, 0) + amount
    problems = []
    if expected != actual:
        problems.append({
            "expected_pots": {sorted(seats[i] for i in key): value
                              for key, value in expected.items()},
            "actual_pots": {sorted(seats[i] for i in key): value
                            for key, value in actual.items()}})
    return {
        "check": "side_pot_split",
        "poker_kit": {
            "total": sum(amount for amount, _indices in pots),
            "pots": [{"amount": amount,
                      "eligible_seats": [seats[index] for index in indices]}
                     for amount, indices in pots],
            "contributed_per_seat": dict(zip(seats, contributions)),
            "folded_seats": [seats[index] for index in sorted(folded)],
        },
        "poker_sense": evidence,
        "comparable_against_poker_sense": False,
        "status": "MATCH" if not problems else "MISMATCH",
        "problems": problems,
        "reason": "PokerKit's split is held against the contributions and fold "
                  "status the same replay reports; PokerSense records no pot "
                  "levels, so this item can never be a two-engine comparison"}


def _differential(starting, final, settlement, order, pots):
    """Compare PokerKit's replay against PokerSense's own settlement evidence.

    PokerKit reports payoffs by player number, PokerSense reports credits by
    physical seat, so the comparison goes through ``order``. That is deliberate:
    a wrong positional order would move a payout onto the wrong seat, and this
    check is where that shows up instead of being silently averaged away.
    """
    total_in, total_out = sum(starting), sum(final.stacks)
    checks = [{
        "check": "pot_commitment_conservation",
        "poker_kit": {"chips_in": total_in, "chips_out": total_out,
                      "difference": total_in - total_out},
        "poker_sense": {"evidence": "no rake is declared or applied here"},
        "status": "MATCH" if total_in - total_out == 0 else "MISMATCH"}]
    observed = {}
    for credit in settlement or []:
        try:
            seat = str(credit.get("seat"))
            observed[seat] = observed.get(seat, 0) + int(
                Decimal(str(credit.get("amount"))))
        except (InvalidOperation, TypeError, ValueError):
            continue
    pulled = _pulled_by_seat(final, order)
    net = {str(order[index]): int(payoff)
           for index, payoff in enumerate(final.payoffs or [])
           if payoff and index < len(order)}
    checks.append({
        "check": "payout_per_seat",
        "poker_kit": dict(sorted(pulled.items())),
        "poker_sense": dict(sorted(observed.items())),
        "poker_kit_net_payoffs": dict(sorted(net.items())),
        "status": "MATCH" if pulled == observed else "MISMATCH"})
    checks.append({
        "check": "forced_bets",
        "poker_kit": {"source": "PHH antes + blinds_or_straddles"},
        "poker_sense": {"evidence": "per-seat opening debits"},
        "status": "NOT_COMPARABLE",
        "reason": "the forced bets written into the PHH come from the declared "
                  "table rules; this round has no confirmed per-seat opening "
                  "debit to check that declaration against"})
    pulled_by_player = _pulled_by_player(final)
    contributions = [starting[index] - int(final.stacks[index])
                     + pulled_by_player.get(index, 0)
                     for index in range(len(starting))]
    checks.append(_side_pot_check(pots, order, contributions,
                                  _folded_indices(final)))
    return checks


def shadow_export(facts):
    """Map confirmed facts to PHH and differentially replay them via PokerKit.

    ``facts`` carries the confirmed side of a closed hand: ``seats``, the
    confirmed ``seat_order``, ``table_rules`` (the saved table-rules document),
    optional per-seat ``starting_stacks``, optional confirmed ``actions``, and
    the terminal's ``settlement`` credits.

    Returns a report. When any PHH required field is unconfirmed the export is
    refused with ``status = NOT_EXPORTABLE``: nothing is defaulted, so a missing
    fact can never reach a downstream consumer as a zero, a False or a loser.
    """
    facts = facts or {}
    seats = facts.get("seats")
    report = {"schema": SHADOW_SCHEMA, "variant": VARIANT,
              "field_map": [list(row) for row in FIELD_MAP],
              "replaces_production_state_machine": False,
              "poker_kit_pinned": POKERKIT_PINNED}
    if not isinstance(seats, int) or seats < 2:
        report.update(status="NOT_EXPORTABLE", reason="seat_count_unconfirmed",
                      missing_required=["seats"])
        return report
    report["seat_count"] = seats
    order_entry, order = _seat_order(facts, seats)
    fields = [{"field": "variant", "value": VARIANT,
               "source": "product scope: hold'em table track",
               "status": "CONFIRMED", "unconfirmed_because": None},
              order_entry]
    fields.extend(_forced_bets(facts, seats))
    fields.append(_starting_stacks(facts, seats, order))
    actions_entry, rejected = _actions(facts)
    fields.append(actions_entry)
    missing = [f["field"] for f in fields
               if f["field"] in PHH_REQUIRED and f["status"] != "CONFIRMED"]
    report.update(fields=fields, missing_required=missing,
                  rejected_action_interpretations=rejected)
    if missing:
        report.update(status="NOT_EXPORTABLE",
                      reason="unconfirmed_required_fields")
        return report
    try:
        Automation, HandHistory = _pokerkit()
    except _PokerKitMissing as error:
        report.update(status="POKERKIT_UNAVAILABLE", reason=str(error))
        return report
    by_name = {f["field"]: f["value"] for f in fields}
    # Construction is INSIDE the guard: PokerKit raises ``ValueError: Unable to
    # repair the hand history`` for an illegal action list, and an illegal list
    # is exactly what the action-legality comparison exists to report. Outside
    # the guard that failure escaped as a traceback instead of a named refusal.
    document = None
    try:
        history = HandHistory(
            variant=VARIANT, ante_trimming_status=False,
            antes=by_name["antes"],
            blinds_or_straddles=by_name["blinds_or_straddles"],
            min_bet=by_name["min_bet"],
            starting_stacks=by_name["starting_stacks"],
            actions=by_name["actions"], seats=order)
        document = history.dumps()
        reloaded = HandHistory.loads(document)
        identity = reloaded.dumps() == document
        final, pots = _replay(reloaded)
        if final is None:
            raise ValueError("the replay produced no state at all")
        legality = not final.status
    except Exception as error:  # noqa: BLE001 - PokerKit raises many types
        refusal = {"status": "POKERKIT_REJECTED",
                   "reason": f"{type(error).__name__}: {error}"}
        if document is not None:
            refusal["phh"] = document
        report.update(refusal)
        return report
    report.update(
        status="EXPORTED", phh=document,
        poker_kit={"phh_text_identity": identity,
                   "physical_seats_carried": list(reloaded.seats or []),
                   "all_actions_accepted": legality,
                   "hand_finished": bool(not final.status),
                   "final_stacks": [int(x) for x in final.stacks],
                   "payoffs": [int(x) for x in (final.payoffs or [])]},
        differential=_differential(by_name["starting_stacks"], final,
                                   facts.get("settlement"), order, pots))
    return report


def shadow_ingest(document):
    """Read a PHH back into PokerSense vocabulary: the REVERSE direction.

    Nothing this returns is marked CONFIRMED, and that is deliberate. A PHH is
    data we produced, not a PokerSense screen observation, so the reverse
    adapter must not be usable to satisfy the forward gate: feeding this output
    back into :func:`shadow_export` REFUSES, because the reconstructed actions
    carry the status ``PHH_ROUND_TRIP`` and the gate accepts only ``CONFIRMED``.
    The reverse direction exists to compare the two engines field by field, not
    to feed one into the other.

    A PHH guarantees only ``p1..pn`` order. Physical seats are recoverable ONLY
    because the forward writer puts them in the optional ``seats`` header --
    PokerKit round-trips that field verbatim. A PHH written without it can never
    be mapped back onto seat numbers, and this function says so by returning
    ``seat_order = None`` rather than inventing ``p1..pn`` as seat numbers.
    """
    blank = {"schema": SHADOW_SCHEMA, "direction": "PHH_TO_POKERSENSE"}
    if not isinstance(document, str) or not document.strip():
        return dict(blank, status="PHH_UNREADABLE",
                    reason="no PHH text was supplied")
    try:
        _automation, history_class = _pokerkit()
    except _PokerKitMissing as error:
        return dict(blank, status="POKERKIT_UNAVAILABLE", reason=str(error))
    try:
        history = history_class.loads(document)
    except Exception as error:  # noqa: BLE001 - PokerKit raises many types
        return dict(blank, status="PHH_UNREADABLE",
                    reason=f"{type(error).__name__}: {error}")
    stacks = [int(value) for value in history.starting_stacks]
    seats = [str(seat) for seat in (history.seats or [])]
    actions = [{"phh": str(text), "interpretation_status": "PHH_ROUND_TRIP"}
               for text in history.actions]
    facts = {
        "variant": history.variant,
        "seats": len(stacks),
        "seat_order": seats or None,
        "seat_order_confirmed": bool(seats) and len(seats) == len(stacks),
        "antes": [int(value) for value in history.antes],
        "blinds_or_straddles": [int(value) for value in
                                history.blinds_or_straddles],
        "min_bet": None if history.min_bet is None else int(history.min_bet),
        "starting_stacks": stacks,
        "actions": actions,
    }
    return dict(blank, status="INGESTED", facts=facts,
                action_count=len(actions),
                notes=["nothing here is CONFIRMED, so the forward gate still "
                       "refuses this output",
                       "physical seats come from the optional `seats` header "
                       "only; without it the positional order cannot be "
                       "restored"])


def _round_trip_row(field, sent, read_back):
    return {"field": field, "sent": sent, "read_back": read_back,
            "status": "MATCH" if sent == read_back else "MISMATCH"}


def shadow_round_trip(facts):
    """``facts -> PHH -> facts`` and report every field that survived.

    This is the forward/reverse pair held against each other, which is a
    stronger statement than ``dumps(loads(dumps)) == dumps``: the round trip
    has to preserve the *meaning* PokerSense put in, not merely the bytes.
    """
    forward = shadow_export(facts)
    if forward.get("status") != "EXPORTED":
        return {"schema": SHADOW_SCHEMA, "status": "ROUND_TRIP_NOT_RUN",
                "forward_status": forward.get("status"),
                "reason": "the forward export refused, so there is no document "
                          "to read back"}
    reverse = shadow_ingest(forward["phh"])
    if reverse.get("status") != "INGESTED":
        return {"schema": SHADOW_SCHEMA, "status": "ROUND_TRIP_FAILED",
                "reverse_status": reverse.get("status"),
                "reason": reverse.get("reason")}
    sent = {field["field"]: field["value"] for field in forward["fields"]}
    got = reverse["facts"]
    checks = [_round_trip_row("variant", sent.get("variant"),
                              got.get("variant"))]
    for field in ("antes", "blinds_or_straddles", "min_bet",
                  "starting_stacks"):
        checks.append(_round_trip_row(field, sent.get(field), got.get(field)))
    checks.append(_round_trip_row(
        "actions", list(sent.get("actions") or []),
        [item["phh"] for item in got.get("actions", [])]))
    checks.append(_round_trip_row(
        "seats", sent.get("seat_order"), got.get("seat_order")))
    mismatched = [row["field"] for row in checks
                  if row["status"] != "MATCH"]
    return {"schema": SHADOW_SCHEMA,
            "status": "ROUND_TRIP_MATCH" if not mismatched
                      else "ROUND_TRIP_MISMATCH",
            "checks": checks, "mismatched_fields": mismatched,
            "phh": forward["phh"], "reverse_notes": reverse["notes"]}
