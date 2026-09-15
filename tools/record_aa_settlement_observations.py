"""Record changed observations from an already running local AA capture session.

GET /api/status only: no device access, images, model calls or server changes.
Values and positive cash events remain unverified observations, not fees/profit.
The initial credit list is a baseline; only newly seen confirmations are emitted.
Sampling cannot establish complete hand/action coverage or settlement attribution.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalStatusAPI:
    """A fixed, read-only endpoint with proxies and redirects disabled."""

    def __init__(self, base_url):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in (
                "127.0.0.1", "localhost", "::1")
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Only a local AA HTTP endpoint is allowed")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError("Invalid local port")
        self.url = base_url.rstrip("/") + "/api/status"
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())

    def __call__(self, path, *, timeout):
        if path != "/api/status":
            raise ValueError("Only GET /api/status is allowed")
        request = urllib.request.Request(self.url, method="GET")
        with self.opener.open(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Status response is too large")
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise ValueError("Status response must be an object")
        return state


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _scalar(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _amount(value):
    if isinstance(value, dict):
        if str(value.get("status", "")).lower() in (
                "unknown", "not_applicable", "invalid", "blocked"):
            return None
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        parsed = Decimal(str(value))
        return str(parsed) if parsed.is_finite() else None
    except InvalidOperation:
        return None


def _cards(value):
    if not isinstance(value, list):
        return None
    return [c if isinstance(c, str) and re.fullmatch(r"[2-9TJQKA][cdhs]", c)
            else None for c in value]


def _ledger(value):
    if not isinstance(value, dict):
        return None
    result = {k: _scalar(value.get(k)) for k in (
        "epoch", "status", "applied_action_count")}
    result.update({k: _amount(value.get(k)) for k in (
        "observed_total", "displayed_pot", "unallocated_difference")})
    commitments = value.get("hand_commitments")
    result["hand_commitments"] = (
        {str(s): _amount(commitments.get(str(s))) for s in range(8)}
        if isinstance(commitments, dict) else None)
    return result


def compact_snapshot(state):
    """Pure whitelist projection; absent/unknown values stay null.

    An explicitly null phase ledger must never fall back to historical totals.
    Visible wagers come from OCR street_wagers, not the causal wager estimate.
    Credits here are the full API list; the recorder filters first-seen events.
    """
    row = _mapping(state.get("payload"))
    observed = _mapping(row.get("observed_state_v2"))
    phase = _mapping(row.get("hand_phase"))
    cards = _mapping(row.get("cards"))
    balances, wagers = _mapping(row.get("stacks")), _mapping(row.get("street_wagers"))
    raw_credits = observed.get("unallocated_positive_cash")
    credits = None
    if isinstance(raw_credits, list):
        credits = []
        for item in raw_credits:
            if not isinstance(item, dict):
                continue
            credit = {k: _scalar(item.get(k)) for k in (
                "epoch", "seat", "first_frame", "confirmed_frame", "source")}
            credit.update({k: _amount(item.get(k)) for k in (
                "amount", "balance_before", "balance_after")})
            credits.append(credit)
    return {
        "sequence": _scalar(state.get("sequence")),
        "source_frame": _scalar(state.get("source_frame")),
        "pts_seconds": _scalar(state.get("pts_seconds")),
        "source_id": _scalar(row.get("source_id")),
        "hero_cards": _cards(cards.get("hero")),
        "board_slots": _cards(cards.get("board_slots")),
        "pot": _amount(row.get("pot")),
        "seats": {str(s): {"balance": _amount(balances.get(str(s))),
                           "visible_street_wager": _amount(wagers.get(str(s)))}
                  for s in range(8)},
        "current_actor": _scalar(row.get("current_actor")),
        "observed_epoch": _scalar(observed.get("observed_epoch")),
        "street_candidate": _scalar(observed.get("street_candidate")),
        "hand_phase": _scalar(phase.get("phase")),
        "current_ledger": _ledger(phase.get("current_ledger")),
        "unallocated_positive_cash": credits,
    }


def _signature(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _credit_key(credit):
    return _signature({k: credit[k] for k in (
        "epoch", "seat", "first_frame", "confirmed_frame", "amount", "source")})


def _difference(before, after):
    return str(Decimal(after) - Decimal(before)) if (
        before is not None and after is not None) else None


def _changes(before, after):
    """Observed sample differences only; never assign the cause of a credit."""
    if before is None:
        return None
    return {
        "from_sequence": before["sequence"],
        "from_source_frame": before["source_frame"],
        "same_observed_epoch": (
            before["observed_epoch"] is not None
            and before["observed_epoch"] == after["observed_epoch"]),
        "pot": {"before": before["pot"], "after": after["pot"],
                "delta": _difference(before["pot"], after["pot"])},
        "balances": {s: {
            "before": before["seats"][s]["balance"],
            "after": after["seats"][s]["balance"],
            "delta": _difference(before["seats"][s]["balance"],
                                 after["seats"][s]["balance"])}
                     for s in after["seats"]},
    }


def record_observations(api, output, *, duration=600, interval=0.5,
                        monotonic=time.monotonic, sleep=time.sleep):
    """Sample until the deadline, first failure, stopped source or changed scope.

    Output is a new JSONL file; existing files are never overwritten. The injected
    API must accept ("/api/status", timeout=seconds). No request starts at/after
    the deadline; late responses are discarded. Socket timeout uses remaining time.
    """
    for name, value in (("duration", duration), ("interval", interval)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(name + " must be finite")
    if not 0 < duration <= 600 or interval < 0.5:
        raise ValueError("Duration must be in (0, 600]; interval must be at least 0.5")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    deadline = started + duration
    binding = source_id = last_start = previous = last_signature = None
    polls = samples = duplicates = 0
    seen_credits = set()
    reason = "duration_reached"
    with output.open("x", encoding="utf-8") as log:
        def emit(row):
            log.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                  **row}, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

        emit({"type": "STARTED", "duration_seconds": duration,
              "interval_seconds": interval, "candidate_only": True,
              "interpretation": "OBSERVATIONS_ONLY_NO_FEE_OR_PROFIT_INFERENCE"})
        try:
            while monotonic() < deadline:
                if last_start is not None:
                    delay = min(max(0, last_start + interval - monotonic()),
                                max(0, deadline - monotonic()))
                    if delay:
                        sleep(delay)
                if monotonic() >= deadline:
                    break
                last_start = monotonic()
                polls += 1
                try:
                    state = api("/api/status", timeout=min(5, deadline - last_start))
                except Exception:
                    # Do not serialize exception text, response bodies or credentials.
                    reason = "request_failed"
                    break
                if monotonic() >= deadline:
                    break
                if not isinstance(state, dict):
                    reason = "invalid_status"
                    break
                current_binding = (state.get("instance_id"), state.get("generation"))
                if (not isinstance(current_binding[0], str) or not current_binding[0]
                        or type(current_binding[1]) is not int):
                    reason = "invalid_session_binding"
                    break
                if binding is not None and current_binding != binding:
                    reason = "session_changed"
                    break
                if state.get("source_kind") != "capture-card":
                    reason = "source_changed_or_not_capture"
                    break
                if state.get("status") != "RUNNING":
                    reason = "capture_not_running"
                    break
                if not isinstance(state.get("payload"), dict) or not state["payload"]:
                    reason = "missing_payload"
                    break
                compact = compact_snapshot(state)
                if binding is not None and compact["source_id"] != source_id:
                    reason = "source_changed"
                    break
                credits = compact.pop("unallocated_positive_cash")
                if binding is None:
                    binding, source_id = current_binding, compact["source_id"]
                    seen_credits.update(_credit_key(c) for c in credits or ())
                    emit({"type": "BOUND", "instance_id": binding[0],
                          "generation": binding[1], "source_kind": "capture-card",
                          "baseline_credit_count": len(credits)
                          if credits is not None else None})
                new_credits = None if credits is None else []
                for credit in credits or ():
                    key = _credit_key(credit)
                    if key not in seen_credits:
                        seen_credits.add(key)
                        new_credits.append(credit)
                signature = _signature({
                    "credit_list_known": credits is not None,
                    **{k: v for k, v in compact.items()
                       if k not in ("sequence", "source_frame", "pts_seconds")}})
                if signature != last_signature or new_credits:
                    samples += 1
                    emit({"type": "SNAPSHOT", "sample": samples, **compact,
                          "changes_since_previous_sample": _changes(previous, compact),
                          "new_unallocated_positive_cash": new_credits})
                    previous, last_signature = compact, signature
                else:
                    duplicates += 1
        except KeyboardInterrupt:
            reason = "interrupted"
        result = {"type": "END", "reason": reason, "samples": samples,
                  "polls": polls, "duplicates_skipped": duplicates,
                  "elapsed_seconds": monotonic() - started}
        emit(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8777")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    result = record_observations(LocalStatusAPI(args.base_url), args.output,
                                 duration=args.duration, interval=args.interval)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
