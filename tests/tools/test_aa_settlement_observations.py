from copy import deepcopy
import json

import pytest

from tools.record_aa_settlement_observations import (
    LocalStatusAPI, _NoRedirect, compact_snapshot, record_observations,
)


def state(frame=10, *, credits=None):
    return {
        "status": "RUNNING", "instance_id": "server-a", "generation": 3,
        "source_kind": "capture-card", "sequence": frame, "source_frame": frame + 100,
        "pts_seconds": frame / 10, "api_key": "DO-NOT-SAVE",
        "payload": {
            "source_id": "capture-a", "cards": {
                "hero": ["Ah", "Ad"], "board_slots": ["Ac", "3s", "5h", None, None]},
            "pot": {"value": "10.1"}, "stacks": {"4": {"value": "100.1"}},
            "street_wagers": {"4": "2"},
            "causal_street_wagers_v2": {"wagers": {"4": "999"}},
            "current_actor": None,
            "observed_state_v2": {"observed_epoch": "hand-1",
                                  "street_candidate": "flop",
                                  "unallocated_positive_cash": credits or []},
            "hand_phase": {"phase": "OBSERVING", "current_ledger": {
                "observed_total": "16.1", "displayed_pot": "10.1",
                "unallocated_difference": "6", "api_key": "DO-NOT-SAVE"}},
            "image": "DO-NOT-SAVE", "fee": "6", "profit": "516",
        },
    }


def credit(*, frame=11):
    return {"epoch": "hand-1", "seat": 4, "first_frame": frame - 1,
            "confirmed_frame": frame, "amount": "0.2",
            "source": "stable_visual_balance_increase",
            "balance_before": "100.1", "balance_after": "100.3",
            "api_key": "DO-NOT-SAVE", "fee": "12"}


class FakeClock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


def run(tmp_path, states, *, duration=2, interval=0.5):
    clock = FakeClock()
    calls = []

    def api(path, *, timeout):
        calls.append((path, clock.now, timeout))
        row = states[min(len(calls) - 1, len(states) - 1)]
        if isinstance(row, Exception):
            raise row
        return deepcopy(row)

    output = tmp_path / "observations.jsonl"
    result = record_observations(api, output, duration=duration, interval=interval,
                                 monotonic=clock.monotonic, sleep=clock.sleep)
    rows = [json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()]
    return result, rows, calls, clock


def test_compaction_is_pure_whitelisted_and_preserves_unknowns():
    observed = state(credits=[credit()])
    original = deepcopy(observed)
    compact = compact_snapshot(observed)
    assert observed == original
    assert compact["seats"]["4"] == {"balance": "100.1", "visible_street_wager": "2"}
    assert compact["seats"]["0"] == {"balance": None, "visible_street_wager": None}
    assert compact["current_actor"] is None
    assert compact["board_slots"][-2:] == [None, None]
    assert compact["current_ledger"]["unallocated_difference"] == "6"
    text = json.dumps(compact)
    assert "DO-NOT-SAVE" not in text
    assert '"fee"' not in text and '"profit"' not in text
    assert "api_key" not in text and '"image"' not in text


def test_cleared_current_ledger_never_falls_back_to_historical():
    observed = state()
    observed["payload"]["hand_phase"].update(
        phase="POT_CLEAR_PENDING", current_ledger=None,
        historical_ledger={"observed_total": "629"})
    observed["payload"]["hand_ledger_v2"] = {"observed_total": "629"}
    assert compact_snapshot(observed)["current_ledger"] is None
    observed["payload"]["stacks"]["4"] = {"status": "UNKNOWN", "value": "12"}
    observed["payload"]["pot"] = {"value": "NaN"}
    result = compact_snapshot(observed)
    assert result["pot"] is None and result["seats"]["4"]["balance"] is None


def test_changed_values_exact_deltas_and_new_credits_once(tmp_path):
    old = credit(frame=3)
    first = state(credits=[old])
    duplicate = state(11, credits=[old])
    second = state(12, credits=[old, credit()])
    second["payload"]["pot"] = {"value": "0"}
    second["payload"]["stacks"]["4"] = {"value": "100.3"}
    result, rows, calls, clock = run(tmp_path, [first, duplicate, second, second])
    assert [c[0] for c in calls] == ["/api/status"] * 4
    assert [c[1] for c in calls] == [0, 0.5, 1, 1.5]
    assert clock.now == 2 and result["reason"] == "duration_reached"
    assert result["samples"] == 2 and result["duplicates_skipped"] == 2
    assert rows[1]["baseline_credit_count"] == 1
    snapshots = [r for r in rows if r["type"] == "SNAPSHOT"]
    assert snapshots[0]["new_unallocated_positive_cash"] == []
    new = snapshots[1]
    assert len(new["new_unallocated_positive_cash"]) == 1
    assert new["changes_since_previous_sample"]["balances"]["4"] == {
        "before": "100.1", "after": "100.3", "delta": "0.2"}
    assert new["changes_since_previous_sample"]["pot"]["delta"] == "-10.1"
    assert "DO-NOT-SAVE" not in json.dumps(rows)


@pytest.mark.parametrize("update, reason", [
    ({"generation": 4}, "session_changed"),
    ({"instance_id": "server-b"}, "session_changed"),
    ({"source_kind": "development-replay"}, "source_changed_or_not_capture"),
    ({"status": "STOPPED"}, "capture_not_running"),
    ({"payload": None}, "missing_payload"),
])
def test_scope_or_stop_rejects_foreign_snapshot(tmp_path, update, reason):
    changed = state(11)
    changed.update(update)
    result, rows, calls, _ = run(tmp_path, [state(), changed])
    assert result["reason"] == reason and len(calls) == 2
    assert sum(r["type"] == "SNAPSHOT" for r in rows) == 1


def test_internal_source_change_is_also_rejected(tmp_path):
    changed = state(11)
    changed["payload"]["source_id"] = "capture-b"
    result, _, calls, _ = run(tmp_path, [state(), changed])
    assert result["reason"] == "source_changed" and len(calls) == 2


def test_unknown_transition_is_recorded_with_null_delta(tmp_path):
    changed = state(11)
    changed["payload"]["observed_state_v2"]["unallocated_positive_cash"] = None
    result, rows, _, _ = run(tmp_path, [state(), changed], duration=1)
    assert result["samples"] == 2
    assert rows[-2]["new_unallocated_positive_cash"] is None
    changed["payload"]["stacks"]["4"] = {"value": None}
    other = tmp_path / "other"
    other.mkdir()
    result, rows, _, _ = run(other, [state(), changed], duration=1)
    assert result["samples"] == 2
    assert rows[-2]["changes_since_previous_sample"]["balances"]["4"] == {
        "before": "100.1", "after": None, "delta": None}


@pytest.mark.parametrize("first", [True, False])
def test_network_failure_stops_without_retry_or_error_body(tmp_path, first):
    failure = TimeoutError("DO-NOT-SAVE API_KEY")
    result, rows, calls, _ = run(tmp_path, [failure] if first else [state(), failure])
    assert result["reason"] == "request_failed"
    assert len(calls) == (1 if first else 2)
    assert "DO-NOT-SAVE" not in json.dumps(rows)


@pytest.mark.parametrize("duration, interval", [
    (601, 0.5), (0, 0.5), (-1, 0.5), (2, 0.49), (float("nan"), 1),
    (1, float("inf")), (True, 1), (2, True),
])
def test_invalid_limits_do_not_call_api_or_create_file(tmp_path, duration, interval):
    output = tmp_path / "absent.jsonl"
    with pytest.raises(ValueError):
        record_observations(None, output, duration=duration, interval=interval)
    assert not output.exists()


def test_600_second_ceiling_and_minimum_interval(tmp_path):
    result, _, calls, clock = run(tmp_path, [state()], duration=600)
    assert clock.now == 600 and len(calls) == 1200
    assert calls[-1][1] == 599.5 and calls[-1][2] == 0.5
    assert result["samples"] == 1


def test_late_response_is_discarded(tmp_path):
    clock = FakeClock()

    def slow_api(path, *, timeout):
        assert path == "/api/status" and timeout == 0.25
        clock.now += 0.25
        return state()

    output = tmp_path / "late.jsonl"
    result = record_observations(slow_api, output, duration=0.25,
                                 monotonic=clock.monotonic, sleep=clock.sleep)
    assert result["reason"] == "duration_reached" and result["samples"] == 0


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "existing.jsonl"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        record_observations(None, output)
    assert output.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("url", [
    "https://127.0.0.1", "http://example.com", "http://localhost/a",
    "http://key@localhost", "http://localhost?key=x", "http://localhost#secret",
    "http://localhost:99999",
])
def test_nonlocal_or_credential_urls_rejected(url):
    with pytest.raises(ValueError):
        LocalStatusAPI(url)


def test_only_get_status_no_proxy_or_redirect(monkeypatch):
    handlers, calls = [], []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, maximum):
            assert maximum == 2 * 1024 * 1024 + 1
            return b'{"status":"RUNNING"}'

    class Opener:
        def open(self, request, *, timeout):
            calls.append(request)
            assert request.full_url == "http://127.0.0.1:8777/api/status"
            assert request.get_method() == "GET" and request.data is None
            assert timeout == 0.5 and not request.has_header("Authorization")
            return Response()

    def build_opener(*args):
        handlers.extend(args)
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build_opener)
    api = LocalStatusAPI("http://127.0.0.1:8777")
    assert api("/api/status", timeout=0.5) == {"status": "RUNNING"}
    assert handlers[0].proxies == {} and isinstance(handlers[1], _NoRedirect)
    assert handlers[1].redirect_request(None, None, 302, "", {}, "http://x") is None
    with pytest.raises(ValueError):
        api("/api/start", timeout=0.5)
    assert len(calls) == 1
