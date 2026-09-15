import json

import pytest

from tools.run_aa_live_review_round import LocalAPI, run_round, sample


def fake_api(*, changed_mark=False, busy=False):
    calls = []
    state = {"status": "RUNNING", "source_kind": "capture-card",
             "instance_id": "server", "generation": 3, "payload": {},
             "source_frame": 7}
    state["payload"] = {"cards": {}}

    def api(path, body=None):
        calls.append((path, body))
        if path == "/api/status":
            return dict(state)
        if path == "/api/review/config":
            return {"key_configured": True, "calls_today": 0,
                    "daily_limit": 20, "busy": busy}
        if path == "/api/review/mark":
            return {"issue": {"issue_id": "test", "preview_sha256": "a" * 64,
                              "observation": {**state,
                                              "generation": 4 if changed_mark else 3}}}
        if path.endswith("/ai"):
            return {"job_id": "job", "model": "deepseek-flash"}
        raise AssertionError(path)

    return api, calls, state


def test_no_consent_or_nonlocal_api():
    with pytest.raises(ValueError):
        run_round(None, "unused")
    for url in ("https://example.com", "http://user:pass@localhost",
                "http://localhost/a", "http://localhost?key=x"):
        with pytest.raises(ValueError):
            LocalAPI(url)


def test_round_respects_rate_and_total(tmp_path):
    api, calls, _ = fake_api()
    clock = [0]

    def sleep(amount):
        assert amount >= 30
        clock[0] += amount

    run_round(api, tmp_path, consent=True, maximum=3,
              monotonic=lambda: clock[0], sleep=sleep)
    assert clock[0] == 60
    assert sum(path.endswith("/ai") for path, _ in calls) == 3
    lines = (tmp_path / "round.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows[-1]["submitted"] == 3
    assert all(row["training_eligible"] is False for row in rows
               if row["status"] == "SUBMITTED")


def test_source_changes_and_busy_never_upload():
    api, calls, _ = fake_api(changed_mark=True)
    assert sample(api, ("server", 3))["reason"] == "source_changed_during_mark"
    assert not any(path.endswith("/ai") for path, _ in calls)
    api, calls, _ = fake_api(busy=True)
    assert sample(api, ("server", 3))["status"] == "SKIP"
    assert not any(path == "/api/review/mark" for path, _ in calls)
    api, calls, state = fake_api()
    state["source_kind"] = "development-replay"
    assert sample(api, ("server", 3))["status"] == "END"
    assert len(calls) == 1


def test_uncertain_submission_is_not_retried(tmp_path):
    api, calls, _ = fake_api()

    def uncertain(path, body=None):
        result = api(path, body)
        if path.endswith("/ai"):
            raise TimeoutError("Response lost after submission")
        return result

    run_round(uncertain, tmp_path, consent=True)
    assert sum(path.endswith("/ai") for path, _ in calls) == 1
    last = json.loads((tmp_path / "round.jsonl").read_text().splitlines()[-1])
    assert last["automatic_retry"] is False
