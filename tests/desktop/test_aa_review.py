from copy import deepcopy
import hashlib
import json
import threading
import time

from fastapi.testclient import TestClient
import pytest

from poker_engine.desktop import aa_review, aa_server


KEY = "synthetic-test-key-not-a-credential"
RESULT = {"summary": "候选金额需核对", "findings": [{
    "field": "底池", "observed": "720", "visible": "120", "status": "mismatch"}]}


def evidence(frame=10):
    return ({"status": "RUNNING", "generation": 1, "source_frame": frame,
             "sequence": frame, "source_kind": "development-replay",
             "payload": {"cards": {"hero": ["5d", "6d"]},
                         "pot": {"value": "720"}, "current_actor": 4}},
            b"synthetic-jpeg-for-provenance-tests")


def marked(desk):
    return desk.mark(evidence(), {"revision": "r1"})["issue"]["issue_id"]


def finished(desk, identifier):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = desk.get(identifier)["ai"]
        if result and result["status"] != "RUNNING":
            return result
        time.sleep(.005)
    pytest.fail("review did not finish")


def configured(tmp_path, transport=None, deadline=30, limit=20):
    desk = aa_review.AAReviewDesk(tmp_path, transport=transport, deadline=deadline)
    desk.configure(KEY, "deepseek-flash", limit)
    return desk


def test_frozen_snapshot_and_manual_history_do_not_rewrite_observations(tmp_path):
    desk = aa_review.AAReviewDesk(tmp_path)
    live = evidence()
    record = desk.mark(live, {"revision": "r1"})
    identifier = record["issue"]["issue_id"]
    original = (tmp_path / identifier / "issue.json").read_bytes()
    live[0]["payload"]["pot"]["value"] = "900"
    saved = desk.get(identifier)["issue"]["observation"]["payload"]
    assert saved["pot"]["value"] == "720"
    first = desk.review(identifier, "incorrect", "底池实际是120", None)
    assert first["human"]["training_eligible"] is False
    assert first["issue"]["review_status"] == "UNREVIEWED"
    with pytest.raises(ValueError, match="其他页面"):
        desk.review(identifier, "correct", "", None)
    desk.review(identifier, "unreadable", "只核对底池", first["human"]["revision"])
    assert len(list((tmp_path / identifier).glob("human-*.json"))) == 3
    assert (tmp_path / identifier / "issue.json").read_bytes() == original
    assert desk.recent()[0]["human_status"] == "unreadable"


@pytest.mark.parametrize("identifier", ["../outside", "bad", "x" * 500, None])
def test_rejects_arbitrary_paths(tmp_path, identifier):
    with pytest.raises(ValueError):
        aa_review.AAReviewDesk(tmp_path).get(identifier)


def test_image_tampering_stops_cloud_request(tmp_path):
    calls = []
    desk = configured(tmp_path, lambda *args: calls.append(args))
    identifier = marked(desk)
    (tmp_path / identifier / "preview.jpg").write_bytes(b"different")
    with pytest.raises(ValueError, match="校验"):
        desk.start(identifier, True)
    assert not calls
    assert desk.config()["calls_today"] == 0


def test_config_has_no_network_or_secret_persistence(tmp_path):
    calls = []
    desk = configured(tmp_path, lambda *args: calls.append(args))
    config = desk.config()
    assert config["key_configured"] and KEY not in json.dumps(config)
    assert not calls
    identifier = marked(desk)
    with pytest.raises(ValueError, match="确认"):
        desk.start(identifier, False)
    assert not calls
    for path in tmp_path.rglob("*.json"):
        assert KEY not in path.read_text(encoding="utf-8")
    desk.clear_key()
    with pytest.raises(ValueError, match="API Key"):
        desk.start(identifier, True)


def test_success_sends_only_selected_frame_and_preserves_gates(tmp_path):
    calls = []

    def transport(key, model, jpeg, observed):
        calls.append((key, model, jpeg, observed))
        return deepcopy(RESULT), {"total_tokens": 100}

    desk = configured(tmp_path, transport)
    identifier = marked(desk)
    desk.mark(evidence(200), {"revision": "r2"})
    desk.start(identifier, True)
    result = finished(desk, identifier)
    assert len(calls) == 1
    assert calls[0][2] == desk.image(identifier)
    assert calls[0][3]["pot"]["value"] == "720"
    assert result["result"] == RESULT
    assert result["training_eligible"] is False
    assert result["strategy_eligible"] is False
    assert result["source_sha256"] == hashlib.sha256(calls[0][2]).hexdigest()
    assert desk.get(identifier)["human"] is None


def test_budget_survives_restart_and_failures_count(tmp_path):
    def failing(*args):
        raise RuntimeError(KEY)

    desk = configured(tmp_path, failing, limit=1)
    identifier = marked(desk)
    desk.start(identifier, True)
    result = finished(desk, identifier)
    assert result["status"] == "ERROR" and result["result"] is None
    assert KEY not in json.dumps(result)
    desk.close()
    restarted = configured(tmp_path, failing, limit=1)
    assert restarted.config()["calls_today"] == 1
    with pytest.raises(ValueError, match="上限"):
        restarted.start(identifier, True)


def test_timeout_keeps_single_worker_and_rejects_late_result(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        release.wait(2)
        return deepcopy(RESULT), {}

    desk = configured(tmp_path, slow, deadline=.03)
    identifier = marked(desk)
    desk.start(identifier, True)
    assert entered.wait(1)
    with pytest.raises(ValueError, match="正在处理"):
        desk.start(identifier, True)
    assert finished(desk, identifier)["status"] == "TIMED_OUT"
    assert desk.config()["busy"]
    worker = desk._active[1]
    release.set()
    worker.join(1)
    result = desk.get(identifier)["ai"]
    assert result["status"] == "TIMED_OUT" and result["result"] is None
    assert not desk.config()["busy"]


def test_restart_exposes_interrupted_request_without_resubmitting(tmp_path):
    desk = configured(tmp_path)
    identifier = marked(desk)
    aa_review.atomic_json(tmp_path / identifier / "ai-review.json",
                          {"job_id": "old", "status": "RUNNING"})
    assert desk.get(identifier)["ai"]["status"] == "INTERRUPTED"
    assert desk.config()["calls_today"] == 0


@pytest.mark.parametrize("bad", [
    {**RESULT, "strategy_eligible": True},
    {"summary": "text", "findings": [{"field": "x", "observed": "0",
                                      "visible": "2", "status": "approved"}]},
    {"summary": "text", "findings": [RESULT["findings"][0]] * 21},
])
def test_invalid_model_payload_is_not_published(tmp_path, bad):
    desk = configured(tmp_path, lambda *args: (bad, {}))
    identifier = marked(desk)
    desk.start(identifier, True)
    result = finished(desk, identifier)
    assert result["status"] == "ERROR" and result["result"] is None


def test_request_uses_fixed_endpoint_image_json_and_bounded_output(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, maximum):
            captured["read_limit"] = maximum
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(RESULT)}}], "usage": {"total_tokens": 10,
                                                            "secret": KEY}}).encode()

    class Opener:
        def open(self, request, timeout):
            captured.update(request=request, timeout=timeout)
            return Response()

    monkeypatch.setattr(aa_review.urllib.request, "build_opener", lambda *a: Opener())
    result, usage = aa_review.request_deepseek(KEY, "deepseek-flash", b"picture", {})
    request = captured["request"]
    assert request.full_url == aa_review.ENDPOINT
    payload = json.loads(request.data)
    assert payload["max_tokens"] == 2200
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,")
    assert KEY not in request.data.decode()
    assert captured["read_limit"] == aa_review.MAX_REPLY + 1
    assert result == RESULT and usage == {"total_tokens": 10}
    with pytest.raises(ValueError):
        aa_review.NoRedirect().redirect_request(None, None, None, None, None, None)


class Session:
    def snapshot(self):
        return evidence()[0]

    def evidence(self):
        return evidence()

    def stop(self):
        pass


def test_routes_mark_review_and_explicit_cloud_consent(tmp_path):
    desk = configured(tmp_path, lambda *args: (deepcopy(RESULT), {}))
    app = aa_server.create_app(tmp_path / "missing", session=Session(),
                               records_dir=tmp_path, review_service=desk)
    headers = {"X-AA-Live": "1"}
    with TestClient(app) as client:
        assert client.get("/review.js").status_code == 200
        assert client.post("/api/review/mark", json={}).status_code == 403
        assert client.post("/api/review/mark", json={}, headers={
            **headers, "Origin": "https://example.org"}).status_code == 403
        record = client.post("/api/review/mark", json={}, headers=headers).json()
        identifier = record["issue"]["issue_id"]
        route = "/api/review/issues/" + identifier
        assert client.get(route + "/image").content == evidence()[1]
        assert client.post(route + "/ai", json={"consent": False},
                           headers=headers).status_code == 400
        assert client.post(route + "/human", json={"verdict": "incorrect",
                           "note": "底池120", "revision": None},
                           headers=headers).status_code == 200
        assert client.post(route + "/ai", json={"consent": True},
                           headers=headers).status_code == 200
        assert finished(desk, identifier)["status"] == "COMPLETE"
        assert KEY not in client.get("/api/review/config").text
        assert len(client.get("/api/review/issues").json()["items"]) == 1
