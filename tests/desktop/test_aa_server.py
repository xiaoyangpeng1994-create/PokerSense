from fastapi.testclient import TestClient

from poker_engine.desktop import aa_server


class Session:
    def __init__(self):
        self.starts = []
        self.stops = 0

    def snapshot(self):
        return {"status": "STOPPED", "payload": None}

    def preview(self):
        return None

    def start(self, options):
        self.starts.append(options)

    def stop(self):
        self.stops += 1


HEADERS = {"X-AA-Live": "1"}


def test_page_status_do_not_start_and_missing_profile_is_visible(tmp_path):
    session = Session()
    app = aa_server.create_app(tmp_path / "missing.json", session=session)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        status = client.get("/api/status").json()
        assert status["profile"]["ready"] is False
        assert status["capture_available"] is False
        assert client.get("/api/preview.jpg").status_code == 404
        assert session.starts == []
    assert session.stops == 1


def test_controls_are_same_origin_and_explicit(tmp_path, monkeypatch):
    session = Session()
    monkeypatch.setattr(aa_server, "preflight_profile",
                        lambda _: {"ready": True, "errors": []})
    app = aa_server.create_app(tmp_path / "profile.json", replay_pool=tmp_path,
                               session=session)
    with TestClient(app) as client:
        body = {"mode": "development-replay"}
        assert client.post("/api/start", json=body).status_code == 403
        cross = {**HEADERS, "Origin": "https://example.com"}
        assert client.post("/api/start", json=body, headers=cross).status_code == 403
        assert client.post("/api/start", json=body,
                           headers=HEADERS).status_code == 200
        assert session.starts == [body]
        assert client.post("/api/start", json={"mode": "capture-card"},
                           headers=HEADERS).status_code == 403
        assert client.post("/api/start", json={**body, "path": "other"},
                           headers=HEADERS).status_code == 400
        assert client.post("/api/stop", json={}, headers=HEADERS).status_code == 200
        assert session.stops == 1


def test_missing_models_prevents_any_start(tmp_path):
    session = Session()
    app = aa_server.create_app(tmp_path / "missing.json", replay_pool=tmp_path,
                               session=session)
    with TestClient(app) as client:
        response = client.post("/api/start", json={"mode": "development-replay"},
                               headers=HEADERS)
        assert response.status_code == 409
        assert session.starts == []


def test_frozen_ui_path(monkeypatch, tmp_path):
    monkeypatch.setattr(aa_server.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert aa_server.ui_root() == tmp_path / "ui" / "aa-live"
