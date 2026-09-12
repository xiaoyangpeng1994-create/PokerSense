import pytest

from tools.check_private_files import private_reason


@pytest.mark.parametrize("path", [
    "capture.MKV", "folder/.env", ".env.production", "data/user.sqlite3", "client.APK",
    "secrets/key.pem", "screenshots/game.png", "private/notes.json",
    "tmp/codex-clipboard-abc.png"])
def test_private_artifacts_flagged(path):
    assert private_reason(path)


@pytest.mark.parametrize("path", [
    ".env.example", ".env.sample", ".env.template", "configs/vision/aa/card.png",
    "tests/fixtures/hand.json", "packaging/assets/pokersense-icon.png", "src/app.py"])
def test_normal_assets_and_sanitized_example_names_not_blocked(path):
    assert private_reason(path) is None
