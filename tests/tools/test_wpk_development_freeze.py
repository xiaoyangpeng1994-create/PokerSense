"""Development reruns preserve old evidence and snapshot dependencies."""

import json

import cv2
import pytest

from tools.capture_card_calibration.hashing import sha256_file
from tools.replay_wpk_card_development import freeze_development


def test_freeze_keeps_parent_and_captures_configuration(tmp_path):
    repo = tmp_path / "repo"
    files = {
        "src/recognizer.py": "# candidate",
        "configs/geometry.json": '{"x": 12}',
        "tools/validate_wpk_card_batch.py": "# validator",
        "pyproject.toml": "# project",
    }
    for name, value in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    batch = tmp_path / "batch"
    batch.mkdir()
    original = batch / "corrected-score.json"
    original.write_text(json.dumps({"corrected_spec": {
        "checkpoints": [{"source_frame": 10, "hero": ["Ac", "Qh"]}],
        "criteria": {"max_wrong_accepted": 0},
    }}), encoding="utf-8")
    before = original.read_bytes()
    output = tmp_path / "development"
    frozen = freeze_development(repo, batch, output)
    result = json.loads(frozen.read_text(encoding="utf-8"))
    assert "NOT holdout" in result["independence_scope"]
    assert result["runtime"]["opencv"] == cv2.__version__
    assert "opencv-python" not in result["runtime"]
    assert result["checkpoints"][0]["hero"] == ["Ac", "Qh"]
    assert result["criteria"] == {"max_wrong_accepted": 0}
    assert result["parent_corrected_score_sha256"] == sha256_file(original)
    assert original.read_bytes() == before
    for relative in files:
        assert result["frozen_model_files"][relative] == sha256_file(repo / relative)
        assert (output / "model-snapshot" / relative).read_bytes() == (
            repo / relative).read_bytes()
    with pytest.raises(FileExistsError):
        freeze_development(repo, batch, output)
    assert original.read_bytes() == before
