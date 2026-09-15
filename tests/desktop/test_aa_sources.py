import hashlib
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from poker_engine.desktop.aa_sources import (
    AACaptureSource, AADevelopmentSource, source_factory,
)


def test_capture_is_lazy_and_uses_aa_canvas():
    calls = []

    class Backend:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def capture(self, target):
            calls.append(target.window_id)
            time.sleep(0.01)
            return SimpleNamespace(image=np.zeros((1080, 498, 3), np.uint8),
                                   frame_seq=8)

        def release(self):
            calls.append("released")

    src = AACaptureSource({"device_index": 2, "api": "DSHOW"},
                          backend_factory=Backend)
    assert len(calls) == 1
    assert calls[0]["normalization"].crop_after_rotation == (711, 0, 1209, 1080)
    assert src.read()["source_frame"] == 8
    assert calls[1] == "uvc-2"
    src.close()
    assert calls[-1] == "released"


@pytest.mark.parametrize("options", [{"device_index": True},
                                     {"device_index": -1}, {"api": "ANY"}])
def test_invalid_device_settings_never_construct_backend(options):
    def forbidden(**kwargs):
        pytest.fail("must reject before constructing backend")
    with pytest.raises(ValueError):
        AACaptureSource(options, backend_factory=forbidden)


def test_manifest_development_only_hash_bound_frames(tmp_path):
    _, encoded = cv2.imencode(".png", np.zeros((1080, 498, 3), np.uint8))
    data = encoded.tobytes()
    (tmp_path / "frame.png").write_bytes(data)
    manifest = {"audit_sha256": "audit", "samples": [
        {"global_frame": 10, "role": "development", "file": "frame.png",
         "sha256": hashlib.sha256(data).hexdigest(), "pts_seconds": "0.3"}]}
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(manifest))
    source = AADevelopmentSource(tmp_path, "audit")
    frame = source.read()
    assert frame["image"].shape == (1080, 498, 3)
    assert frame["pts_seconds"] == 0.3 and frame["source_pts_exact"] == "0.3"
    assert source.read() is None
    (tmp_path / "frame.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        AADevelopmentSource(tmp_path, "audit").read()
    manifest["samples"][0]["role"] = "holdout"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="development frames only"):
        AADevelopmentSource(tmp_path, "audit")


def test_hardware_disabled_by_default(tmp_path):
    factory = source_factory(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="未启用采集卡"):
        factory({"mode": "capture-card"})


def test_browser_cannot_select_arbitrary_replay(tmp_path):
    factory = source_factory(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="已配置"):
        factory({"mode": "development-replay", "path": str(tmp_path)})
