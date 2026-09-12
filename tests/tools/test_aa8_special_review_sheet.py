import json

import pytest

from tools.aa8_special_modes import AUDIT
from tools.aa8_special_review_sheet import run


@pytest.mark.parametrize("pts,role", [
    ("300", "development"), ("600", "development"),
    ("819.99", "development"), ("10", "holdout"), ("995.366", "development"),
])
def test_rejects_non_development_ranges(tmp_path, pts, role):
    source = tmp_path / "source"
    source.mkdir()
    (source / "samples.json").write_text(json.dumps({
        "audit_sha256": AUDIT,
        "samples": [{"pts_seconds": pts, "role": role}],
    }))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="development-only"):
        run(source, output)
    assert not output.exists()
