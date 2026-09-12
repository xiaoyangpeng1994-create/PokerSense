import pytest

from tools.record_aa_capture_test import record


@pytest.mark.parametrize("duration", [0, -1, True, 1801, 1.5])
def test_invalid_duration_cannot_start_capture(tmp_path, duration):
    target = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="duration"):
        record(target, duration)
    assert not target.exists()
