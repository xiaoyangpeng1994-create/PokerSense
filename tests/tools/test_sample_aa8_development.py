from decimal import Decimal
import json
from pathlib import Path

import pytest

from tools.sample_aa8_development import permitted, sample


def plan():
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/reproduction/aa8_recording_split_plan_20260909.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("start,end", [("0", "300"), ("20", "80"), ("820", "995.366")])
def test_development_ranges_only(start, end):
    assert permitted(Decimal(start), Decimal(end), plan())


@pytest.mark.parametrize("start,end", [
    ("299", "301"), ("300", "600"), ("600", "820"), ("800", "850"), ("0", "0")])
def test_other_roles_and_crossing_ranges_rejected(start, end):
    assert not permitted(Decimal(start), Decimal(end), plan())


@pytest.mark.parametrize("step", ["0", "-1", "NaN", "Infinity"])
def test_invalid_sampling_never_touches_files(tmp_path, step):
    with pytest.raises(ValueError, match="sampling"):
        sample(tmp_path, tmp_path, tmp_path, tmp_path / "output", step=step)
    assert not (tmp_path / "output").exists()
