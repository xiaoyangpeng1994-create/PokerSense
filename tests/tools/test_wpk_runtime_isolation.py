"""Runtime isolation is measured, never inferred from installed metadata alone."""

from importlib.metadata import PackageNotFoundError

import pytest

from tools.train_wpk_rank_head import installed_training_inventory
from tools.wpk_runtime_fingerprint import isolation_checks


def test_contrib_only_training_does_not_require_opencv_python(monkeypatch):
    def version(name):
        if name == "opencv-contrib-python":
            return "4.10.0.84"
        if name.startswith("opencv"):
            raise PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr("tools.train_wpk_rank_head.version", version)
    rows = installed_training_inventory()
    assert "opencv-contrib-python==4.10.0.84" in rows
    assert not any(row.startswith("opencv-python==") for row in rows)


def test_fresh_environment_with_local_modules_passes(tmp_path):
    prefix = tmp_path / "clean"
    modules = {"cv2": prefix / "Lib/site-packages/cv2/__init__.py"}
    result = isolation_checks(prefix, tmp_path / "base", modules,
                              [str(prefix / "Lib/site-packages")], False)
    assert result["passed"]


@pytest.mark.parametrize("failure", ["user_site", "foreign_path", "foreign_module",
                                     "not_venv", "no_modules"])
def test_contaminated_or_unverified_runtime_cannot_pass(tmp_path, failure):
    prefix, base = tmp_path / "clean", tmp_path / "base"
    modules = {"cv2": prefix / "Lib/site-packages/cv2/__init__.py"}
    paths, user_site = [str(prefix / "Lib/site-packages")], False
    if failure == "user_site":
        user_site = True
    elif failure == "foreign_path":
        paths.append(str(base / "Lib/site-packages"))
    elif failure == "foreign_module":
        modules["cv2"] = base / "Lib/site-packages/cv2/__init__.py"
    elif failure == "not_venv":
        base = prefix
    else:
        modules = {}
    assert not isolation_checks(prefix, base, modules, paths, user_site)["passed"]
