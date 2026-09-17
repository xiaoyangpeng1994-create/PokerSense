from pathlib import Path
import shutil
import subprocess

import pytest


def test_actual_analysis_scripts_in_deterministic_dom():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the deterministic AA UI regression")
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [node, str(root / "tests/ui/test_aa_analysis_ui.js")],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "deterministic AA analysis UI cases" in completed.stdout
