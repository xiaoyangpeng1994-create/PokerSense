"""The readable rule examples must continue matching an independent engine."""

import pytest


def test_rulebook_examples_against_pokerkit():
    pytest.importorskip("pokerkit")
    from tools.verify_nlhe_rulebook import verify_rulebook
    report = verify_rulebook()
    assert report["passed"] is True
    assert len(report["cases"]) == 12
    assert report["production_wpk_parity"] == "NOT_YET_VERIFIED"
