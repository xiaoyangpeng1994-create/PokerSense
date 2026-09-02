"""Task 1A smoke test: verify package import and version."""

import poker_engine


def test_package_importable():
    assert poker_engine is not None


def test_version_present():
    assert isinstance(poker_engine.__version__, str)
    assert poker_engine.__version__ != ""


def test_core_package_importable():
    from poker_engine import core  # noqa: F401

    assert core is not None
