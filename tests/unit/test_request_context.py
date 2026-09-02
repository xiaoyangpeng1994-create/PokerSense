"""Tests for RequestContext."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from poker_engine.core.request_context import RequestContext

UTC = timezone.utc


def _ctx(**overrides):
    args = dict(
        hand_id="h1",
        state_version=1,
        request_id="r1",
        requested_at=datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC),
    )
    args.update(overrides)
    return RequestContext(**args)


def test_valid():
    c = _ctx()
    assert c.hand_id == "h1"
    assert c.state_version == 1


def test_empty_hand_id_invalid():
    with pytest.raises(ValueError):
        _ctx(hand_id="")


def test_negative_state_version_invalid():
    with pytest.raises(ValueError):
        _ctx(state_version=-1)


def test_empty_request_id_invalid():
    with pytest.raises(ValueError):
        _ctx(request_id="")


def test_frozen():
    c = _ctx()
    with pytest.raises(FrozenInstanceError):
        c.hand_id = "h2"  # type: ignore[misc]


def test_naive_requested_at_rejected():
    with pytest.raises(TypeError):
        _ctx(requested_at=datetime(2026, 8, 18))


def test_aware_requested_at_ok():
    c = _ctx()
    assert c.requested_at.tzinfo is UTC


def test_expiry_and_deadline_valid():
    requested = datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)
    c = _ctx(
        requested_at=requested,
        expires_at=requested + timedelta(milliseconds=300),
        deadline_ms=300,
    )
    assert c.expires_at > c.requested_at
    assert c.deadline_ms == 300


def test_expiry_must_be_after_request():
    requested = datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        _ctx(requested_at=requested, expires_at=requested)


@pytest.mark.parametrize("deadline", (0, -1))
def test_deadline_must_be_positive(deadline):
    with pytest.raises(ValueError):
        _ctx(deadline_ms=deadline)


def test_deadline_bool_rejected():
    with pytest.raises(TypeError):
        _ctx(deadline_ms=True)
