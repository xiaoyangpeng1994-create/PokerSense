"""The challenge stays fixed and replays both books on identical worlds."""

from pathlib import Path

from tools.evaluate_range_posterior_v1 import (
    CHALLENGE, CHALLENGE_SHA256, run_challenge,
)
from tools.strategy_evaluation_v1 import digest


def test_precommitted_challenge_replays_every_world():
    assert digest(Path(CHALLENGE).read_bytes()) == CHALLENGE_SHA256
    report = run_challenge()
    rows = report["cases"]
    assert len(rows) == 54
    assert len({r["case_id"] for r in rows}) == 54
    assert report["summary"]["paired_count"] == 54
    assert report["summary"]["blocked_count"] == 0
    assert report["summary"]["zero_fallback_cases"] == 54
    for group in {r["group"] for r in rows}:
        for pattern in {r["pattern"] for r in rows}:
            family = [r for r in rows if r["group"] == group
                      and r["pattern"] == pattern]
            assert len(family) == 3
            assert len({r["candidate_book_sha256"] for r in family}) == 1
            assert len({r["baseline_book_sha256"] for r in family}) == 1
            assert all(r["world_sha256"] for r in family)
    assert report["summary"]["positive_delta_count"] > 0
    assert report["summary"]["negative_delta_count"] > 0
    assert report["strategy_eligible"] is False
    assert report["advice_emitted"] is False
