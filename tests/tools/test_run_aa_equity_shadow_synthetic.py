import pytest

from tools.run_aa_equity_shadow_synthetic import build_case, run


@pytest.mark.parametrize("count", (6, 7, 8))
def test_synthetic_cases_exercise_exact_equity_without_advice(count):
    result = build_case(count)
    assert result.status.value == "COMPLETE"
    assert result.equity_report.method.value == "exact"
    assert result.rake.amount == 4
    assert result.gross_expected_chips == 150
    assert result.configured_net_expected_chips == 146
    assert not result.advice_emitted and not result.strategy_eligible


def test_report_is_explicitly_synthetic_and_immutable(tmp_path):
    report = run(tmp_path / "first")
    assert len(report["results"]) == 3
    assert report["real_visual_input"] is False
    assert report["provider_executed"] is False
    assert report["advice_emitted"] is False
    with pytest.raises(FileExistsError):
        run(tmp_path / "first")
