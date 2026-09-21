"""Native-current disagreement must reject, never fill temporal UNKNOWN."""

from copy import deepcopy

import pytest

from tools.aa8_cards import AA8CardReader
from tools.aa8_cards_v2 import AA8CardReaderV2


@pytest.mark.parametrize('temporal,current,expected,reason', [
    ('4h', None, None, 'native_current_unknown'),
    ('4h', '6h', None, 'native_current_conflicts_with_temporal'),
    (None, '6h', None, 'temporal_candidate_unknown'),
    ('6h', '6h', '6h', 'native_current_and_temporal_agree'),
])
def test_gate_rejects_without_promoting(
        monkeypatch, temporal, current, expected, reason):
    original = {'board_slots': ['Qh', 'Tc', '9h', '3h', temporal],
                'raw_board_slots': ['Qh', 'Tc', '9h', '3h', temporal],
                'evidence': {f'board_{i}': {'rect': (i, 1, 53, 78)}
                             for i in range(5)},
                'strategy_eligible': False, 'model_calibrated_for_aa8': False}
    monkeypatch.setattr(AA8CardReader, 'read', lambda *a: deepcopy(original))
    reader = object.__new__(AA8CardReaderV2)
    monkeypatch.setattr(reader, '_native_current',
                        lambda image, rect: current if rect[0] == 4
                        else original['board_slots'][rect[0]])
    # Stable repetition must not turn a consistently unsupported read into KNOWN.
    for frame in range(4):
        row = reader.read(None, frame, frame / 30, 'synthetic')
        assert row['board_slots'][-1] == expected
        assert row['raw_board_slots'] == original['raw_board_slots']
        gate = row['board_qualification_v2']
        assert gate['unqualified_temporal_board'] == original['board_slots']
        assert gate['checks'][-1]['reason'] == reason
        assert row['strategy_eligible'] is False
        assert row['model_calibrated_for_aa8'] is False


def test_unsupported_canvas_has_no_native_crop(monkeypatch):
    monkeypatch.setattr(AA8CardReader, 'read', lambda *a: {
        'board_slots': [None] * 5, 'raw_board_slots': [None] * 5,
        'evidence': {}, 'reason': 'unsupported_canvas'})
    reader = object.__new__(AA8CardReaderV2)
    assert reader.read(None, 0, 0, 's')['board_slots'] == [None] * 5
