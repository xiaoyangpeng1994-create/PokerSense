import pytest

from tools.aa8_glyph_supplement_v3 import resolve, GlyphSupplementV3


@pytest.mark.parametrize("base,scores,expected", [
    ("fold", {"fold": .91, "muck": .97, "all_in": .3}, None),
    ("fold", {"fold": .98, "muck": .90, "all_in": .3}, "fold"),
    (None, {"fold": .3, "muck": .4, "all_in": .98}, "all_in"),
    (None, {"fold": .5, "muck": .6, "all_in": .7}, None),
    ("fold", {"fold": .94, "muck": .95, "all_in": .3}, "fold"),
    ("call", {"fold": .3, "muck": .4, "all_in": .98}, None),
    ("check", {"fold": .2, "muck": .3, "all_in": .4}, "check"),
])
def test_competition_preserves_abstention_and_conflict(base, scores, expected):
    assert resolve(base, scores) == expected


def test_unsupported_frame_never_emits_glyph():
    reader = GlyphSupplementV3.__new__(GlyphSupplementV3)
    assert all(v is None for v in reader.recognize(
        None, {"0": "fold"}, supported=False).values())
