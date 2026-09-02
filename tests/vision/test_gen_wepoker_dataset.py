"""Tests for the WePoker-style synthetic dataset generator.

Validates (a) determinism, (b) small.json-aligned GT contract, (c) sample_id
uniqueness, and (d) produced PNGs are readable and correctly shaped. This is
the offline, compliant training-data pipeline that replaces any need to scrape
real platforms.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from tools.gen_wepoker_dataset import generate_dataset, render_table


@pytest.fixture(scope="module")
def _tmp_out(tmp_path_factory):
    return str(tmp_path_factory.mktemp("wepoker"))


def test_generation_is_deterministic(_tmp_out):
    p1 = generate_dataset(_tmp_out, count=20, seed=7)
    p2 = generate_dataset(_tmp_out, count=20, seed=7)
    d1 = json.load(open(p1, encoding="utf-8"))
    d2 = json.load(open(p2, encoding="utf-8"))
    assert d1 == d2


def test_manifest_contract_aligns_with_small_json(_tmp_out):
    p = generate_dataset(_tmp_out, count=30, seed=1)
    d = json.load(open(p, encoding="utf-8"))
    assert d["dataset"] == "wepoker-synthetic"
    for s in d["samples"]:
        assert s["platform_id"] == "wpk"
        assert s["layout_id"] == "6max"
        assert s["source"] == "synthetic-render"
        assert s["sample_id"].startswith("wepoker-")
        gt = s["ground_truth"]
        # full GT field set (small.json contract)
        for key in ("street", "pot", "bet_size", "hero_cards", "board_cards",
                    "stack", "action"):
            assert key in gt, f"missing GT field {key}"


def test_sample_ids_are_unique(_tmp_out):
    p = generate_dataset(_tmp_out, count=50, seed=2)
    d = json.load(open(p, encoding="utf-8"))
    ids = [s["sample_id"] for s in d["samples"]]
    assert len(ids) == len(set(ids))


def test_street_board_card_count_consistent(_tmp_out):
    p = generate_dataset(_tmp_out, count=40, seed=3)
    d = json.load(open(p, encoding="utf-8"))
    expected = {"PREFLOP": 0, "FLOP": 3, "TURN": 4, "RIVER": 5}
    for s in d["samples"]:
        gt = s["ground_truth"]
        assert len(gt["board_cards"]) == expected[gt["street"]]


def test_rendered_pngs_are_valid(_tmp_out):
    p = generate_dataset(_tmp_out, count=5, seed=4)
    d = json.load(open(p, encoding="utf-8"))
    for s in d["samples"]:
        path = s["image_path"]
        assert os.path.isfile(path)
        img = cv2.imread(path)
        assert img is not None
        assert img.shape == (400, 600, 3)
        assert img.dtype == np.uint8


def test_render_table_shape_and_nonempty():
    img = render_table(("AS", "KH", "QD"), ("2C", "3D"), "50", "20",
                       ("100", "200", "300"), ("CHECK", "CALL"))
    assert img.shape == (400, 600, 3)
    # not a blank canvas (some pixels differ from the navy background)
    bg = np.full((400, 600, 3), (38, 44, 60), dtype=np.uint8)
    assert not np.array_equal(img, bg)


# ---------------------------------------------------------------------------
# end-to-end: synthetic WePoker data feeds the recognizer pipeline == GT
# ---------------------------------------------------------------------------

def test_synthetic_frame_feeds_recognizer_matching_gt():
    import sys

    sys.path.insert(0, "tools")
    sys.path.insert(0, "tests")

    import run_benchmark
    from tools.gen_wepoker_dataset import render_table

    hero = ("AS", "KD")
    board = ("QH", "JD", "TC")
    pot, bet = "80", "25"
    stacks = ("120", "240", "360")
    actions = ("CHECK", "BET")

    img = render_table(board, hero, pot, bet, stacks, actions)
    frame = run_benchmark._frame(img, 0)
    obs = run_benchmark.build_engine().process(frame, run_benchmark.table_map())

    # hero + board recognized exactly (the recognizer reads what we rendered)
    def cs(cards):
        return tuple(sorted(c.rank.value + c.suit.value.upper() for c in cards))

    assert cs(obs.hero_cards.value) == tuple(sorted(hero))
    assert cs(obs.board_cards.value) == tuple(sorted(board))
    # street from board count (FLOP = 3 cards)
    assert obs.street.value.value == "flop"
    # pot + bet amounts recognized exactly
    assert obs.pot.value is not None and str(obs.pot.value) == pot
    assert obs.bet_size.value is not None and str(obs.bet_size.value) == bet


def test_synthetic_preflop_frame_empty_board_recognized():
    import sys

    sys.path.insert(0, "tools")
    sys.path.insert(0, "tests")

    import run_benchmark
    from tools.gen_wepoker_dataset import render_table

    hero = ("7H", "2D")
    img = render_table((), hero, "10", "5", ("100", "200", "300"),
                       ("CHECK", "CALL"))
    obs = run_benchmark.build_engine().process(
        run_benchmark._frame(img, 0), run_benchmark.table_map()
    )
    assert obs.board_cards.value == ()          # empty board
    assert obs.street.value.value == "preflop"  # 0 cards -> PREFLOP
