"""Task 7B synthetic benchmark runner + real-adapter smoke test.

Synthetic pipeline: Golden sample -> Frame + TableMap -> VisionEngine ->
RawObservation -> GT comparison -> FieldMetric -> report.

Real mode is a **Real Adapter Smoke Test**: it verifies the end-to-end data
path (image -> Frame -> TableMap -> VisionEngine -> RawObservation -> GT), NOT
a full real-world accuracy benchmark. It does NOT establish real accuracy.

Covers hero_cards, board_cards, street, pot, bet_size, stack, action.

``--mode synthetic`` renders its own tables via ``gen_wepoker_dataset`` and
exercises pipeline plumbing end to end. It does NOT measure real-world
recognition accuracy — synthetic art is far cleaner than a real client's, so
a synthetic score says nothing about a real table. For that, see
``tests/vision/test_corner_glyph_recognizer.py``, which runs against real
captured card art.

``--mode real`` scores a golden JSON of real captures. No such dataset is
committed yet; the hero-card calibration in ``configs/vision/wepoker/`` is
currently validated by the test above instead.

Run:
  python tools/run_benchmark.py --mode synthetic --out benchmark-results.json
  python tools/run_benchmark.py --mode real --golden path/to/golden.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from poker_engine.core.enums import ActionType, Street  # noqa: E402
from poker_engine.perceptual.capture.base import Frame, WindowRect  # noqa: E402
from poker_engine.perceptual.vision import (  # noqa: E402
    BoardSlotLayout,
    CalibrationBins,
    CardSubROI,
    ConfidenceCalibrator,
    HeroSlotLayout,
    ROIKind,
    ROI,
    TableMap,
    VisionAssetManifest,
    VisionEngine,
)
from poker_engine.perceptual.vision.action_recognizer import (  # noqa: E402
    ActionTemplateSet,
    TemplateActionRecognizer,
)
from poker_engine.perceptual.vision.amount_recognizer import (  # noqa: E402
    DigitTemplateSet,
    TemplateAmountRecognizer,
)
from poker_engine.perceptual.vision.board_slot_detector import (  # noqa: E402
    TemplateBoardSlotDetector,
)
from poker_engine.perceptual.vision.corner_glyph_recognizer import (  # noqa: E402
    CornerGlyphCardRecognizer,
    CornerGlyphGeometry,
    CornerGlyphTemplateSet,
    isolate_glyph,
)
from poker_engine.perceptual.vision.street_detector import (  # noqa: E402
    TemplateStreetDetector,
)

from vision.fixtures.synthetic import (  # noqa: E402
    render_card,
    render_char,
    render_empty_slot,
)

UTC = timezone.utc


def _street_enum(s: str):
    return {"PREFLOP": Street.PREFLOP, "FLOP": Street.FLOP,
            "TURN": Street.TURN, "RIVER": Street.RIVER}[s]


def _card_str(card):
    """Card -> 'AS' style string (rank value + uppercase suit value)."""
    return card.rank.value + card.suit.value.upper()


def _cards_str(cards):
    return tuple(sorted(_card_str(c) for c in cards))


# The synthetic renders in tests/vision/fixtures/synthetic.py draw a 60x84
# card whose corner index sits lower and wider than real WePoker art, so it
# needs its own corner geometry rather than the real-platform default.
SYNTHETIC_GEOMETRY = CornerGlyphGeometry(
    rank_band=(0.10, 0.46), suit_band=(0.52, 0.85), x_window=(0.05, 0.50)
)


def build_engine():
    board_layout = BoardSlotLayout(
        layout_id="b", version=1,
        slots=tuple(CardSubROI(x=i * 0.2, y=0.0, width=0.18, height=1.0)
                    for i in range(5)),
    )
    hero_layout = HeroSlotLayout(
        layout_id="h", version=1,
        slots=(CardSubROI(x=0.0, y=0.0, width=0.48, height=1.0),
               CardSubROI(x=0.5, y=0.0, width=0.48, height=1.0)),
    )

    rank_labels = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    suit_labels = ["S", "H", "D", "C"]
    card_templates = CornerGlyphTemplateSet(
        rank_templates={
            label: isolate_glyph(
                render_card(label, "S"), SYNTHETIC_GEOMETRY.rank_band,
                SYNTHETIC_GEOMETRY,
            )
            for label in rank_labels
        },
        suit_templates={
            suit: isolate_glyph(
                render_card("A", suit), SYNTHETIC_GEOMETRY.suit_band,
                SYNTHETIC_GEOMETRY,
            )
            for suit in suit_labels
        },
        version="v1",
        geometry=SYNTHETIC_GEOMETRY,
    )
    card = CornerGlyphCardRecognizer(card_templates)
    board_det = TemplateBoardSlotDetector(
        board_layout, empty_min_evidence=0.3, card_min_presence=0.25,
    )
    amount = TemplateAmountRecognizer(
        DigitTemplateSet({d: render_char(d) for d in "0123456789."}, "v1")
    )

    # action templates (seat-rendered labels)
    import cv2
    import numpy as np

    def _action_img(label):
        img = np.full((24, 90, 3), 255, dtype=np.uint8)
        cv2.putText(img, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        return img

    action_templates = ActionTemplateSet(
        templates={
            ActionType.CHECK: _action_img("CHECK"),
            ActionType.CALL: _action_img("CALL"),
            ActionType.BET: _action_img("BET"),
            ActionType.FOLD: _action_img("FOLD"),
        },
        version="v1",
    )
    action = TemplateActionRecognizer(action_templates)

    bins = CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9))
    calibrators = {
        k: ConfidenceCalibrator(k, 1, bins,
                                abstain_floor=0.5 if k == "card" else None)
        for k in (
            "card", "amount", "stack", "street", "action", "board", "dealer",
        )
    }
    manifest = VisionAssetManifest(
        platform_id="wpk", layout_id="6max", card_layout_version=1,
        template_set_version="sha-bench", calibration_version=1,
        recognizer_versions={"card": "1", "amount": "1", "stack": "1",
                             "street": "1", "action": "1", "board": "1",
                             "dealer": "1"},
    )
    return VisionEngine(
        board_layout=board_layout, hero_layout=hero_layout,
        card_recognizer=card, board_slot_detector=board_det,
        street_detector=TemplateStreetDetector(), amount_recognizer=amount,
        stack_recognizer=amount,
        action_recognizer=action, calibrators=calibrators, manifest=manifest,
        bet_size_semantics="global",
    )


def table_map():
    return TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(
            ROI(kind=ROIKind.BOARD_CARDS, x=0.0, y=0.0, width=1.0, height=0.25),
            ROI(kind=ROIKind.HERO_CARDS, x=0.0, y=0.7, width=0.4, height=0.25),
            ROI(kind=ROIKind.POT, x=0.4, y=0.45, width=0.2, height=0.1),
            ROI(kind=ROIKind.BET_SIZE, x=0.4, y=0.55, width=0.2, height=0.1),
            # three per-seat stacks (below bet ROI to avoid overlap)
            ROI(kind=ROIKind.STACK, x=0.55, y=0.68, width=0.12, height=0.08, slot_id=1),
            ROI(kind=ROIKind.STACK, x=0.70, y=0.68, width=0.12, height=0.08, slot_id=2),
            ROI(kind=ROIKind.STACK, x=0.85, y=0.68, width=0.12, height=0.08, slot_id=3),
            # two per-seat actions (below stacks)
            ROI(kind=ROIKind.ACTION, x=0.55, y=0.80, width=0.15, height=0.06,
                slot_id=1),
            ROI(kind=ROIKind.ACTION, x=0.75, y=0.80, width=0.15, height=0.06,
                slot_id=2),
        ),
    )


def _draw_amount_on_frame(frame, txt, x0, y0, w=120, h=40):
    """Fill a ROI patch fully white and draw black amount text (so the amount
    recognizer sees only dark-text-on-white, with no dark frame bleed)."""
    import cv2

    frame[y0:y0 + h, x0:x0 + w] = 255
    text_y = y0 + h - 8
    for i, ch in enumerate(txt):
        cv2.putText(frame, ch, (x0 + 4 + i * 20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame


def build_frame(board_slots, hero_slots, pot_text="10", bet_text="5",
                stacks=("100", "200", "300"), actions=("CHECK", "CALL")):
    import cv2
    import numpy as np

    W, H = 600, 400
    frame = np.full((H, W, 3), 60, dtype=np.uint8)

    def px(x, y):
        return int(x * W), int(y * H)

    # board: 5 original-size cards in board ROI sub-slots
    for i, slot_img in enumerate(board_slots):
        x0, y0 = px(i * 0.2 + 0.01, 0.03)
        frame[y0:y0 + slot_img.shape[0], x0:x0 + slot_img.shape[1]] = slot_img

    # hero: two cards bottom-left
    for i, slot_img in enumerate(hero_slots):
        x0, y0 = px(0.08 + i * 0.2, 0.73)
        frame[y0:y0 + slot_img.shape[0], x0:x0 + slot_img.shape[1]] = slot_img

    # pot (ROI 0.4..0.6 x 0.45..0.55 = 120x40) -> full white patch + text
    x0, y0 = px(0.40, 0.45)
    _draw_amount_on_frame(frame, pot_text, x0, y0, w=120, h=40)

    # bet_size (ROI 0.4..0.6 x 0.55..0.65 = 120x40)
    x0, y0 = px(0.40, 0.55)
    _draw_amount_on_frame(frame, bet_text, x0, y0, w=120, h=40)

    # stacks (each ROI ~72x32 at x=0.55/0.7/0.85, y=0.68)
    stack_xs = (0.55, 0.70, 0.85)
    for i, txt in enumerate(stacks):
        x0, y0 = px(stack_xs[i], 0.68)
        _draw_amount_on_frame(frame, txt, x0, y0, w=72, h=32)

    # actions (each ROI ~90x24 at x=0.55/0.75, y=0.80)
    act_xs = (0.55, 0.75)
    for i, txt in enumerate(actions):
        x0, y0 = px(act_xs[i], 0.80)
        frame[y0:y0 + 24, x0:x0 + 90] = 255
        cv2.putText(frame, txt, (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
    return frame


def _frame(img, seq):
    return Frame(
        frame_seq=seq,
        timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        window_id="t", window_rect=WindowRect(0, 0, 600, 400),
        image=img, width=600, height=400,
    )


# (board cards, hero cards, street, pot, bet, stacks, actions)
_SAMPLES = [
    ([], ["AS", "KD"], "PREFLOP", "10", "5",
     ("100", "200", "300"), ("CHECK", "CALL")),
    (["AS", "KH", "QD"], ["2C", "3D"], "FLOP", "25", "10",
     ("100", "200", "300"), ("CHECK", "BET")),
    (["AS", "KH", "QD", "JC"], ["2C", "3D"], "TURN", "50", "20",
     ("150", "250", "350"), ("BET", "CALL")),
    (["AS", "KH", "QD", "JC", "TS"], ["2C", "3D"], "RIVER", "100", "40",
     ("200", "300", "400"), ("BET", "FOLD")),
]


def _run_synthetic(repeat: int) -> list[dict]:
    """Run the synthetic smoke benchmark.

    NOTE: ``repeat`` re-runs the SAME fixed 4 scenarios per loop. These are
    REPEATED synthetic samples, NOT independent samples — ``n`` must NOT be
    treated as a real sample count, and synthetic results MUST NEVER count
    toward acceptance (they are smoke-only). Real acceptance requires a real,
    non-duplicated Golden dataset.
    """
    eng = build_engine()
    tm = table_map()
    entries = []
    seq = 0
    for _ in range(repeat):
        for si, (board_cards, hero_cards, street, pot, bet, stacks, actions) in \
                enumerate(_SAMPLES):
            board_imgs = [render_card(c[0], c[1]) if c else render_empty_slot()
                          for c in board_cards]
            hero_imgs = [render_card(c[0], c[1]) for c in hero_cards]
            img = build_frame(board_imgs, hero_imgs, pot, bet, stacks, actions)
            obs = eng.process(_frame(img, seq), tm)
            seq += 1
            # sample_id repeats across `repeat` loops AND the image content is
            # identical -> duplicate detection flags this synthetic set.
            e = _compare(obs, board_cards, hero_cards, street,
                         pot, bet, stacks, actions,
                         sample_id=f"syn-{si}", source="synthetic-render")
            e["_image_sha"] = _image_sha(img)
            entries.append(e)
    entries.extend(_run_negatives(eng, tm))
    return entries


def _run_negatives(eng, tm) -> list[dict]:
    """Deterministic negative scenarios: fields expected absent -> truth=None.

    - no expected bet -> bet_size must not be hallucinated
    - no expected action -> action must not be hallucinated
    - no card present (empty board) -> board card must not be hallucinated
    - empty-seat stack absent -> stack must not be hallucinated

    (hero_cards "not dealt" is intentionally omitted: in the hero's point of
    view a hero hand always exists, so it is not a semantically valid negative
    state — it would only inflate n_negative artificially.)
    """
    seq = 1000
    out: list[dict] = []

    # 1) no bet expected: bet ROI blank (no bet on this street)
    board_cards, hero_cards, street, pot, _bet, stacks, actions = _SAMPLES[0]
    board_imgs = [render_card(c[0], c[1]) if c else render_empty_slot()
                  for c in board_cards]
    hero_imgs = [render_card(c[0], c[1]) for c in hero_cards]
    img = build_frame(board_imgs, hero_imgs, pot, "", stacks, actions)
    obs = eng.process(_frame(img, seq), tm)
    out.append({
        "_sample_id": "neg-bet", "_source": "synthetic-render",
        "_image_sha": _image_sha(img),
        "bet_size": {
            "truth": None,
            "pred": (
                str(obs.bet_size.value) if obs.bet_size.value is not None else None
            ),
            "status": obs.bet_size.validation_status.value,
        },
    })
    seq += 1

    # 2) no action expected: no per-seat action signal rendered
    img = build_frame(board_imgs, hero_imgs, pot, "5", stacks, ())
    obs = eng.process(_frame(img, seq), tm)
    act_vals = [s.field.value for s in obs.slot_actions]
    act_status = [s.field.validation_status.value for s in obs.slot_actions]
    out.append({
        "_sample_id": "neg-action", "_source": "synthetic-render",
        "_image_sha": _image_sha(img),
        "action": {
            "truth": None,
            "pred": (
                tuple(sorted(str(v.value) for v in act_vals if v is not None))
                or None
            ),
            "status": _aggregate_status(act_status) if act_status else "unknown",
        },
    })
    seq += 1

    # 3) no card present: empty board (pred should be empty, no hallucination)
    img = build_frame(
        [render_empty_slot() for _ in range(5)], hero_imgs,
        pot, "5", stacks, actions,
    )
    obs = eng.process(_frame(img, seq), tm)
    out.append({
        "_sample_id": "neg-board", "_source": "synthetic-render",
        "_image_sha": _image_sha(img),
        "board_cards": {
            "truth": None,
            "pred": (
                _cards_str(obs.board_cards.value) if obs.board_cards.value else tuple()
            ),
            "status": obs.board_cards.validation_status.value,
        },
    })
    seq += 1

    # 4) empty-seat stack absent: one stack ROI blank -> must not hallucinate
    #    (semantically valid: an empty seat has no stack value)
    img = build_frame(board_imgs, hero_imgs, pot, "5", ("", "200", "300"), actions)
    obs = eng.process(_frame(img, seq), tm)
    stack_vals = {s.slot_id: s.field.value for s in obs.slot_stacks}
    stack_status = {s.slot_id: s.field.validation_status.value
                    for s in obs.slot_stacks}
    seat1 = stack_vals.get(1)
    out.append({
        "_sample_id": "neg-stack", "_source": "synthetic-render",
        "_image_sha": _image_sha(img),
        "stack": {
            "truth": None,
            "pred": str(seat1) if seat1 is not None else None,
            "status": stack_status.get(1, "unknown"),
        },
    })
    return out


def _aggregate_status(statuses) -> str:
    """Aggregate per-slot statuses into one field-level status.

    Priority: conflict > unknown > valid. (Used for multi-slot fields like
    stack/action whose per-slot status must collapse to one value.)
    """
    vals = set(statuses)
    if "conflict" in vals:
        return "conflict"
    if "unknown" in vals:
        return "unknown"
    if vals and vals <= {"valid"}:
        return "valid"
    return "unknown"


def _image_sha(img) -> str:
    """Content hash of a frame image, used for duplicate-content detection."""
    data = img.tobytes() if hasattr(img, "tobytes") else bytes(img)
    return hashlib.sha256(data).hexdigest()


def _compare(obs, board_cards, hero_cards, street, pot, bet, stacks, actions,
             sample_id="", source="synthetic-render"):
    """Build a GT-comparison entry (truth + pred + status) for all fields.

    ``status`` carries the field's ``validation_status.value`` EXPLICITLY, so
    the metric layer scores by status rather than inferring from pred.
    ``sample_id`` / ``source`` are dataset metadata used for acceptance
    eligibility (uniqueness + real-platform provenance).
    """
    entry = {"_sample_id": sample_id, "_source": source}

    # hero_cards / board_cards as sorted card strings
    hero_truth = tuple(sorted(hero_cards))
    hero_pred = _cards_str(obs.hero_cards.value) if obs.hero_cards.value else None
    entry["hero_cards"] = {
        "truth": hero_truth, "pred": hero_pred,
        "status": obs.hero_cards.validation_status.value,
    }

    board_truth = tuple(sorted(board_cards))
    board_pred = (
        _cards_str(obs.board_cards.value)
        if obs.board_cards.value
        else tuple()
    )
    entry["board_cards"] = {
        "truth": board_truth, "pred": board_pred,
        "status": obs.board_cards.validation_status.value,
    }

    # street
    street_truth = _street_enum(street).value
    entry["street"] = {
        "truth": street_truth,
        "pred": (obs.street.value.value if obs.street.value else None),
        "status": obs.street.validation_status.value,
    }

    # pot / bet_size
    entry["pot"] = {
        "truth": pot,
        "pred": (str(obs.pot.value) if obs.pot.value is not None else None),
        "status": obs.pot.validation_status.value,
    }
    entry["bet_size"] = {
        "truth": bet,
        "pred": (str(obs.bet_size.value) if obs.bet_size.value is not None else None),
        "status": obs.bet_size.validation_status.value,
    }

    # stack (compare per slot, ordered by slot_id)
    stack_pred = {s.slot_id: (str(s.field.value) if s.field.value is not None else None)
                  for s in obs.slot_stacks}
    stack_status = {s.slot_id: s.field.validation_status.value
                    for s in obs.slot_stacks}
    entry["stack"] = {
        "truth": tuple(stacks),
        "pred": tuple(str(stack_pred.get(i + 1)) for i in range(len(stacks))),
        "status": _aggregate_status(
            stack_status.get(i + 1, "unknown") for i in range(len(stacks))
        ),
    }

    # action (compare per slot, ordered by slot_id; lowercase .value matches GT)
    act_pred = {s.slot_id: (s.field.value.value if s.field.value else None)
                for s in obs.slot_actions}
    act_status = {s.slot_id: s.field.validation_status.value
                  for s in obs.slot_actions}
    acts_lower = tuple(a.lower() for a in actions)
    entry["action"] = {
        "truth": acts_lower,
        "pred": tuple(str(act_pred.get(i + 1)) for i in range(len(actions))),
        "status": _aggregate_status(
            act_status.get(i + 1, "unknown") for i in range(len(actions))
        ),
    }
    return entry


def _run_real_obj(data: dict) -> list[dict]:
    """Real adapter core (operates on an already-loaded Golden descriptor)."""
    import cv2

    eng = build_engine()
    tm = table_map()
    entries = []
    for i, sample in enumerate(data.get("samples", [])):
        # Validate platform/layout metadata: a sample labelled for another
        # platform/layout must FAIL FAST, never silently run on wrong assets.
        sp = sample.get("platform_id")
        sl = sample.get("layout_id")
        if sp != tm.platform_id:
            raise ValueError(
                f"sample {i} platform_id {sp!r} != active table_map "
                f"{tm.platform_id!r}"
            )
        if sl != tm.layout_id:
            raise ValueError(
                f"sample {i} layout_id {sl!r} != active table_map "
                f"{tm.layout_id!r}"
            )

        img = cv2.imread(sample["image_path"])
        if img is None:
            raise FileNotFoundError(f"cannot load {sample['image_path']}")
        h, w = img.shape[:2]
        frame = Frame(
            frame_seq=i,
            timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
            window_id=sample.get("window_id", "real"),
            window_rect=WindowRect(0, 0, w, h),
            image=img, width=w, height=h,
        )
        obs = eng.process(frame, tm)
        gt = sample["ground_truth"]
        source = sample.get("source", "synthetic-render")
        # stable sample_id MUST come from the descriptor (fail fast if absent)
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"sample {i} must carry a non-empty str 'sample_id'"
            )
        e = _compare_real(obs, gt, sample_id=sample_id, source=source)
        e["_image_sha"] = _image_sha(img)
        entries.append(e)
    return entries


def _compare_real(obs, gt: dict, sample_id: str = "",
                  source: str = "synthetic-render") -> dict:
    """Compare a RawObservation against a real Golden ground-truth dict.

    Each present gt key is compared; absent keys are skipped (not faked).
    ``sample_id`` / ``source`` are dataset metadata for acceptance eligibility.
    Supports the full field set: hero_cards, board_cards, street, pot,
    bet_size, stack, action. ``status`` carries the field's validation_status.
    """
    entry = {"_sample_id": sample_id, "_source": source}

    if "street" in gt:
        truth = gt["street"]
        if isinstance(truth, str):
            truth = truth.lower()
        entry["street"] = {
            "truth": truth,
            "pred": (obs.street.value.value if obs.street.value else None),
            "status": obs.street.validation_status.value,
        }

    if "pot" in gt:
        entry["pot"] = {
            "truth": str(gt["pot"]),
            "pred": (str(obs.pot.value) if obs.pot.value is not None else None),
            "status": obs.pot.validation_status.value,
        }

    if "bet_size" in gt:
        entry["bet_size"] = {
            "truth": str(gt["bet_size"]),
            "pred": (
                str(obs.bet_size.value)
                if obs.bet_size.value is not None
                else None
            ),
            "status": obs.bet_size.validation_status.value,
        }

    if "hero_cards" in gt:
        hero_truth = tuple(sorted(gt["hero_cards"]))
        hero_pred = _cards_str(obs.hero_cards.value) if obs.hero_cards.value \
            else None
        entry["hero_cards"] = {
            "truth": hero_truth, "pred": hero_pred,
            "status": obs.hero_cards.validation_status.value,
        }

    if "board_cards" in gt:
        board_truth = tuple(sorted(gt["board_cards"]))
        board_pred = _cards_str(obs.board_cards.value) if obs.board_cards.value \
            else tuple()
        entry["board_cards"] = {
            "truth": board_truth, "pred": board_pred,
            "status": obs.board_cards.validation_status.value,
        }

    if "stack" in gt:
        stack_pred = {s.slot_id: (str(s.field.value)
                                  if s.field.value is not None else None)
                      for s in obs.slot_stacks}
        stack_status = {s.slot_id: s.field.validation_status.value
                        for s in obs.slot_stacks}
        stack_truth = tuple(gt["stack"])
        entry["stack"] = {
            "truth": tuple(str(v) for v in stack_truth),
            "pred": tuple(str(stack_pred.get(i + 1))
                          for i in range(len(stack_truth))),
            "status": _aggregate_status(
                stack_status.get(i + 1, "unknown")
                for i in range(len(stack_truth))
            ),
        }

    if "action" in gt:
        act_pred = {s.slot_id: (s.field.value.value if s.field.value else None)
                    for s in obs.slot_actions}
        act_status = {s.slot_id: s.field.validation_status.value
                      for s in obs.slot_actions}
        acts_lower = tuple(str(a).lower() for a in gt["action"])
        entry["action"] = {
            "truth": acts_lower,
            "pred": tuple(str(act_pred.get(i + 1))
                          for i in range(len(gt["action"]))),
            "status": _aggregate_status(
                act_status.get(i + 1, "unknown")
                for i in range(len(gt["action"]))
            ),
        }

    return entry


def _run_real(golden_json: str) -> list[dict]:
    """Load a real Golden descriptor JSON and run the adapter."""
    with open(golden_json, encoding="utf-8") as f:
        data = json.load(f)
    return _run_real_obj(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--out", default="benchmark-results.json")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--golden", default=None,
                    help="real golden JSON file (required for --mode real)")
    args = ap.parse_args()

    from benchmark_vision import evaluate

    if args.mode == "real":
        if not args.golden:
            print("error: --golden required for --mode real", file=sys.stderr)
            return 2
        entries = _run_real(args.golden)
        label = "real-adapter-smoke"
    else:
        entries = _run_synthetic(args.repeat)
        label = "synthetic"

    result = evaluate(entries, label)

    plain = result.to_dict()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2)
    print(json.dumps(plain, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
