"""One-frame integration of offline WPK candidates through VisionEngine.

No strategy/event-store consumer is allowed: calibrated release eligibility is
always false, regardless of individual candidate matches. All inputs are current
pixels; later reviewed cards/actions/cash never enter process().
"""

from dataclasses import replace
from enum import Enum
import json
from pathlib import Path

from poker_engine.core.enums import ActionType
from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from poker_engine.perceptual.vision.protocols import ActionRecognition
from poker_engine.perceptual.vision.table_map import ROI, ROIKind
from poker_engine.perceptual.vision.supported_scene import GreenTableSupport
from tools.probe_wpk_observation_fields import field_record
from tools.validate_wpk_card_batch import RecordingRecognizer
from tools.wpk_field_candidate import crop, stack_table
from tools.wpk_field_features_v2 import FeatureActionCandidate
from tools.wpk_gray_amount import load_reviewed_bank
from tools.wpk_hero_balance_layout import HeroBalanceLayout, DIGIT_RECTS
from tools.wpk_video_dataset import read_image


def simple_field(field):
    value = field.value.value if isinstance(field.value, Enum) else field.value
    return {"value": value, "status": field.validation_status.value,
            "raw_score": field.evidence.get("raw_score")}


def unavailable_frame(frame, reason):
    unknown = {"value": None, "status": "unknown", "raw_score": 0.}
    return {
        "source_frame": frame.frame_seq, "timestamp": frame.timestamp.isoformat(),
        "hero": [None] * 2, "board": [None] * 5, "pot": unknown.copy(),
        "stacks": {str(i): unknown.copy() for i in range(8)},
        "actions": {str(i): {"value": None, "accepted_candidate": False,
                             "score": 0., "runner_up": 0.} for i in range(8)},
        "production_action_status": {str(i): "unknown" for i in range(8)},
        "production_card_status": {"hero": "unknown", "board": "unknown"},
        "hero_balance_location": None, "hero_layout_status": "UNKNOWN",
        "seat_presence": {str(i): unknown.copy() for i in range(8)},
        "dealer": unknown.copy(), "hero_ui_turn": unknown.copy(),
        "street_observation": unknown.copy(), "scene_reason": reason,
        "candidate_only": True, "release_eligible": False,
        "confidence_calibration_verified": False,
        "participation": "UNKNOWN", "events_emitted": 0,
    }


class FrameActions:
    def __init__(self):
        self.reads = {}

    def recognize(self, patch, slot_id):
        read = self.reads.get(str(slot_id))
        if read is None:
            return ActionRecognition(None, 0.)
        value = ActionType(read["value"]) if read["value"] is not None else None
        return ActionRecognition(value, read["score"], read["runner_up"])


class CombinedVisionCandidate:
    def __init__(self, bank, heads, profile, action_templates, layout_templates,
                 *, action_variants=None, pot_bank=None, scene_reference=None,
                 badge_scale_tolerance=False):
        self.scene = (GreenTableSupport(scene_reference)
                      if scene_reference is not None else None)
        self.table, self.vision = load_calibration(
            CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
        self.table = stack_table(self.table, profile)
        regions = tuple(r for r in self.table.rois if r.kind is not ROIKind.ACTION)
        for row in profile["slots"]:
            x, y, w, h = row["badge"]
            regions += (ROI(ROIKind.ACTION, x / 498, y / 1080, w / 498, h / 1080,
                            row["slot"]),)
        self.table = replace(self.table, rois=regions)
        self.layout = HeroBalanceLayout(layout_templates)
        self.cards = RecordingRecognizer(FusedCardRecognizerAdapter(FusedCardRecognizer(
            load_card_heads(heads), rank_floor=.50, suit_floor=.30)), floor=.30)
        self.vision._card = self.cards
        self.amount = load_reviewed_bank(bank, augment=True)
        self.vision._stack_amount = self.amount
        if pot_bank is not None:
            self.vision._amount = load_reviewed_bank(pot_bank, augment=True)
            remaining = tuple(r for r in self.table.rois if r.kind is not ROIKind.POT)
            pot_roi = ROI(ROIKind.POT, 217 / 498, 284 / 1080, 64 / 498, 21 / 1080)
            self.table = replace(self.table, rois=remaining + (pot_roi,))
        self.actions = FeatureActionCandidate(
            profile, action_templates, variants=action_variants,
            badge_scale_tolerance=badge_scale_tolerance)
        self.action_adapter = FrameActions()
        self.vision._action = self.action_adapter

    def reset(self):
        self.cards.reset()
        self.action_adapter.reads.clear()

    def process(self, frame):
        self.cards.reads.clear()
        self.action_adapter.reads.clear()
        if frame.image.shape != (1080, 498, 3):
            self.reset()
            raise ValueError("unreviewed combined canvas")
        if self.scene is not None:
            support = self.scene.recognize(frame.image)
            if not support.supported:
                self.reset()
                return unavailable_frame(frame, support.reason)
        layout = self.layout.recognize(frame.image)
        self.action_adapter.reads = self.actions.recognize(frame.image)
        rois = tuple(r for r in self.table.rois
                     if not (r.kind is ROIKind.STACK and r.slot_id == 0))
        if layout.location is not None:
            x, y, w, h = DIGIT_RECTS[layout.location]
            rois += (ROI(ROIKind.STACK, x / 498, y / 1080, w / 498, h / 1080, 0),)
        # Actual VisionEngine routing/protocols, not a review-fed state replay.
        observed = self.vision.process(frame, replace(self.table, rois=rois))
        return {
            "source_frame": frame.frame_seq, "timestamp": frame.timestamp.isoformat(),
            "hero": [self.cards.reads.get(("hero", i), {}).get("card")
                     for i in range(2)],
            "board": [self.cards.reads.get(("board", i), {}).get("card")
                      for i in range(5)],
            "pot": field_record(observed.pot),
            "stacks": {str(s.slot_id): field_record(s.field)
                       for s in observed.slot_stacks},
            "actions": self.action_adapter.reads.copy(),
            "production_action_status": {str(s.slot_id): s.field.validation_status.value
                                         for s in observed.slot_actions},
            "production_card_status": {
                "hero": observed.hero_cards.validation_status.value,
                "board": observed.board_cards.validation_status.value,
            },
            "hero_balance_location": layout.location,
            "hero_layout_status": layout.status,
            "seat_presence": {str(s.slot_id): simple_field(s.field)
                              for s in observed.slot_occupancies},
            "dealer": simple_field(observed.dealer_pos),
            "hero_ui_turn": simple_field(observed.actor),
            "street_observation": simple_field(observed.street),
            "candidate_only": True, "release_eligible": False,
            "scene_reason": "supported_layout_candidate",
            "confidence_calibration_verified": False,
            "participation": "UNKNOWN", "events_emitted": 0,
        }


def from_artifacts(corpus, bank, *, check_variant=None, badge_scale_tolerance=False):
    repo = Path(__file__).resolve().parents[1]
    profile = json.loads((repo / "tests/fixtures/wpk_reference_hands/"
                          "field_candidate_v1.json").read_text())
    templates = {p.stem: read_image(p) for p in
                 (repo / "configs/vision/wepoker_android_capture_card/action_glyph")
                 .glob("*.png")}
    for row in profile["new_templates"]:
        image = read_image(corpus / "hands/aq_allin_8240_8900/frames"
                           / f'{row["source_frame"]:06d}.png')
        templates[row["action"]] = crop(image, row["rect"])
    from tools.wpk_hero_balance_layout import LOCATIONS
    folder = corpus / "transitions/scene_10400_11500/frames"
    layout = {key: crop(read_image(folder / f"{index:06d}.png"), LOCATIONS[key])
              for key, index in (("lower", 11000), ("raised", 10700))}
    extra = crop(read_image(folder / "011250.png"), (426, 443, 51, 23))
    variants = {"all_in": [extra]}
    if check_variant is not None:
        variants["check"] = [check_variant]
    heads = corpus / "training/rank_v6_candidate_01/card_heads.npz"
    pot_bank = corpus / "gray_amount/session001_pot_proposals_v2"
    scene_reference = read_image(corpus / "hands/aq_allin_8240_8900/frames/008500.png")
    return CombinedVisionCandidate(
        bank, heads, profile, templates, layout,
        action_variants=variants, pot_bank=pot_bank,
        scene_reference=scene_reference, badge_scale_tolerance=badge_scale_tolerance)
