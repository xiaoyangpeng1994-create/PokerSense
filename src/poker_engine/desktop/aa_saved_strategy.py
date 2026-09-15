"""Seed an editable river study from one saved frame; no temporal/live approval."""

from dataclasses import asdict
from pathlib import Path
import threading

import cv2
import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa8_hero_turn import hero_turn_candidate

from .aa_reader import preflight_profile
from .aa_review import ReviewError


class SavedStrategyInputs:
    def __init__(self, profile, bundle_sha256=None):
        self.profile, self.bundle_sha256 = profile, bundle_sha256
        self.bank = None
        self.lock = threading.Lock()

    def from_record(self, review, issue_id):
        record = review.get(issue_id)
        issue = record["issue"]
        snapshot = issue["observation"]
        row = snapshot.get("payload") or {}
        encoded = review.image(issue_id)
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (1080, 498, 3):
            raise ReviewError("固定画面的尺寸或格式不符合AA八座布局")
        with self.lock:
            if self.bank is None:
                status = preflight_profile(
                    self.profile, bundle_sha256=self.bundle_sha256)
                if not status["ready"]:
                    raise ReviewError("识别资源校验未通过")
                with np.load(Path(status["paths"]["bank_path"]),
                             allow_pickle=False) as data:
                    self.bank = GrayAmountRecognizer(data["features"], data["labels"],
                                                     augment=True)
            modes = row.get("special_modes") or {}
            supported = (row.get("scene_supported") is True
                         and not modes.get("block_state_updates")
                         and modes.get("insurance") != "VISIBLE")
            controls = hero_turn_candidate(image) if supported else {}
            price = {"value": None, "reason": "no_controls"}
            if controls.get("hero_turn") is True:
                price = asdict(self.bank.diagnose(
                    image[849:876, 340:397].min(axis=2)))
        return {"source": {"issue_id": issue_id,
                           "source_frame": snapshot.get("source_frame"),
                           "preview_sha256": issue["preview_sha256"],
                           "scope": "SAVED_FRAME_MANUAL_HYPOTHESES"},
                "hero_cards": (row.get("cards") or {}).get("hero"),
                "board_cards": (row.get("cards") or {}).get("board_slots"),
                "pot_before": (row.get("pot") or {}).get("value"),
                "call_cost": price["value"], "call_diagnostic": price,
                "opponents": None, "max_hero_deduction": None,
                "two_frame_confirmation": False, "strategy_eligible": False}
