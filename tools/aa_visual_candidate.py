"""AA-specific offline geometry and positive scene evidence, never live advice.

Physical slots have no inferred poker positions. Scene support is not proof of
participation, mode, field accuracy or readiness for strategy.
"""

import math

import cv2
import numpy as np

from poker_engine.perceptual.vision.table_map import ROI, ROIKind, TableMap
from tools.wpk_field_candidate import crop, validate_rect


def validate_layout(profile):
    if (profile["platform_id"] != "aa_android_capture_card"
            or profile["status"] != "CANDIDATE_NOT_PRODUCTION"
            or profile["canvas"] != [498, 1080]):
        raise ValueError("explicit AA candidate profile required")
    slots = [row["slot"] for row in profile["slots"]]
    eight = profile.get("layout_id") == "aa_green_8slots_498x1080_candidate_v1"
    count = 8 if eight else 9
    if any(type(slot) is not int for slot in slots) or slots != list(range(count)):
        name = "eight" if eight else "nine"
        raise ValueError(f"{name} ordered physical slots required")
    if eight and profile["hero_slot"] != 4:
        raise ValueError("eight-slot Hero mapping must be explicit slot4")
    if type(profile["hero_slot"]) is not int or profile["hero_slot"] not in slots:
        raise ValueError("explicit Hero slot required")
    for row in profile["slots"]:
        for kind in ("avatar", "stack"):
            validate_rect(row[kind], profile["canvas"])
    for kind in ("hero_cards", "board_cards", "pot", "room_rules", "table_logo"):
        validate_rect(profile["globals"][kind], profile["canvas"])


def table_map(profile):
    validate_layout(profile)
    rois = []

    def add(kind, rect, slot=None):
        x, y, width, height = rect
        rois.append(ROI(kind, x / 498, y / 1080, width / 498, height / 1080, slot))

    for key in ("hero_cards", "board_cards", "pot"):
        add(ROIKind(key), profile["globals"][key])
    for row in profile["slots"]:
        add(ROIKind.STACK, row["stack"], row["slot"])
    return TableMap(profile["platform_id"], profile["layout_id"], (498, 1080),
                    rois=tuple(rois))


def canvas_ok(image):
    return (isinstance(image, np.ndarray) and image.shape == (1080, 498, 3)
            and image.dtype == np.uint8)


class AASceneCandidate:
    """Fixed logo evidence + colour gate; thresholds are NOT calibrated rates."""

    def __init__(self, reference, profile, *, floor=.85):
        validate_layout(profile)
        if not canvas_ok(reference):
            raise ValueError("AA canvas required")
        if isinstance(floor, bool) or not math.isfinite(floor) or not 0 < floor <= 1:
            raise ValueError("invalid scene floor")
        self.rect = tuple(profile["globals"]["table_logo"])
        self.template = cv2.cvtColor(crop(reference, self.rect), cv2.COLOR_BGR2GRAY)
        if float(self.template.std()) < 3:
            raise ValueError("non-flat reviewed logo required")
        self.template.setflags(write=False)
        self.brand_template = cv2.cvtColor(reference[415:433, 217:281],
                                           cv2.COLOR_BGR2GRAY)
        self.brand_template.setflags(write=False)
        self.floor = floor

    def recognize(self, image):
        result = {"scene": "UNKNOWN", "reason": "unsupported_canvas",
                  "logo_score": None, "green_fraction": None,
                  "participation": "UNKNOWN", "critical_hit_enabled": "UNKNOWN",
                  "critical_hit_triggered": "UNKNOWN", "squid_enabled": "UNKNOWN",
                  "squid_triggered": "UNKNOWN", "strategy_eligible": False}
        if not canvas_ok(image):
            return result
        patch = cv2.cvtColor(crop(image, self.rect), cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image[100:780, 100:400], cv2.COLOR_BGR2HSV)
        green = float(np.mean(cv2.inRange(hsv, (35, 70, 40), (95, 255, 255)) > 0))
        score = (float(cv2.matchTemplate(
            patch, self.template, cv2.TM_CCOEFF_NORMED)[0, 0])
                 if float(patch.std()) >= 3 else -1.)
        supported = math.isfinite(score) and score >= self.floor and green >= .45
        brand = cv2.cvtColor(image[415:433, 217:281], cv2.COLOR_BGR2GRAY)
        brand_score = (float(cv2.matchTemplate(
            brand, self.brand_template, cv2.TM_CCOEFF_NORMED)[0, 0])
                       if min(float(brand.std()), float(self.brand_template.std())) >= 3
                       else -1.)
        supported = supported or (math.isfinite(brand_score)
                                  and brand_score >= self.floor and green >= .45)
        result.update(scene="AA_TABLE_CANDIDATE" if supported else "UNKNOWN",
                      reason="positive_aa_brand_and_colour" if supported
                      else "unsupported_or_occluded", logo_score=score,
                      brand_score=brand_score,
                      green_fraction=green)
        return result


def render_geometry(image, profile):
    validate_layout(profile)
    if not canvas_ok(image):
        raise ValueError("AA canvas required")
    result = image.copy()
    for row in profile["slots"]:
        for key in ("avatar", "stack"):
            x, y, width, height = row[key]
            cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1),
                          (0, 255, 255), 1)
            cv2.putText(result, f'{row["slot"]}:{key}', (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .28, (0, 255, 255), 1)
    for key, (x, y, width, height) in profile["globals"].items():
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1),
                      (255, 0, 255), 1)
    return result
