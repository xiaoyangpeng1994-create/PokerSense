"""Offline AA8 literal special-mode evidence; no inferred fees or triggers.

Templates are development witnesses, not an OCR engine or independent test.
Missing matching text is UNKNOWN, never proof that a mode is inactive.
"""

import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_action_transfer import inventory, load, sha


AUDIT = "ce14fe204a2228ecc87698e14944ea41df0f7ee6b41eef0d647cba68424a7e61"
# Fixed AA8 498x1080 development layout, independently reviewed source literals.
REGIONS = {
    "insurance_banner": (207, 602, 292, 626),
    "insurance_purchase_countdown": (420, 572, 463, 590),
    "insurance_purchase_6": (434, 572, 482, 590),
    "insurance_purchase_prefix": (441, 573, 468, 589),
    "mushroom_counter_icon": (19, 95, 56, 132),
    "mushroom_3bb_rule": (184, 673, 244, 689),
    "critical_hit_7bb_rule": (256, 673, 315, 689),
}
SOURCES = {key: 4890 for key in REGIONS}
SOURCES["insurance_banner"] = 4824
SOURCES["insurance_purchase_countdown"] = 4824


def patch(image, box):
    if image is None or image.shape != (1080, 498, 3):
        raise ValueError("normalized eight-seat canvas required")
    x1, y1, x2, y2 = box
    return cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def score(first, second):
    if first.shape != second.shape or min(first.std(), second.std()) < 3:
        return 0.0
    return max(0., float(cv2.matchTemplate(
        first, second, cv2.TM_CCOEFF_NORMED)[0, 0]))


def numeric_candidate(image, box, bank):
    """Frozen-bank candidate; refuse clipped ink instead of padding it away."""
    if bank is None:
        return {"value": None, "reason": "no_amount_bank"}
    value = patch(image, box)
    _, binary = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = (binary[0], binary[-1], binary[:, 0], binary[:, -1])
    if any(np.any(edge) for edge in edges):
        return {"value": None, "reason": "clipped_or_border"}
    return asdict(bank.diagnose(np.pad(value, 2, constant_values=int(value.min()))))


class SpecialModeReader:
    def __init__(self, images, floor=.96, amount_bank=None, mushroom_references=()):
        if not .9 <= floor <= 1:
            raise ValueError("conservative template floor required")
        self.floor = floor
        self.amount_bank = amount_bank
        self.templates = {k: patch(images[k], box) for k, box in REGIONS.items()}
        self.mushroom_templates = [self.templates["mushroom_counter_icon"]] + [
            patch(image, REGIONS["mushroom_counter_icon"])
            for image in mushroom_references]
        if any(v.std() < 3 for v in self.templates.values()):
            raise ValueError("empty template")

    def recognize(self, image):
        values = {k: score(patch(image, box), self.templates[k])
                  for k, box in REGIONS.items()}
        values["mushroom_counter_icon"] = max(score(
            patch(image, REGIONS["mushroom_counter_icon"]), template)
            for template in self.mushroom_templates)
        visible = {k for k, v in values.items() if v >= self.floor}
        insurance = bool(visible & {
            "insurance_banner", "insurance_purchase_countdown", "insurance_purchase_6",
            "insurance_purchase_prefix"})
        prefix = "insurance_purchase_prefix" in visible
        amount = numeric_candidate(image, (468, 573, 479, 589), self.amount_bank) if (
            prefix) else {"value": None, "reason": "no_purchase_prefix"}
        counter = numeric_candidate(image, (58, 111, 96, 128), self.amount_bank) if (
            "mushroom_counter_icon" in visible) else {
                "value": None, "reason": "no_mushroom_icon"}
        return {
            "literal_evidence": sorted(visible), "scores": values,
            "insurance": "VISIBLE" if insurance else "UNKNOWN",
            "insurance_purchase_label": "投保6" if (
                "insurance_purchase_6" in visible) else None,
            "insurance_label_slot": 3 if "insurance_purchase_6" in visible else None,
            "insurance_purchase_prefix": "投保" if prefix else None,
            "insurance_amount_candidate": amount,
            "insurance_amount_is_verified_cash_movement": False,
            "mushroom_counter_candidate": counter,
            "mushroom_counter_semantics": "DISPLAYED_COUNTER_NOT_ACTIVATION",
            "mushroom_rule": "3BB" if "mushroom_3bb_rule" in visible else None,
            "critical_hit_rule": "7BB" if "critical_hit_7bb_rule" in visible else None,
            "mushroom_trigger": "UNKNOWN", "critical_hit_trigger": "UNKNOWN",
            "squid": "NOT_IMPLEMENTED_NO_RECORDING_EVIDENCE",
            "wager_actions": [], "fee_policy_verified": False,
            "strategy_eligible": False,
        }


def separate_cash(total_debit, pot_increase, evidenced_nonwager=()):
    """Pure accounting. Only explicit evidenced movements get a category.

    A rule label, insurance label or arithmetical 3BB match is NOT cash evidence.
    Callers must supply independently verified ledger movement IDs/amounts.
    """
    def amount(value):
        if not isinstance(value, str):
            raise ValueError("explicit decimal text required")
        result = Decimal(value)
        if not result.is_finite() or result < 0:
            raise ValueError("nonnegative finite money required")
        return result

    debit, pot = amount(total_debit), amount(pot_increase)
    allocations, seen = [], set()
    for item in evidenced_nonwager:
        if (item.get("kind") not in {
                "insurance_premium", "mushroom", "critical_hit", "rake"}
                or item.get("basis") != "verified_cash_movement"
                or not isinstance(item.get("evidence_id"), str)
                or not item["evidence_id"] or item["evidence_id"] in seen):
            raise ValueError("unique verified cash movement evidence required")
        seen.add(item["evidence_id"])
        allocations.append({**item, "amount": str(amount(item["amount"]))})
    allocated = sum((Decimal(i["amount"]) for i in allocations), Decimal(0))
    residual = debit - pot - allocated
    return {"allocations": allocations, "unallocated_difference": str(residual),
            "status": "RECONCILED" if residual == 0 else "UNALLOCATED",
            "automatic_rake_inference": False, "strategy_eligible": False}


def run(source, pools, output):
    rows = inventory(source, AUDIT)
    reader = SpecialModeReader({key: load(source, rows[frame])
                                for key, frame in SOURCES.items()})
    predictions = []
    for pool in pools:
        for frame, row in inventory(pool, AUDIT).items():
            value = reader.recognize(load(pool, row))
            predictions.append({"frame": frame,
                                "source_sha256": row["sha256"], **value})
    report = {
        "schema": "aa8_special_mode_dev_v1", "frame_count": len(predictions),
        "templates": {k: {"frame": SOURCES[k],
                          "source_sha256": rows[SOURCES[k]]["sha256"],
                          "mask_sha256": hashlib.sha256(v.tobytes()).hexdigest()}
                      for k, v in reader.templates.items()},
        "visible_counts": {key: sum(key in p["literal_evidence"]
                                    for p in predictions)
                           for key in REGIONS},
        "implementation_sha256": sha(Path(__file__)), "floor": reader.floor,
        "independent_acceptance": False, "full_visual_acceptance": False,
        "limitations": ["same-development literal templates",
                        "fixed slot3 purchase6 only",
                        "no trigger recognition", "no cash movement attribution"],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.jsonl").write_text("\n".join(
        json.dumps(p, ensure_ascii=False) for p in predictions) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.pool, args.output), indent=2))
