"""Hash-bound reviewed-bank loader; recognition lives in the normal vision package."""

import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import (
    GrayAmountRecognizer, GrayRead, gray_glyphs,
)
from tools.capture_card_calibration.hashing import sha256_file

__all__ = ["GrayAmountRecognizer", "GrayRead", "gray_glyphs", "load_reviewed_bank"]


def load_reviewed_bank(folder, *, augment=False):
    folder = Path(folder)
    review = json.loads((folder / "visual-review.json").read_text())
    if (sha256_file(folder / "inventory.json") != review["inventory_sha256"]
            or sha256_file(folder / "proposals.npz") != review["npz_sha256"]):
        raise ValueError("review does not bind this digit bank")
    inventory = json.loads((folder / "inventory.json").read_text())
    with np.load(folder / "proposals.npz", allow_pickle=False) as data:
        features, labels = data["features"], data["labels"]
    if [r["label"] for r in inventory["records"]] != labels.tolist():
        raise ValueError("bank labels disagree with source inventory")
    reviewed, rejected = review["reviewed_ids"], review["rejected_ids"]
    if any(type(i) is not int for i in reviewed + rejected):
        raise ValueError("review ids must be integers")
    if len(reviewed) != len(set(reviewed)) or set(reviewed) != set(range(len(labels))):
        raise ValueError("every proposal must have an explicit review")
    if not set(rejected) <= set(reviewed) or len(rejected) != len(set(rejected)):
        raise ValueError("invalid rejected ids")
    accepted = [i for i in reviewed if i not in rejected]
    return GrayAmountRecognizer(features[accepted], labels[accepted], augment=augment)
