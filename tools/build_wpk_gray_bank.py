"""Propose source-bound session001 gray glyphs; visual review is mandatory."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.probe_wpk_field_candidate import save_image
from tools.wpk_field_candidate import crop
from tools.wpk_gray_amount import gray_glyphs
from tools.wpk_video_dataset import read_image


def build(dataset, profile_path, output, *, field="stack", supplement=None):
    if output.exists():
        raise ValueError("preserve all earlier banks")
    labels_path = dataset / "labels/frames.jsonl"
    source_hash = sha256_file(labels_path)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    supplement_hash = sha256_file(supplement) if supplement is not None else None
    if supplement is not None:
        if field != "pot":
            raise ValueError("supplement currently supports only reviewed pot crops")
        extra = json.loads(supplement.read_text())
        if any(row["session_id"] != "session_001" for row in extra):
            raise ValueError("training supplement cannot include session002")
        labels += extra
    profile = json.loads(profile_path.read_text())
    output.mkdir(parents=True)
    records, features, seen, rejected = [], [], {}, Counter()
    for label in labels:
        if label["session_id"] != "session_001" or not label["stable"]:
            continue
        image_path = dataset / "normalized/frames" / label["frame"]
        digest = sha256_file(image_path)
        if digest != label["sha256"]:
            raise ValueError("source label image hash mismatch")
        image = read_image(image_path)
        targets = ([{"slot_id": None, "stack": label["pot"]}]
                   if field == "pot" else label["slots"])
        for slot in targets:
            seat = slot["slot_id"]
            # Avoid session-specific Hero layout and unknown labels.
            if seat == 0 or slot["stack"]["status"] != "VALID":
                continue
            text = str(slot["stack"]["value"])
            if not text.isdigit():
                continue
            rect = ([217, 284, 64, 21] if field == "pot"
                    else profile["slots"][seat]["stack"])
            patch = crop(image, rect)
            glyphs, reason = gray_glyphs(patch)
            if reason or len(glyphs) != len(text):
                rejected[reason or "length_mismatch"] += 1
                continue
            for digit, (feature, glyph_rect) in zip(text, glyphs):
                key = (digit, feature.tobytes())
                origin = {"frame": label["frame"], "hand_id": label["hand_id"],
                          "image_sha256": digest, "slot": seat, "amount": text,
                          "roi": rect, "glyph_rect": glyph_rect}
                if key in seen:
                    records[seen[key]]["origins"].append(origin)
                    continue
                identifier = len(records)
                seen[key] = identifier
                records.append({"id": identifier, "label": digit,
                                "review": "PENDING", "origins": [origin]})
                features.append(feature)
    np.savez_compressed(output / "proposals.npz", features=np.array(features),
                        labels=np.array([r["label"] for r in records]))
    # Each numbered glyph is large enough for review; this is not a new sample.
    for start in range(0, len(records), 120):
        batch = records[start:start + 120]
        sheet = np.zeros((((len(batch) + 11) // 12) * 76, 12 * 66, 3), np.uint8)
        for offset, row in enumerate(batch):
            y, x = offset // 12 * 76, offset % 12 * 66
            cv2.putText(sheet, f'{row["id"]}:{row["label"]}', (x + 2, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            gray = np.rint(features[row["id"]] * 255).astype(np.uint8)
            sheet[y + 18:y + 74, x + 5:x + 61] = cv2.cvtColor(
                cv2.resize(gray, (56, 56)), cv2.COLOR_GRAY2BGR)
        save_image(output / f"review-{start:04d}.png", sheet)
    report = {"source_labels_sha256": source_hash,
              "session": "session_001", "field": field,
              "supplement_sha256": supplement_hash,
              "profile_sha256": sha256_file(profile_path), "records": records,
              "rejected": rejected, "counts": Counter(r["label"] for r in records),
              "usable_without_review": False}
    if sha256_file(labels_path) != source_hash:
        raise ValueError("labels changed")
    if supplement is not None and sha256_file(supplement) != supplement_hash:
        raise ValueError("supplement changed")
    (output / "inventory.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return {"unique_proposals": len(records), "counts": report["counts"],
            "rejected": rejected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--field", choices=("stack", "pot"), default="stack")
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.profile, args.output, field=args.field,
                           supplement=args.supplement),
                     indent=2))
