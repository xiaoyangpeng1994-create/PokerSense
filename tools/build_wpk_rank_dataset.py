"""Build an unapproved, private rank-glyph review set from session_001 only.

Legacy labels are hints, not automatically accepted truth. Training requires
a separate hash-bound visual review. No session_002 imagery is loaded here.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.corner_glyph_recognizer import (
    DEFAULT_GEOMETRY, isolate_glyph, locate_card_face,
)
from poker_engine.perceptual.vision.engine import _crop_slot
from poker_engine.perceptual.vision.fused_card_recognizer import GlyphNormalizer
from poker_engine.perceptual.vision.roi import extract_roi
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, read_image, safe_reference_path


def write_image(path, picture):
    ok, encoded = cv2.imencode(".png", picture)
    if not ok:
        raise ValueError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())


def build(dataset: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve prior dataset")
    labels_path = dataset / "labels/frames.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8")
              .splitlines() if line.strip()]
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    output.mkdir(parents=True)
    (output / "glyphs").mkdir()
    unique, rejected = {}, []
    for label in labels:
        if label["session_id"] != "session_001":
            continue
        image_path = safe_reference_path(dataset / "normalized/frames", label["frame"])
        if sha256_file(image_path) != label["sha256"]:
            raise ValueError("legacy image hash mismatch")
        image = read_image(image_path)
        if image.shape != (1080, 498, 3):
            raise ValueError("unexpected training canvas")
        frame = Frame(0, datetime.now(timezone.utc), "rank-training-review",
                      WindowRect(0, 0, 498, 1080), image, 498, 1080)
        for group in ("hero", "board"):
            field = label[f"{group}_cards"]
            if field["status"] != "VALID":
                continue
            roi = next(r for r in table.rois if r.kind.value == f"{group}_cards")
            base = extract_roi(frame, roi)
            layout = vision._hero_layout if group == "hero" else vision._board_layout
            for slot, code in enumerate(field["value"] or []):
                rank = code[0].upper()
                origin = {"frame": label["frame"], "image_sha256": label["sha256"],
                          "legacy_hand_id": label["hand_id"], "group": group,
                          "slot": slot, "card_hint": code, "rank_hint": rank}
                crop = _crop_slot(base, layout.slots[slot])
                glyph = isolate_glyph(
                    locate_card_face(crop), DEFAULT_GEOMETRY.rank_band)
                canvas = GlyphNormalizer.normalize(glyph)
                if canvas is None:
                    rejected.append({**origin, "reason": "no_normalized_rank"})
                    continue
                digest = pixels_digest(glyph)
                if digest not in unique:
                    identifier = f"r{len(unique):03d}"
                    path = output / "glyphs" / f"{identifier}.png"
                    write_image(path, glyph)
                    unique[digest] = {
                        "id": identifier, "glyph": f"glyphs/{identifier}.png",
                        "glyph_sha256": sha256_file(path), "pixel_sha256": digest,
                        "session": "session_001", "split": "train",
                        "rank_hints": [], "origins": [],
                    }
                item = unique[digest]
                item["rank_hints"] = sorted(set(item["rank_hints"] + [rank]))
                item["origins"].append(origin)
    rows = list(unique.values())
    # Contact sheets contain rank glyphs only, not player names or phone UI.
    for page in range((len(rows) + 47) // 48):
        sheet = np.full((8 * 142, 6 * 150, 3), 245, np.uint8)
        for k, item in enumerate(rows[page * 48:(page + 1) * 48]):
            y, x = (k // 6) * 142, (k % 6) * 150
            glyph = read_image(output / item["glyph"])
            h, w = glyph.shape[:2]
            scale = min(105 / h, 125 / w)
            enlarged = cv2.resize(glyph, (round(w * scale), round(h * scale)),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.putText(sheet, f'{item["id"]}: {"/".join(item["rank_hints"])}',
                        (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
            sheet[y + 29:y + 29 + enlarged.shape[0],
                  x + 5:x + 5 + enlarged.shape[1]] = enlarged
        write_image(output / f"review-{page + 1:02d}.png", sheet)
    repo = Path(__file__).resolve().parents[1]
    files = list((repo / "src/poker_engine/perceptual").rglob("*.py"))
    files += [p for p in (repo / "configs").rglob("*.json")]
    files += [repo / "src/poker_engine/desktop/live.py", Path(__file__)]
    sources = {str(p.relative_to(repo).as_posix()): sha256_file(p) for p in files}
    for relative, digest in sources.items():
        destination = output / "feature-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
        if sha256_file(destination) != digest:
            raise ValueError("feature source changed during snapshot")
    manifest = {"schema_version": 1, "purpose": "rank-only training candidate",
                "requires_visual_review": True, "session_partition": {
                    "train": ["session_001"], "excluded": ["session_002"]},
                "legacy_hand_boundaries_verified": False,
                "source_labels_sha256": sha256_file(labels_path),
                "feature_source_files": sources, "samples": rows,
                "rejected": rejected, "summary": {
                    "unique_glyphs": len(rows),
                    "origin_count": sum(len(r["origins"]) for r in rows),
                    "rank_hint_counts": dict(Counter(
                        rank for r in rows for rank in r["rank_hints"])),
                    "rejected_count": len(rejected)}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
    write_sha256sums(output)
    return manifest["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.output), indent=2))
