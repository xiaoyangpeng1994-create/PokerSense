"""Select diverse AA video scenes for review; clusters are NOT game modes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import verify_sha256sums, write_sha256sums
from tools.wpk_video_dataset import read_image
from tools.aa_data_separation import assert_training_frames


def feature(image):
    whole = cv2.resize(image, (10, 24), interpolation=cv2.INTER_AREA)
    center = cv2.resize(image[250:790], (14, 18), interpolation=cv2.INTER_AREA)
    return np.concatenate((whole.reshape(-1), center.reshape(-1))).astype(np.float32)


def mine(window, output):
    if verify_sha256sums(window):
        raise ValueError("window manifest failure")
    samples = json.loads((window / "samples.json").read_text())
    rows = samples["samples"]
    if not rows:
        raise ValueError("empty source selection")
    reservations = json.loads((Path(__file__).resolve().parents[1] /
                               "configs/reproduction/aa_holdout_reservations_v1.json")
                              .read_text())
    if reservations["source_sha256"] != samples["source_sha256"]:
        raise ValueError("reservation source mismatch")
    assert_training_frames([row["source_frame"] for row in rows], reservations)
    features = []
    for row in rows:
        path = (window / row["file"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("source path escapes window")
        features.append(feature(read_image(path)))
    values = np.asarray(features, dtype=np.float32)
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(
        values, min(40, len(rows)), None,
        (cv2.TERM_CRITERIA_MAX_ITER, 50, .1), 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    distances = np.sum((values - centers[labels]) ** 2, axis=1)
    selected = set()
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        selected.add(int(members[np.argmin(distances[members])]))
    extras = 0
    for index in np.argsort(-distances, kind="stable"):
        frame = rows[int(index)]["source_frame"]
        if all(abs(frame - rows[i]["source_frame"]) >= 60 for i in selected):
            selected.add(int(index))
            extras += 1
            if extras >= 40:
                break
    chosen = sorted(selected, key=lambda i: rows[i]["source_frame"])
    output.mkdir(parents=True, exist_ok=False)
    tiles, review = [], []
    for i in chosen:
        row = rows[i]
        image = read_image(window / row["file"])
        tile = np.zeros((384, 166, 3), np.uint8)
        tile[24:] = cv2.resize(image, (166, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, str(row["source_frame"]), (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(tile)
        review.append({**row, "cluster": int(labels[i]),
                       "special_mode": "UNKNOWN", "event_verified": False})
    for start in range(0, len(tiles), 12):
        batch = tiles[start:start + 12]
        batch += [np.zeros_like(tiles[0])] * (12 - len(batch))
        sheet = np.vstack([np.hstack(batch[i:i + 6]) for i in (0, 6)])
        ok, encoded = cv2.imencode(".png", sheet)
        if not ok:
            raise ValueError("PNG encoding failure")
        (output / f"sheet_{start // 12:02d}.png").write_bytes(encoded.tobytes())
    (output / "review_queue.json").write_text(json.dumps({
        "source_sha256": samples["source_sha256"],
        "purpose": "visual_diversity_mining_not_mode_classification",
        "input_samples": len(rows), "selected": review,
        "excluded_reserved_intervals": samples["excluded_reserved_intervals"],
        "independent_holdout": False}, indent=2))
    write_sha256sums(output)
    print(json.dumps({"input_samples": len(rows), "selected": len(chosen)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mine(args.window, args.output)
