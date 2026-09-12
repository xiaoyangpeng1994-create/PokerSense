"""Audit scene support against legacy menu candidates, not an independent gold set."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from poker_engine.perceptual.vision.supported_scene import GreenTableSupport
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import read_image


def run(dataset, reference, output):
    if output.exists():
        raise ValueError("preserve earlier audits")
    labels_path = dataset / "labels/frames.jsonl"
    label_hash, reference_hash = sha256_file(labels_path), sha256_file(reference)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    guard = GreenTableSupport(read_image(reference))
    seen, rows = set(), []
    for label in labels:
        if label["scene"] != "menu" or label["sha256"] in seen:
            continue
        path = dataset / "normalized/frames" / label["frame"]
        if sha256_file(path) != label["sha256"]:
            raise ValueError("legacy input changed")
        seen.add(label["sha256"])
        rows.append({"frame": label["frame"], "sha256": label["sha256"],
                     **asdict(guard.recognize(read_image(path)))})
    output.mkdir(parents=True)
    report = {"scope": "deduplicated legacy menu labels; inspect accepts manually",
              "rows": rows, "counts": Counter(r["reason"] for r in rows),
              "labels_sha256": label_hash, "reference_sha256": reference_hash,
              "independent_holdout": False}
    if (sha256_file(labels_path) != label_hash
            or sha256_file(reference) != reference_hash):
        raise ValueError("input changed")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    source = Path(__file__).resolve().parents[1] / (
        "src/poker_engine/perceptual/vision/supported_scene.py")
    shutil.copy2(source, output / source.name)
    write_sha256sums(output)
    return report["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.reference, args.output), indent=2))
