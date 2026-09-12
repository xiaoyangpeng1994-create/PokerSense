"""Hash-checked full-canvas development-only sheets for human mode review."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_action_transfer import load
from tools.aa8_special_modes import AUDIT


def run(pool, output):
    manifest = json.loads((pool / "samples.json").read_text())
    rows = manifest["samples"]
    if manifest["audit_sha256"] != AUDIT or any(
            r["role"] != "development" or not (
                0 <= float(r["pts_seconds"]) < 300 or
                820 <= float(r["pts_seconds"]) < 995.366) for r in rows):
        raise ValueError("development-only source required")
    output.mkdir(parents=True, exist_ok=False)
    tiles = []
    for row in rows:
        image = cv2.resize(load(pool, row), (249, 540), interpolation=cv2.INTER_AREA)
        tile = np.pad(image, ((25, 0), (0, 0), (0, 0)))
        cv2.putText(tile, f"f{row['global_frame']} t{row['pts_seconds']}", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
        tiles.append(tile)
    for offset in range(0, len(tiles), 8):
        group = tiles[offset:offset + 8]
        group += [np.zeros_like(tiles[0])] * (8 - len(group))
        sheet = np.vstack([np.hstack(group[:4]), np.hstack(group[4:])])
        ok, encoded = cv2.imencode(".png", sheet)
        if not ok:
            raise ValueError("encoding failed")
        (output / f"sheet_{offset // 8:02d}.png").write_bytes(encoded.tobytes())
    (output / "sources.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pool, args.output)
