"""Minimal engineering calibration tool for authoring TableMap JSON.

Usage (manual, on a real host, with the ``calibrate`` extra installed):

    python tools/calibrate.py \
        --image /path/to/table_screenshot.png \
        --platform wpk \
        --layout 6max_default \
        --out configs/platform/wpk__6max_default.json

Flow:
1. Load a full-table screenshot.
2. For each ROI you want to define, drag a rectangle with cv2.selectROI.
3. Global ROIs (hero_cards/board_cards/pot/bet_size/dealer/actor) have slot=None.
   Per-seat ROIs (stack/action) prompt for a slot_id.
4. Coordinates are normalized against the image size (image is treated as the
   reference_size for the mapping).
5. Output is written as JSON and reloaded for validation before saving.

This is an engineering tool only — NOT a product UI, NOT Vision recognition.
"""

from __future__ import annotations

import argparse
import sys

import cv2  # noqa  (calibrate extra)

sys.path.insert(0, "src")

from poker_engine.perceptual import ROIKind, ROI, TableMap  # noqa: E402

_GLOBAL_KINDS = [
    ROIKind.HERO_CARDS,
    ROIKind.BOARD_CARDS,
    ROIKind.POT,
    ROIKind.BET_SIZE,
    ROIKind.DEALER,
    ROIKind.ACTOR,
]


def _select_roi(image, kind: ROIKind) -> ROI:
    win_name = f"select {kind.value}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image)
    r = cv2.selectROI(win_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_name)
    x, y, w, h = r
    if w == 0 or h == 0:
        print(f"skipped {kind.value} (no selection)")
        return None  # type: ignore[return-value]

    slot_id = None
    if kind in (ROIKind.STACK, ROIKind.ACTION):
        slot_id = int(input(f"slot_id for {kind.value}: ").strip())

    ih, iw = image.shape[:2]
    return ROI(
        kind=kind,
        x=x / iw,
        y=y / ih,
        width=w / iw,
        height=h / ih,
        slot_id=slot_id,
    )


def _ask(prompt: str, default: str) -> str:
    v = input(f"{prompt} [{default}]: ").strip()
    return v or default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aspect-tolerance", type=float, default=0.02)
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 1

    ih, iw = image.shape[:2]
    print(f"image {iw}x{ih}")

    rois = []
    for kind in _GLOBAL_KINDS:
        roi = _select_roi(image, kind)
        if roi is not None:
            rois.append(roi)

    while True:
        ans = _ask("add a per-seat STACK/ACTION ROI?", "n")
        if ans.lower() not in ("y", "yes"):
            break
        kind = ROIKind(_ask("kind (stack/action)", "stack"))
        roi = _select_roi(image, kind)
        if roi is not None:
            rois.append(roi)

    table_map = TableMap(
        platform_id=args.platform,
        layout_id=args.layout,
        reference_size=(iw, ih),
        aspect_tolerance=args.aspect_tolerance,
        rois=tuple(rois),
    )

    # Reload from JSON to validate before writing.
    reloaded = TableMap.from_json(table_map.to_json())
    assert reloaded == table_map, "round-trip validation failed"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(table_map.to_json())

    print(f"wrote {args.out} with {len(rois)} ROI(s)")

    # --- overlay verification preview ---
    preview = image.copy()
    ih, iw = image.shape[:2]
    for roi in reloaded.rois:
        x0 = int(roi.x * iw)
        y0 = int(roi.y * ih)
        x1 = int((roi.x + roi.width) * iw)
        y1 = int((roi.y + roi.height) * ih)
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = (
            roi.kind.value if roi.slot_id is None
            else f"{roi.kind.value}:{roi.slot_id}"
        )
        cv2.putText(preview, label, (x0, max(0, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.namedWindow("overlay verification", cv2.WINDOW_NORMAL)
    cv2.imshow("overlay verification", preview)
    print("showing overlay preview — press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
