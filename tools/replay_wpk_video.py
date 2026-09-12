"""Run archived WPK raw video through the production capture-card pipeline.

Read-only diagnostic: no labels are inferred, no corpus/config is rewritten,
and no live device is opened. These metrics do not establish ground-truth
accuracy, real capture-to-screen latency, or profitability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics

import cv2

from poker_engine.desktop.live import build_pipeline, _resource_root
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize


class ArchivedVideoSource:
    def __init__(self, path: Path, config: NormalizationConfig,
                 start_frame: int, frames: int):
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise ValueError(f"cannot open archived video: {path}")
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._config = config
        self._remaining = frames
        self._seq = 0
        self._base = datetime.now(timezone.utc)

    def next_frame(self):
        if self._remaining <= 0:
            return None
        ok, image = self._capture.read()
        if not ok:
            return None
        image = normalize(image, self._config)
        height, width = image.shape[:2]
        frame = Frame(self._seq,
                      self._base + timedelta(milliseconds=100 * self._seq),
                      "archived-wpk-video", WindowRect(0, 0, width, height),
                      image, width, height)
        self._seq += 1
        self._remaining -= 1
        return frame

    def close(self):
        self._capture.release()


def run(path: Path, start: int, frames: int) -> dict:
    config = NormalizationConfig.from_json((
        _resource_root() / "configs/vision/wepoker_android_capture_card"
        / "normalization.json"
    ).read_text(encoding="utf-8"))
    source = ArchivedVideoSource(path, config, start, frames)
    counts = defaultdict(Counter)
    timings = []
    identities = set()
    consumed = 0
    try:
        pipeline = build_pipeline(source="capture-card", frame_source=source)
        while (step := pipeline.step()) is not None:
            consumed += 1
            for field, status in step.analysis.confidence.field_status:
                counts[field][status] += 1
            counts["equity_availability"][step.analysis.equity.unavailable_reason
                                          or "available"] += 1
            counts["canonical_street"][step.analysis.state.street.value] += 1
            identities.add((step.analysis.state.hand_id,
                            step.analysis.state.state_version))
            timings.append(step.timing("total"))
    finally:
        source.close()
    if not consumed:
        raise ValueError("video yielded no frames at the requested position")
    return {"video": str(path), "requested_start_frame": start,
            "frames_processed": consumed, "canonical_identities": len(identities),
            "field_statuses": dict(counts),
            "processing_median_ms": round(statistics.median(timings), 3),
            "processing_max_ms": round(max(timings), 3),
            "release_eligible": False,
            "limitations": ["No ground-truth event comparison",
                            "OpenCV seek position, not verified raw frame identity",
                            "Diagnostic timestamps use 100ms/frame, not VFR wall time",
                            "No strategy/transport/render/live-capture measurement"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if args.start_frame < 0 or args.frames <= 0:
        parser.error("start-frame must be >=0 and frames must be >0")
    print(json.dumps(run(args.video, args.start_frame, args.frames), indent=2))


if __name__ == "__main__":
    main()
