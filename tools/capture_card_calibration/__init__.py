"""Capture-card calibration toolkit.

Implements the hardware-independent parts of
``docs/capture-card-calibration-guide.zh-CN.md``: dataset scaffolding,
label schemas, geometry drafts, coverage checks, split generation and the
acceptance report.

What this toolkit deliberately does NOT do:

- It does not produce calibration evidence on its own. Stages A (freeze
  hardware) and B (record 45-90 minutes of real hands through a capture
  card) require real hardware and a human operator.
- It does not inherit anything from the ``wepoker_android`` (LDPlayer) or
  ``wepoker`` (H5) platforms. Guide rules 1 and 2 forbid reusing their
  ROIs, coordinates, thresholds, sample counts or confidence conclusions.
- It never upgrades an unmet requirement into a pass. Every checker
  reports gaps explicitly so the final status stays ``PARTIAL`` or
  ``BLOCKED`` until the evidence actually exists.

Modules are import-safe without OpenCV; only the recording and frame
extraction helpers (which need a real device or a video file) import cv2.
"""

__all__ = ["SCHEMA_VERSION", "SLOT_COUNT"]

SCHEMA_VERSION = 1
SLOT_COUNT = 8
