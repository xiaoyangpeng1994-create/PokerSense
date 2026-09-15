"""Explicit AA8 candidate reader for a source-checkout desktop session.

Preflight reads configuration and file metadata only. Constructing AA8Reader
loads the configured development model references; it never opens a device.
Candidates remain distinct from complete legal state and live strategy.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from types import ModuleType
import uuid

import numpy as np


_POOLS = ("source", "context_source", "late_source", "bomb_pool")
_FILES = ("bank_path", "profile_path", "heads_path", "reservations")
_REQUIRED = set(_POOLS + _FILES + ("audit",))
_HASH = re.compile(r"[0-9a-f]{64}")


def _path(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("nonempty_path_required")
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _profile(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or not _REQUIRED <= set(raw)
            or set(raw) - _REQUIRED - {"glyph_supplement"}):
        raise ValueError("explicit_aa8_factory_fields_required")
    if not isinstance(raw["audit"], str) or not _HASH.fullmatch(raw["audit"]):
        raise ValueError("training_audit_sha256_required")
    spec = {key: str(_path(raw[key], path.parent)) for key in _POOLS + _FILES}
    spec["audit"] = raw["audit"]
    refs = raw.get("glyph_supplement", [])
    if not isinstance(refs, list):
        raise ValueError("glyph_supplement_list_required")
    if refs and (len(refs) != 3 or any(not isinstance(r, dict) for r in refs)):
        raise ValueError("three_competing_glyph_references_required")
    supplements = []
    for row in refs:
        if (set(row) != {"label", "path", "slot", "sha256"}
                or row["label"] not in ("fold", "muck", "all_in")
                or type(row["slot"]) is not int or not 0 <= row["slot"] < 8
                or not isinstance(row["sha256"], str)
                or not _HASH.fullmatch(row["sha256"])):
            raise ValueError("invalid_glyph_reference")
        supplements.append({**row, "path": str(_path(row["path"], path.parent))})
    if supplements and len({r["label"] for r in supplements}) != 3:
        raise ValueError("three_competing_glyph_references_required")
    return path, spec, supplements


def preflight_profile(path):
    """Describe readiness without loading NPZs, images, or capture devices."""
    result = {"ready": False, "errors": [], "paths": {},
              "strategy_eligible": False, "profile_path": str(path)}
    try:
        resolved, spec, supplements = _profile(path)
        result["profile_path"] = str(resolved)
        result["paths"] = {k: spec[k] for k in _POOLS + _FILES}
        for key in _POOLS:
            if not Path(spec[key]).is_dir():
                result["errors"].append("missing_directory:" + key)
            elif not (Path(spec[key]) / "samples.json").is_file():
                result["errors"].append("missing_manifest:" + key)
        for key in _FILES:
            if not Path(spec[key]).is_file():
                result["errors"].append("missing_file:" + key)
        for row in supplements:
            if not Path(row["path"]).is_file():
                result["errors"].append("missing_glyph:" + row["label"])
        if Path(spec["profile_path"]).is_file():
            layout = json.loads(Path(spec["profile_path"]).read_text(encoding="utf-8"))
            if (not isinstance(layout, dict) or layout.get("canvas") != [498, 1080]
                    or layout.get("hero_slot") != 4
                    or not isinstance(layout.get("slots"), list)
                    or len(layout["slots"]) != 8
                    or any(not isinstance(r, dict) for r in layout["slots"])
                    or [r.get("slot") for r in layout["slots"]] != list(range(8))):
                result["errors"].append("aa8_hero4_layout_required")
        result["glyph_supplement_configured"] = bool(supplements)
        result["ready"] = not result["errors"]
    except (OSError, ValueError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


class _GlyphReader:
    def __init__(self, base, modes, supplement):
        self.base, self.modes, self.supplement = base, modes, supplement
        self.scene = base.scene
        self.suspended = True

    def recognize(self, image):
        special = self.modes.recognize(image)
        self.suspended = bool(
            self.scene.recognize(image).get("scene") != "AA_TABLE_CANDIDATE"
            or special.get("block_state_updates")
            or special.get("insurance") == "VISIBLE")
        if self.suspended:
            return {str(slot): None for slot in range(8)}
        base = self.base.recognize(image)
        return self.supplement.recognize(image, base, supported=True) if (
            self.supplement is not None) else base


class _Tracker:
    def __init__(self, reader):
        from tools.aa8_glyph_transitions_v2 import GlyphTransitionsV2
        self.reader = reader
        self.tracker = GlyphTransitionsV2()

    def observe(self, frame, glyphs):
        return self.tracker.observe(frame, glyphs, suspended=self.reader.suspended)


def _create_candidate(spec):
    # Explicit module, imported only after user starts a configured session.
    from tools.aa8_candidate_v2 import create_candidate
    return create_candidate(spec)


def _copy_candidate(state):
    """Copy temporal/model state while retaining its imported implementation.

    FrozenPredictionState stores the pipeline module in ``p``. Modules cannot
    be pickled, and must remain shared implementation references, not copies.
    A per-call memo avoids changing deepcopy behavior anywhere else.
    """
    memo = {id(value): value for value in vars(state).values()
            if isinstance(value, ModuleType)}
    return deepcopy(state, memo)


class AA8Reader:
    """Read sampled AA8 observations, never promoting model candidates.

    ``frame`` is the caller's observation sequence, not an invented raw-device
    frame ID. The caller must retain raw capture counters separately. Skips,
    source changes and time gaps reset all temporal state without reloading
    training images. Source elapsed seconds must increase; sampling does not
    establish complete action coverage.
    """

    def __init__(self, profile_path, *, factory=None):
        self.preflight = preflight_profile(profile_path)
        if not self.preflight["ready"]:
            raise ValueError("; ".join(self.preflight["errors"]))
        _, spec, references = _profile(profile_path)
        state = (factory or _create_candidate)(spec)
        supplement = None
        if references:
            from tools.aa8_glyph_supplement_v3 import GlyphSupplementV3
            from tools.wpk_video_dataset import read_image
            images = {}
            for ref in references:
                path = Path(ref["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("glyph_reference_hash_mismatch:" + ref["label"])
                images[ref["label"]] = (read_image(path), ref["slot"])
            supplement = GlyphSupplementV3(state.profile, images)
        state.reader = _GlyphReader(state.reader, state.modes, supplement)
        state.tracker = _Tracker(state.reader)
        self._initial = _copy_candidate(state)
        self._state = state
        self._source = "aa8-session-" + uuid.uuid4().hex
        self._last = None
        self._invalidated = False

    def read(self, image, frame, sample):
        try:
            if (not isinstance(image, np.ndarray)
                    or image.shape != (1080, 498, 3) or image.dtype != np.uint8):
                raise ValueError("aa8_uint8_bgr_498x1080_required")
            if type(frame) is not int or frame < 0 or not isinstance(sample, dict):
                raise ValueError("valid_observation_sequence_and_sample_required")
            pts = sample.get("pts_seconds")
            if (isinstance(pts, bool) or not isinstance(pts, (int, float))
                    or not math.isfinite(pts) or pts < 0):
                raise ValueError("finite_nonnegative_source_elapsed_seconds_required")
            source = sample.get("source_id", sample.get("session_id", self._source))
            if not isinstance(source, str) or not source:
                raise ValueError("nonempty_source_identity_required")
            previous = self._last
            if previous and source == previous[2] and (
                    frame <= previous[0] or pts <= previous[1]):
                raise ValueError("duplicate_or_backwards_observation")
            reset = bool(self._invalidated or previous and (
                source != previous[2] or frame != previous[0] + 1
                or pts - previous[1] > 1.0))
            if reset:
                self._state = _copy_candidate(self._initial)
            self._state.audit = source
            current_hash = hashlib.sha256(image.tobytes()).hexdigest()
            row = self._state.read(image, frame, {
                "pts_seconds": pts, "sha256": current_hash})
            row.update(
                source_id=source, training_audit_sha256=self._initial.audit,
                observation_sequence=frame, reader_gap_reset=reset,
                pixel_hash_encoding="raw BGR uint8 498x1080",
                candidate_only=True, strategy_eligible=False, advice_emitted=False,
                complete_legal_state=False,
                actions_complete_and_canonical_verified=False)
            row["source_frame"] = sample.get("source_frame")
            row["context_only"] = sample.get("context_only", False)
            row["source_provenance"] = {
                key: sample[key] for key in (
                    "source_kind", "source_pts_exact", "source_manifest_sha256",
                    "source_png_sha256", "source_playlist_sha256",
                    "source_pool", "source_file")
                if key in sample}
            self._last = (frame, pts, source)
            self._invalidated = False
            return row
        except Exception:
            # A rejected/read-failed frame must never bridge temporal evidence.
            self._invalidated = True
            raise
