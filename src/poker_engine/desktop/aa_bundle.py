"""Portable, explicitly hash-pinned development references for the AA8 reader.

No image/model decoding, capture, training or holdout access happens here.
The trusted manifest digest must travel separately from the bundle itself.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


SCHEMA = "aa8-runtime-bundle-v1"
MANIFEST = "bundle-manifest.json"
PROFILE = "profile.json"
POOLS = ("source", "context_source", "late_source", "bomb_pool")
FILES = ("bank_path", "profile_path", "heads_path", "reservations")
HASH = re.compile(r"[0-9a-f]{64}")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key:" + key)
        result[key] = value
    return result


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def _relative(root, value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute()
            or any(p in (".", "..", "") for p in value.split("/"))):
        raise ValueError("bundle_relative_path_required")
    path = root.joinpath(*PurePosixPath(value).parts)
    # Reject links even if their current target is inside the bundle: portable
    # deployment must not depend on machine-specific indirection.
    current = path
    while current != root:
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError("bundle_link_not_allowed")
        current = current.parent
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("bundle_path_escape")
    return path


def required_frames():
    """Exact reference closure of FrozenPredictionState + CandidateStateV2.

    Glyph and special-mode maps are sourced from the actual reader modules;
    fixed geometry/timer/participation witnesses follow their constructors.
    This bundle is a model-reference package, not a replay dataset.
    """
    from tools.aa8_action_reader import TEMPLATES
    from tools.aa8_special_modes import SOURCES
    return {
        "source": sorted({f for f, _ in TEMPLATES.values()} | {
            1263, 1320, 1500, 1650, 1950, 2400, 2550, 3319}),
        "context_source": sorted(set(SOURCES.values()) | {4755}),
        "late_source": [25204, 28654, 29104],
        "bomb_pool": [8640],
    }


def _selected(spec):
    """Read metadata only, reject roles/reservations before touching PNGs."""
    from tools.aa8_action_transfer import inventory
    from tools.aa_data_separation import assert_training_frames
    result = {}
    required = required_frames()
    for key in POOLS:
        pool = Path(spec[key])
        manifest = _json(pool / "samples.json")
        if key in ("source", "context_source"):
            rows = inventory(pool, spec["audit"])
        elif key == "late_source":
            if (manifest.get("audit_sha256") != spec["audit"]
                    or any(r.get("role") != "development"
                           for r in manifest["samples"])):
                raise ValueError("late_training_roles_audit_mismatch")
            samples = manifest["samples"]
            rows = {r["global_frame"]: r for r in samples}
            if len(rows) != len(samples):
                raise ValueError("duplicate_late_frame")
        else:
            reservations = _json(Path(spec["reservations"]))
            if manifest.get("source_sha256") != reservations.get("source_sha256"):
                raise ValueError("bomb_source_mismatch")
            assert_training_frames(required[key], reservations)
            samples = manifest["samples"]
            rows = {r["source_frame"]: r for r in samples}
            if len(rows) != len(samples):
                raise ValueError("duplicate_bomb_frame")
        selected = []
        for frame in required[key]:
            if frame not in rows:
                raise ValueError("missing_required_frame:" + key + ":" + str(frame))
            row = rows[frame]
            if row.get("split" if key == "bomb_pool" else "role") != "development":
                raise ValueError("development_reference_required")
            if (not isinstance(row.get("sha256"), str)
                    or not HASH.fullmatch(row["sha256"])):
                raise ValueError("reference_sha256_required")
            path = _relative(pool, row["file"])
            if path.suffix.lower() != ".png":
                raise ValueError("png_development_reference_required")
            selected.append((frame, row, path))
        result[key] = (manifest, selected)
    return result


def validate_bundle(profile_path, expected_manifest_sha256):
    """Verify ALL files before caller can construct/decode the candidate.

    Digest is supplied by a trusted caller/receipt, never read from this bundle.
    Returns a compact receipt; throws on tampering, missing dependencies or drift.
    """
    if (not isinstance(expected_manifest_sha256, str)
            or not HASH.fullmatch(expected_manifest_sha256)):
        raise ValueError("trusted_bundle_manifest_sha256_required")
    profile_path = Path(profile_path).absolute()
    root = profile_path.parent
    if profile_path.name != PROFILE:
        raise ValueError("bundle_profile_filename_required")
    manifest_path = _relative(root, MANIFEST)
    data = manifest_path.read_bytes()
    if _digest(data) != expected_manifest_sha256:
        raise ValueError("bundle_manifest_hash_mismatch")
    manifest = json.loads(data, object_pairs_hook=_unique)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("development_only") is not True
            or manifest.get("profile") != PROFILE
            or manifest.get("required_frames") != required_frames()
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("invalid_bundle_manifest_or_dependency_drift")
    files = manifest["files"]
    for name, record in files.items():
        if (not isinstance(record, dict) or set(record) != {"sha256", "bytes"}
                or not isinstance(record["sha256"], str)
                or not HASH.fullmatch(record["sha256"])
                or type(record["bytes"]) is not int or record["bytes"] < 0):
            raise ValueError("invalid_bundle_file_record")
        path = _relative(root, name)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("bundle_missing_or_size_mismatch:" + name)
    if PROFILE not in files:
        raise ValueError("bundle_profile_not_bound")

    def verify(name):
        if _digest(_relative(root, name).read_bytes()) != files[name]["sha256"]:
            raise ValueError("bundle_file_hash_mismatch:" + name)

    verify(PROFILE)
    raw = _json(profile_path)
    profile_keys = set(POOLS + FILES + ("audit", "bundle_manifest", "glyph_supplement"))
    if (not isinstance(raw, dict) or raw.get("bundle_manifest") != MANIFEST
            or set(raw) != profile_keys):
        raise ValueError("invalid_bundle_profile")
    spec = {k: str(_relative(root, raw[k])) for k in POOLS + FILES}
    spec["audit"] = raw["audit"]
    expected = {PROFILE}
    expected.update(raw[k] for k in FILES)
    expected.add(PurePosixPath(raw["heads_path"]).with_suffix(".json").as_posix())
    expected.update(raw[k] + "/samples.json" for k in POOLS)
    if not expected <= set(files):
        raise ValueError("bundle_configuration_not_bound")
    # Roles/reservations must be checked before opening any reference PNG,
    # including when a caller pins a syntactically valid but unsuitable bundle.
    configuration_names = expected.copy()
    for name in configuration_names:
        verify(name)
    selected = _selected(spec)
    for key, (_, rows) in selected.items():
        for _, row, path in rows:
            name = path.relative_to(root).as_posix()
            if name not in files or files[name]["sha256"] != row["sha256"]:
                raise ValueError("bundle_reference_not_bound")
            expected.add(name)
    supplements = raw["glyph_supplement"]
    if not isinstance(supplements, list):
        raise ValueError("glyph_supplement_list_required")
    for row in supplements:
        path = _relative(root, row["path"])
        name = path.relative_to(root).as_posix()
        if name not in files or files[name]["sha256"] != row["sha256"]:
            raise ValueError("bundle_glyph_not_bound")
        expected.add(name)
    if expected != set(files):
        raise ValueError("bundle_file_closure_mismatch")
    for name in files:
        if name not in configuration_names:
            verify(name)
    return {"schema": SCHEMA, "manifest_sha256": expected_manifest_sha256,
            "files": len(files), "bytes": sum(r["bytes"] for r in files.values()),
            "reference_frames": sum(map(len, required_frames().values())),
            "development_only": True, "strategy_eligible": False}


def export_bundle(profile_path, output):
    """Copy only current reachable development witnesses; never decode them.

    Source/context manifests retain their full contiguous metadata inventory to
    preserve the unchanged constructor contract. Only selected PNGs are copied.
    Destination must not exist; interrupted output is never reported as valid.
    """
    from .aa_reader import _profile, preflight_profile
    ready = preflight_profile(profile_path)
    if not ready["ready"]:
        raise ValueError("; ".join(ready["errors"]))
    _, spec, supplements = _profile(profile_path)
    selected = _selected(spec)
    pending = {}
    new_profile = {"audit": spec["audit"], "bundle_manifest": MANIFEST,
                   "glyph_supplement": []}

    def add(name, path, expected_hash=None):
        if name in pending:
            raise ValueError("duplicate_bundle_destination")
        data = path.read_bytes()
        if expected_hash and _digest(data) != expected_hash:
            raise ValueError("source_reference_hash_mismatch:" + name)
        pending[name] = data

    for key in FILES:
        source = Path(spec[key])
        name = "assets/" + key + source.suffix
        add(name, source)
        new_profile[key] = name
        if key == "heads_path":
            add(PurePosixPath(name).with_suffix(".json").as_posix(),
                source.with_suffix(".json"))
    for key, (manifest, rows) in selected.items():
        base = "references/" + key
        new_profile[key] = base
        # Keep only provenance and reader-required metadata, never absolute
        # paths / unrelated log fields from an original collection manifest.
        identity = "source_sha256" if key == "bomb_pool" else "audit_sha256"
        frame_key = "source_frame" if key == "bomb_pool" else "global_frame"
        fields = (frame_key, "file", "sha256", "pts_seconds",
                  "split" if key == "bomb_pool" else "role")
        kept = manifest["samples"] if key in ("source", "context_source") else [
            row for _, row, _ in rows]
        reduced = {identity: manifest[identity], "samples": [
            {k: row[k] for k in fields if k in row} for row in kept]}
        pending[base + "/samples.json"] = _encoded(reduced)
        for _, row, path in rows:
            add(base + "/" + row["file"], path, row["sha256"])
    for row in supplements:
        name = "glyphs/" + row["label"] + ".png"
        add(name, Path(row["path"]), row["sha256"])
        new_profile["glyph_supplement"].append({**row, "path": name})
    pending[PROFILE] = _encoded(new_profile)
    manifest = {"schema": SCHEMA, "profile": PROFILE, "development_only": True,
                "required_frames": required_frames(), "files": {
                    name: {"sha256": _digest(data), "bytes": len(data)}
                    for name, data in sorted(pending.items())}}
    encoded = _encoded(manifest)
    root = Path(output).absolute()
    root.mkdir(parents=True, exist_ok=False)
    for name, data in pending.items():
        path = _relative(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / MANIFEST).write_bytes(encoded)
    receipt = validate_bundle(root / PROFILE, _digest(encoded))
    return {**receipt, "profile_path": str(root / PROFILE)}


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
