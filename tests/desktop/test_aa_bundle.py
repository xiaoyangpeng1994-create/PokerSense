"""Bundle relocation and trust boundaries, without any private assets/devices."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from poker_engine.desktop.aa_bundle import (
    FILES, MANIFEST, POOLS, export_bundle, required_frames, validate_bundle,
)
from poker_engine.desktop.aa_reader import AA8Reader, preflight_profile


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "old-machine"
    root.mkdir()
    spec = {"audit": "a" * 64}
    for key, needed in required_frames().items():
        folder = root / key
        (folder / "frames").mkdir(parents=True)
        spec[key] = str(folder)
        frame_key = "source_frame" if key == "bomb_pool" else "global_frame"
        frames = range(min(needed), max(needed) + 1) if key in (
            "source", "context_source") else needed
        rows = []
        for frame in frames:
            data = (key + str(frame)).encode()
            row = {frame_key: frame, "file": "frames/" + str(frame) + ".png",
                   "sha256": hashlib.sha256(data).hexdigest(),
                   "pts_seconds": str(frame / 30),
                   "split" if key == "bomb_pool" else "role": "development"}
            rows.append(row)
            if frame in needed:
                (folder / row["file"]).write_bytes(data)
        identity = "source_sha256" if key == "bomb_pool" else "audit_sha256"
        write_json(folder / "samples.json", {
            identity: "b" * 64 if key == "bomb_pool" else "a" * 64,
            "private_original_path": str(root), "samples": rows})
    for key in FILES:
        suffix = ".npz" if key in ("bank_path", "heads_path") else ".json"
        path = root / (key + suffix)
        write_json(path, {})
        spec[key] = str(path)
        if key == "heads_path":
            write_json(path.with_suffix(".json"), {"format": "mlp-v1", "heads": {}})
    write_json(Path(spec["profile_path"]), {
        "canvas": [498, 1080], "hero_slot": 4,
        "slots": [{"slot": i} for i in range(8)]})
    write_json(Path(spec["reservations"]), {
        "source_sha256": "b" * 64, "reserved_inclusive_intervals": [[9000, 9100]],
        "known_exploration_frames": [],
        "known_development_inclusive_intervals": []})
    profile = root / "reader.json"
    write_json(profile, spec)
    return profile


def test_minimal_export_relocates_without_old_paths_or_decoding(
        source, tmp_path, monkeypatch):
    from tools import wpk_video_dataset
    import numpy as np
    monkeypatch.setattr(wpk_video_dataset, "read_image", lambda p: pytest.fail(
        "no decoding"))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no model loading"))
    result = export_bundle(source, tmp_path / "export")
    relocated = tmp_path / "new-machine" / "AA Resources"
    relocated.parent.mkdir()
    shutil.move(str(tmp_path / "export"), str(relocated))
    shutil.move(str(source.parent), str(tmp_path / "old-unavailable"))
    receipt = validate_bundle(relocated / "profile.json", result["manifest_sha256"])
    assert receipt["reference_frames"] == 21
    assert len(list(relocated.rglob("*.png"))) == 21
    assert receipt["files"] == 31
    assert receipt["development_only"] and not receipt["strategy_eligible"]
    assert preflight_profile(relocated / "profile.json", bundle_sha256=result[
        "manifest_sha256"])["ready"]
    profile = json.loads((relocated / "profile.json").read_text())
    assert all(not Path(profile[k]).is_absolute() for k in POOLS + FILES)
    all_json = "".join(p.read_text() for p in relocated.rglob("*.json"))
    assert str(source.parent) not in all_json
    assert len(json.loads((relocated / profile["source"] / "samples.json").read_text())[
        "samples"]) > 21  # Metadata stays contiguous; other PNGs are never copied.


@pytest.mark.parametrize("resource", ["profile", "manifest", "bank", "image"])
def test_tampering_rejected_before_candidate_construction(source, tmp_path, resource):
    target = tmp_path / "export"
    receipt = export_bundle(source, target)
    profile = target / "profile.json"
    spec = json.loads(profile.read_text())
    chosen = {"profile": profile, "manifest": target / MANIFEST,
              "bank": target / spec["bank_path"],
              "image": next(target.rglob("*.png"))}[resource]
    data = chosen.read_bytes()
    chosen.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError):
        AA8Reader(profile, bundle_sha256=receipt["manifest_sha256"],
                  factory=lambda spec: pytest.fail("must not construct candidate"))


def test_missing_pin_fails_closed(source, tmp_path):
    receipt = export_bundle(source, tmp_path / "export")
    result = preflight_profile(receipt["profile_path"])
    assert not result["ready"]
    assert "trusted_bundle_manifest_sha256_required" in result["errors"]


@pytest.mark.parametrize("key", POOLS)
def test_source_holdout_role_rejected_without_reading_png(
        source, tmp_path, monkeypatch, key):
    spec = json.loads(source.read_text())
    manifest_path = Path(spec[key]) / "samples.json"
    manifest = json.loads(manifest_path.read_text())
    frame_key = "source_frame" if key == "bomb_pool" else "global_frame"
    selected = next(r for r in manifest["samples"]
                    if r[frame_key] == required_frames()[key][0])
    selected["split" if key == "bomb_pool" else "role"] = "holdout"
    write_json(manifest_path, manifest)
    read_bytes = Path.read_bytes

    def forbidden(path):
        if path.suffix == ".png":
            pytest.fail("role must be checked before reference bytes")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError):
        export_bundle(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_reserved_bomb_rejected_before_reference_read(source, tmp_path, monkeypatch):
    spec = json.loads(source.read_text())
    path = Path(spec["reservations"])
    data = json.loads(path.read_text())
    data["reserved_inclusive_intervals"] = [[8640, 8640]]
    write_json(path, data)
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: pytest.fail("reserved read")
                        if p.suffix == ".png" else read_bytes(p))
    with pytest.raises(ValueError, match="reserved frame"):
        export_bundle(source, tmp_path / "export")


@pytest.mark.parametrize("unsafe", [
    "../escape.png", "/outside.png", "C:/outside.png", "a\\b.png"])
def test_source_path_escape_rejected(source, tmp_path, unsafe):
    spec = json.loads(source.read_text())
    path = Path(spec["late_source"]) / "samples.json"
    data = json.loads(path.read_text())
    data["samples"][0]["file"] = unsafe
    write_json(path, data)
    with pytest.raises(ValueError, match="relative_path"):
        export_bundle(source, tmp_path / "export")


def test_export_missing_witness_and_bad_hash_fail_without_output(source, tmp_path):
    image = next(source.parent.rglob("*.png"))
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source_reference_hash_mismatch"):
        export_bundle(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()
    image.unlink()
    with pytest.raises(FileNotFoundError):
        export_bundle(source, tmp_path / "export")


def test_external_pin_cannot_be_replaced_by_internal_manifest(source, tmp_path):
    target = tmp_path / "export"
    result = export_bundle(source, target)
    path = next(target.rglob("*.png"))
    path.write_bytes(b"replacement")
    manifest_path = target / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    row = manifest["files"][path.relative_to(target).as_posix()]
    row.update(sha256=digest(path), bytes=path.stat().st_size)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest_hash_mismatch"):
        validate_bundle(target / "profile.json", result["manifest_sha256"])


def test_export_never_overwrites_existing_output(source, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        export_bundle(source, target)
    assert marker.read_text() == "keep"


def repin(root, changed):
    path = root / MANIFEST
    manifest = json.loads(path.read_text())
    for name in changed:
        file = root / name
        manifest["files"][name] = {"sha256": digest(file), "bytes": file.stat().st_size}
    write_json(path, manifest)
    return digest(path)


def test_reserved_metadata_checked_before_any_bundle_png_bytes(
        source, tmp_path, monkeypatch):
    target = tmp_path / "export"
    export_bundle(source, target)
    raw = json.loads((target / "profile.json").read_text())
    path = target / raw["reservations"]
    data = json.loads(path.read_text())
    data["reserved_inclusive_intervals"] = [[8640, 8640]]
    write_json(path, data)
    external_hash = repin(target, [raw["reservations"]])
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: pytest.fail("reserved read")
                        if p.suffix == ".png" else read_bytes(p))
    with pytest.raises(ValueError, match="reserved frame"):
        validate_bundle(target / "profile.json", external_hash)


def test_pinned_manifest_must_bind_every_required_file(source, tmp_path):
    target = tmp_path / "export"
    export_bundle(source, target)
    path = target / MANIFEST
    manifest = json.loads(path.read_text())
    del manifest["files"][next(p for p in manifest["files"] if p.endswith(".png"))]
    write_json(path, manifest)
    with pytest.raises(ValueError, match="reference_not_bound"):
        validate_bundle(target / "profile.json", digest(path))


def test_dependency_version_drift_rejects_bundle(source, tmp_path):
    target = tmp_path / "export"
    export_bundle(source, target)
    path = target / MANIFEST
    manifest = json.loads(path.read_text())
    manifest["required_frames"]["source"].append(99999)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="dependency_drift"):
        validate_bundle(target / "profile.json", digest(path))


def test_link_cannot_escape_bundle(source, tmp_path):
    target = tmp_path / "export"
    result = export_bundle(source, target)
    path = next(target.rglob("*.png"))
    outside = tmp_path / "outside.png"
    path.rename(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit unprivileged symlink creation")
    with pytest.raises(ValueError, match="link_not_allowed"):
        validate_bundle(target / "profile.json", result["manifest_sha256"])


def test_card_head_sidecar_required_at_export_and_validation(source, tmp_path):
    target = tmp_path / "export"
    result = export_bundle(source, target)
    raw = json.loads((target / "profile.json").read_text())
    (target / raw["heads_path"]).with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="bundle_missing"):
        validate_bundle(target / "profile.json", result["manifest_sha256"])
    original = json.loads(source.read_text())
    Path(original["heads_path"]).with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError):
        export_bundle(source, tmp_path / "another")
    assert not (tmp_path / "another").exists()
