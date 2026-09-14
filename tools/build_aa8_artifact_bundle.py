"""Copy the current AA8 JSON evidence chain into a closed non-media bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from tools.verify_decision_opportunity_artifacts import (
    LIMITATIONS, ROLE_SPECS,
)


DATASET_ID = "aa8-current-action-candidates-NOT-DATA-v1"


def _source_bytes(path, maximum=64 * 1024 * 1024):
    before = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or bool(getattr(before, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))):
        raise ValueError("bundle_source_must_be_plain_single_link_file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ValueError("bundle_source_exceeds_size_limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()

    def identity(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_mode, value.st_nlink,
                getattr(value, "st_file_attributes", 0))

    if (identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or identity(after) != identity(final)):
        raise ValueError("bundle_source_changed_during_read")
    return b"".join(chunks)


def build_bundle(inputs, output):
    if set(inputs) != set(ROLE_SPECS):
        raise ValueError("exact_artifact_input_roles_required")
    snapshots = {}
    for role, source in inputs.items():
        if not isinstance(source, Path) or not source.is_file():
            raise ValueError("artifact_input_file_missing:" + role)
        snapshots[role] = _source_bytes(source)
    output.mkdir(exist_ok=False)
    files, bindings = [], []
    for index, role in enumerate(ROLE_SPECS, start=1):
        raw = snapshots[role]
        file_format, schema_id = ROLE_SPECS[role]
        name = role + (".jsonl" if file_format == "jsonl" else ".json")
        target = output / name
        target.write_bytes(raw)
        file_id = f"artifact-{index:02d}"
        files.append({
            "file_id": file_id, "relative_path": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw), "format": file_format,
            "schema_id": schema_id})
        bindings.append({
            "owner_kind": "dataset", "owner_id": DATASET_ID,
            "role": role, "file_id": file_id})
    manifest = {
        "schema_version": 1,
        "status": "CURRENT_AA8_BLOCKED_ARTIFACT_BUNDLE_V1",
        "dataset_id": DATASET_ID, "files": files, "bindings": bindings,
        "limitations": LIMITATIONS}
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes((json.dumps(
        manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return {"bundle_file_count": len(files),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLE_SPECS:
        parser.add_argument("--" + role.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_bundle(
            {role: getattr(args, role) for role in ROLE_SPECS}, args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"artifact bundle build rejected: {exc}\n")
    print(json.dumps({key: result[key] for key in (
        "bundle_file_count", "manifest_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
