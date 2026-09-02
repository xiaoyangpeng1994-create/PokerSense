"""SHA-256 evidence helpers.

Guide rule 8 requires every data file, config, template and report that
enters the result set to carry a SHA-256. Section 18 defines the manifest
format (``shasum -a 256`` compatible: 64 hex chars, two spaces, path).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

_CHUNK = 1024 * 1024
DEFAULT_MANIFEST_NAME = "SHA256SUMS"


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Stream a file and return its hex SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Encode JSON deterministically so equal content hashes equally."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_json(obj: Any) -> str:
    """Hash a JSON-serializable object through its canonical encoding."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def iter_evidence_files(
    root: Path, *, skip_names: tuple[str, ...] = (DEFAULT_MANIFEST_NAME,)
) -> Iterator[Path]:
    """Yield every file under ``root`` except skipped manifest names.

    Sorted so the manifest is reproducible across machines and runs.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in skip_names:
            continue
        yield path


def write_sha256sums(
    root: Path, *, filename: str = DEFAULT_MANIFEST_NAME
) -> Path:
    """Write a ``SHA256SUMS`` manifest and return its path."""
    root = Path(root)
    lines = []
    for path in iter_evidence_files(root, skip_names=(filename,)):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    target = root / filename
    body = "\n".join(lines)
    target.write_text(body + ("\n" if body else ""), encoding="utf-8")
    return target


def verify_sha256sums(
    root: Path, *, filename: str = DEFAULT_MANIFEST_NAME
) -> list[str]:
    """Return human-readable problems; an empty list means every file matches.

    Missing files and hash mismatches are both reported rather than raising,
    so a reviewer sees the full damage in one pass.
    """
    root = Path(root)
    target = root / filename
    if not target.is_file():
        return [f"missing manifest: {filename}"]
    problems: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, relative = line.partition("  ")
        if not separator:
            problems.append(f"malformed manifest line: {line!r}")
            continue
        candidate = root / relative
        if not candidate.is_file():
            problems.append(f"missing file: {relative}")
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            problems.append(f"hash mismatch: {relative}")
    return problems
