"""Record loaded OpenCV bytes separately from overlapping package metadata."""

import argparse
import base64
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import site
import sys

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import sha256_file


def isolation_checks(prefix: Path, base_prefix: Path, module_paths: dict,
                     search_paths: list[str], user_site_enabled: bool) -> dict:
    prefix = prefix.resolve()
    foreign = [p for p in search_paths if p and "site-packages" in Path(p).parts
               and not Path(p).resolve().is_relative_to(prefix)]
    local = {name: Path(path).resolve().is_relative_to(prefix)
             for name, path in module_paths.items()}
    venv = prefix != base_prefix.resolve()
    return {"virtual_environment": venv, "prefix": str(prefix),
            "user_site_enabled": user_site_enabled,
            "foreign_site_packages": foreign, "modules_inside_prefix": local,
            "passed": venv and not user_site_enabled and not foreign
            and bool(local) and all(local.values())}


def fingerprint() -> dict:
    directory = Path(cv2.__file__).parent
    binaries = list(directory.glob("*.pyd")) + list(directory.glob("*.so"))
    hashes = {str(path): sha256_file(path) for path in binaries}
    packages = []
    for distribution in distributions():
        name = distribution.metadata.get("Name", "")
        if not name.lower().startswith("opencv"):
            continue
        matches = []
        for record in distribution.files or ():
            resolved = str(Path(distribution.locate_file(record)).resolve())
            if resolved not in hashes or record.hash is None:
                continue
            expected = base64.urlsafe_b64encode(
                bytes.fromhex(hashes[resolved])).decode("ascii").rstrip("=")
            matches.append({"file": resolved, "record_algorithm": record.hash.mode,
                            "matches_record": record.hash.mode == "sha256"
                            and expected == record.hash.value})
        packages.append({"name": name, "version": distribution.version,
                         "binary_record_checks": matches})
    isolation = isolation_checks(
        Path(sys.prefix), Path(sys.base_prefix),
        {"cv2": cv2.__file__, "numpy": np.__file__}, sys.path,
        bool(site.ENABLE_USER_SITE))
    return {"python": platform.python_version(), "numpy_loaded": np.__version__,
            "isolation_checks": isolation,
            "opencv_loaded": cv2.__version__, "opencv_module_path": cv2.__file__,
            "opencv_binary_sha256": hashes, "opencv_distributions": packages,
            "overlapping_opencv_distributions": len(packages) > 1,
            "clean_environment_verified": False,
            "note": "Installed metadata is not a loaded-module lockfile. "
                    "Use an isolated environment for independent reproducibility."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous runtime fingerprint")
    report = fingerprint()
    report["tool_sha256"] = sha256_file(Path(__file__))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
