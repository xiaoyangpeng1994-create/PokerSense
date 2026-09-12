"""Read-only filename guard for Git's index, optionally including local history.

Never reads file contents or prints secrets. A clean result is not a content
audit and cannot detect personal information in normal code/JSON/doc files.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess


PRIVATE_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".apk", ".apks",
                      ".xapk", ".pem", ".key", ".p12", ".pfx", ".sqlite",
                      ".sqlite3", ".db"}
PRIVATE_DIRECTORIES = {"private", "private-data", "captures", "screenshots"}
ENV_EXAMPLES = {".env.example", ".env.sample", ".env.template"}


def private_reason(filename):
    path = PurePosixPath(filename.replace("\\", "/").lower())
    if path.suffix in PRIVATE_EXTENSIONS:
        return "private_media_credential_or_database_extension"
    environment = path.name == ".env" or path.name.startswith(".env.")
    if environment and path.name not in ENV_EXAMPLES:
        return "environment_credentials_filename"
    if path.name.startswith("codex-clipboard-") and path.suffix == ".png":
        return "clipboard_screenshot"
    if any(part in PRIVATE_DIRECTORIES or part.startswith("aa-package-audit-")
           for part in path.parts[:-1]):
        return "private_data_directory"
    return None


def git_names(root, history=False):
    args = (["log", "--all", "--format=", "--name-only", "-z"] if history
            else ["ls-files", "-z"])
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            check=True)
    return sorted({name for name in result.stdout.decode("utf-8", errors="replace")
                   .split("\0") if name})


def check(root, history=False):
    names = git_names(root, history)
    flagged = [{"path": name, "reason": private_reason(name)} for name in names
               if private_reason(name)]
    return {"scope": "local_git_history" if history else "current_git_index",
            "paths_checked": len(names), "flagged": flagged,
            "content_scanned": False, "remote_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    report = check(Path(__file__).resolve().parents[1], args.history)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["flagged"] else 0)
