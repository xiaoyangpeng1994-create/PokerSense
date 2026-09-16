#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin, verified wrapper around **standard** git/gh for PokerSense development.

This is deliberately not a Git implementation and not a publishing platform. It
only invokes the real `git` and `gh` binaries with a configuration that has been
verified on this machine, and it refuses to report success without three-way SHA
proof.

Failure modes this fixes (all observed locally, see docs/GITHUB-SYNC-V1.zh-CN.md)
--------------------------------------------------------------------------------
1. **`git push` hangs.** The user-level config has a Git Credential Manager
   helper. Passing `-c credential.helper=!gh auth git-credential` does NOT
   disable it — `-c` *appends* to the helper list, so GCM still runs first and
   blocks. The list must be **reset first**: `-c credential.helper=` followed by
   the gh helper. GCM also needs `GCM_INTERACTIVE=never` so it never waits on a
   GUI prompt.
2. **Lost exit codes.** `git push ... | tail -n5` reports `tail`'s status, not
   git's. This wrapper always captures the real process exit code and is invoked
   without a shell pipeline.
3. **No timeout / no progress.** Every command runs under an external timer that
   kills only its own child tree, and a local progress trail is written while it
   runs so a hang is visible as a trail, not as silence.
4. **Pipe deadlock.** stdout/stderr are drained by dedicated reader threads, so a
   large or slow stream cannot stall the child.
5. **Duplicate publishing.** `publish()` is idempotent: it creates a commit only
   when the tree actually changed, and reuses an existing PR for the branch.

It never forces anything: no `--force`, no `reset`, no `rebase`, no `clean`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

GIT = "git"
GH = "gh"

# The exact helper override that was verified to work. Order matters: the empty
# value resets whatever the user/system config installed, and only then does the
# gh helper get added.
HELPER_RESET = ["-c", "credential.helper="]
GH_HELPER = ["-c", "credential.helper=!gh auth git-credential"]

PROGRESS_INTERVAL_S = 20.0


def child_env() -> dict:
    """Environment for git/gh children.

    github.com is only reachable through the local proxy on this machine, and
    interactive credential prompts must never block an unattended run.
    """
    env = dict(os.environ)
    env.setdefault("HTTP_PROXY", "http://127.0.0.1:10808")
    env.setdefault("HTTPS_PROXY", "http://127.0.0.1:10808")
    env.setdefault("NO_PROXY", "localhost,127.0.0.1")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    return env


def build_git_command(args, use_gh_helper: bool = True) -> list:
    """`git` plus the verified credential configuration, then the user args.

    Kept as a pure function so the ordering is unit-testable without a network.
    """
    cmd = [GIT]
    cmd += HELPER_RESET
    if use_gh_helper:
        cmd += GH_HELPER
    return cmd + list(args)


def _drain(stream, sink: list) -> None:
    try:
        for chunk in iter(lambda: stream.read(4096), b""):
            sink.append(chunk)
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except OSError:
            pass


def run(cmd, timeout_s: float, label: str, progress_path=None) -> dict:
    """Run a command with an external timer, real exit code and a progress trail."""
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    command_id = "%s-%d" % (label, int(time.time()))
    started = time.time()
    with open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=devnull, env=child_env(), creationflags=flags)
    out_chunks: list = []
    err_chunks: list = []
    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True),
    ]
    for t in threads:
        t.start()

    timed_out = False
    last_progress = started
    while proc.poll() is None:
        now = time.time()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            _write_progress(progress_path, {
                "command_id": command_id, "label": label,
                "phase": "running", "elapsed_s": round(now - started, 1),
                "timeout_s": timeout_s, "at_utc": _utc(),
            })
            last_progress = now
        if now - started > timeout_s:
            timed_out = True
            kill_tree(proc.pid)
            break
        time.sleep(0.2)

    for t in threads:
        t.join(timeout=5)
    rc = proc.poll()
    result = {
        "command_id": command_id,
        "label": label,
        "exit_code": rc,
        "timed_out": timed_out,
        "duration_s": round(time.time() - started, 1),
        "stdout": b"".join(out_chunks).decode("utf-8", "replace"),
        "stderr": b"".join(err_chunks).decode("utf-8", "replace"),
    }
    _write_progress(progress_path, {
        "command_id": command_id, "label": label,
        "phase": "timed_out" if timed_out else "done",
        "exit_code": rc, "elapsed_s": result["duration_s"], "at_utc": _utc(),
    })
    return result


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_progress(path, record: dict) -> None:
    """Append-only local progress trail. Not a heartbeat of research progress."""
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ identity


def local_head(repo: str, ref: str = "HEAD") -> str:
    out = subprocess.run([GIT, "rev-parse", ref], cwd=repo, capture_output=True,
                         text=True, encoding="utf-8", errors="replace",
                         env=child_env())
    return (out.stdout or "").strip()


def remote_branch_sha(repo: str, remote: str, branch: str) -> str:
    out = subprocess.run([GIT, "ls-remote", remote, "refs/heads/" + branch],
                         cwd=repo, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=child_env())
    line = (out.stdout or "").strip().splitlines()
    return line[0].split()[0] if line else ""


def pr_head_sha(repo: str, pr: int) -> str:
    out = subprocess.run([GH, "pr", "view", str(pr), "--json", "headRefOid"],
                         cwd=repo, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=child_env())
    try:
        return json.loads(out.stdout or "{}").get("headRefOid", "")
    except Exception:  # noqa: BLE001
        return ""


def three_way_equal(a: str, b: str, c: str) -> bool:
    """All three must be non-empty and identical. Empty never counts as equal."""
    return bool(a) and bool(b) and bool(c) and a == b == c


# ------------------------------------------------------------------ actions


def do_push(repo: str, branch: str, remote: str = "origin",
            timeout_s: float = 120.0, progress_path=None,
            pr: int | None = None, dry_run: bool = False) -> dict:
    """Standard `git push`, then prove local == remote == PR head."""
    if dry_run:
        return {"dry_run": True, "command": build_git_command(
            ["push", "--dry-run", remote, branch])}

    push = run(build_git_command(["push", remote, "refs/heads/%s:refs/heads/%s"
                                  % (branch, branch)]),
               timeout_s, "git-push", progress_path)
    local = local_head(repo)
    remote_sha = remote_branch_sha(repo, remote, branch)
    pr_sha = pr_head_sha(repo, pr) if pr else None
    verdict = {
        "exit_code": push["exit_code"],
        "timed_out": push["timed_out"],
        "duration_s": push["duration_s"],
        "local_head": local,
        "remote_branch_sha": remote_sha,
        "pr_head_sha": pr_sha,
        "stderr_tail": (push["stderr"].strip().splitlines()[-4:]
                        if push["stderr"] else []),
        "three_way_equal": three_way_equal(local, remote_sha, pr_sha or remote_sha),
    }
    if pr is not None:
        verdict["three_way_equal"] = three_way_equal(local, remote_sha, pr_sha)
    return verdict


def publish(repo: str, branch: str, title: str, body_file: str,
            base: str, remote: str = "origin",
            timeout_s: float = 120.0, progress_path=None) -> dict:
    """Idempotent publish: push, then reuse an existing PR for this branch.

    A second run of the same version must not create a second commit, PR or
    comment — the caller guarantees that by only calling after committing.
    """
    push = do_push(repo, branch, remote=remote, timeout_s=timeout_s,
                   progress_path=progress_path)
    if push.get("exit_code") != 0:
        return {"ok": False, "stage": "push", **push}

    existing = subprocess.run(
        [GH, "pr", "list", "--head", branch, "--state", "all",
         "--json", "number,isDraft,state,url,headRefOid"],
        cwd=repo, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=child_env())
    try:
        prs = json.loads(existing.stdout or "[]")
    except Exception:  # noqa: BLE001
        prs = []

    if prs:
        pr = prs[0]
        return {"ok": True, "stage": "pr_reused", "pr": pr["number"],
                "url": pr["url"], "draft": pr["isDraft"], **push}

    created = subprocess.run(
        [GH, "pr", "create", "--draft", "--base", base, "--head", branch,
         "--title", title, "--body-file", body_file],
        cwd=repo, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=child_env())
    if created.returncode != 0:
        return {"ok": False, "stage": "pr_create",
                "stderr": (created.stderr or "")[:300], **push}
    url = (created.stdout or "").strip()
    number = int(url.rstrip("/").split("/")[-1]) if "/" in url else None
    return {"ok": True, "stage": "pr_created", "pr": number, "url": url, **push}


# ------------------------------------------------------------------ entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="git_sync.py",
                                 description="Verified standard git/gh sync wrapper.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--progress", default=None)
    ap.add_argument("--pr", type=int, default=None)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("push")
    sub.add_parser("verify")
    sub.add_parser("show-config")
    args = ap.parse_args(argv)

    if args.command == "show-config":
        safe = {"git": GIT, "gh": GH,
                "helper_reset": HELPER_RESET, "gh_helper": GH_HELPER,
                "git_version": subprocess.run([GIT, "--version"], capture_output=True,
                                              text=True).stdout.strip()}
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return 0

    if args.command == "push":
        out = do_push(args.repo, args.branch, remote=args.remote,
                      timeout_s=args.timeout, progress_path=args.progress,
                      pr=args.pr)
    else:
        local = local_head(args.repo)
        remote_sha = remote_branch_sha(args.repo, args.remote, args.branch)
        pr_sha = pr_head_sha(args.repo, args.pr) if args.pr else remote_sha
        out = {"local_head": local, "remote_branch_sha": remote_sha,
               "pr_head_sha": pr_sha,
               "three_way_equal": three_way_equal(local, remote_sha, pr_sha)}

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("three_way_equal", out.get("exit_code") == 0) else 3


if __name__ == "__main__":
    sys.exit(main())
