#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin, verified wrapper around **standard** git/gh for PokerSense development.

Not a Git implementation and not a publishing platform: it only invokes the real
`git` and `gh` binaries with a configuration verified on this machine, and it
refuses to report success without the applicable SHA proof.

Fixes carried from round 1
--------------------------
* **Credential helper ordering.** `-c credential.helper=X` only *appends* to the
  helper list, so a pre-existing Git Credential Manager helper still runs first
  and blocks an unattended push. The list is reset (`-c credential.helper=`)
  before the gh helper is added, with `GIT_TERMINAL_PROMPT=0` and
  `GCM_INTERACTIVE=never`.

Fixes added after review GITHUB-001-R1
--------------------------------------
* **P1-A owned process groups.** On POSIX every command starts in its own session
  (`start_new_session=True`) and only *that* group is signalled; on Windows the
  tree is killed with a bounded wait and a survived-process check. Kill failures
  are reported instead of assumed. Deadlines use a monotonic clock, and
  non-finite / non-positive timeouts are rejected.
* **P1-B repo binding.** Every git call carries `-C <resolved repo>` and every gh
  call runs with `cwd=<repo>`. The remote identity, the target ref and the commit
  to be published are validated *before* any remote write.
* **P1-C one execution path.** `ls-remote`, `pr view`, `pr list`, `pr create` all
  go through `run()`, so they inherit the timeout, the credential fix, the real
  exit code and the progress trail. Auth/API failures are reported, never treated
  as empty output.
* **P1-D honest verdicts.** `GIT_SYNC` (local vs remote) and `PR_SYNC` (local vs
  remote vs PR head) are graded separately. Without a PR number the verdict is
  `PR_NOT_CHECKED`, never a two-way result dressed up as three-way. A PR is only
  reused when it is OPEN, Draft, on the expected base, and its actual head
  matches. An API failure is an error, not "no PR exists".

It never forces anything: no `--force`, no `reset`, no `rebase`, no `clean`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

GIT = "git"
GH = "gh"

HELPER_RESET = ["-c", "credential.helper="]
GH_HELPER = ["-c", "credential.helper=!gh auth git-credential"]

PROGRESS_INTERVAL_S = 20.0
TERMINATE_GRACE_S = 15.0

# Pushed branches in this repository must be task branches, never main.
ALLOWED_REF_PREFIX = "refs/heads/codex/"

EXPECTED_REPO_SLUG = "xiaoyangpeng1994-create/PokerSense"


class SyncError(RuntimeError):
    """Configuration or precondition failure. Never silently swallowed."""


# ------------------------------------------------------------------ environment


def resolve_proxy(cli_proxy: str | None = None) -> str:
    """Explicit local proxy only.

    Never injects a hardcoded loopback proxy: other machines and CI must not be
    forced through a proxy that does not exist there. Precedence:
    --proxy > $POKERSENSE_GIT_PROXY > whatever HTTP(S)_PROXY already is in the
    inherited environment (possibly empty).
    """
    if cli_proxy:
        return cli_proxy
    return (os.environ.get("POKERSENSE_GIT_PROXY") or "").strip()


def child_env(proxy: str | None = None) -> dict:
    env = dict(os.environ)
    proxy = resolve_proxy(proxy)
    if proxy:
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env.setdefault("NO_PROXY", "localhost,127.0.0.1")
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    return env


def build_git_command(args, repo: str | None = None, use_gh_helper: bool = True):
    """`git` + verified credential config + explicit repo binding + user args."""
    cmd = [GIT]
    cmd += HELPER_RESET
    if use_gh_helper:
        cmd += GH_HELPER
    if repo:
        cmd += ["-C", repo]
    return cmd + list(args)


# ------------------------------------------------------------------ execution


def _validate_timeout(timeout_s) -> float:
    try:
        value = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise SyncError("timeout must be a number, got %r" % (timeout_s,)) from exc
    if not math.isfinite(value) or value <= 0:
        raise SyncError("timeout must be finite and > 0, got %r" % (timeout_s,))
    return value


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


def terminate_owned_group(proc, pgid: int | None) -> dict:
    """Kill ONLY the process group/session this command owns.

    `pgid` is recorded at spawn time; on POSIX `start_new_session=True` makes the
    child a session leader, so its group id equals its pid and the caller's group
    can never be signalled.
    """
    result = {"kill_attempted": True, "kill_ok": False, "survivors": []}
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       timeout=TERMINATE_GRACE_S)
    else:
        import signal
        target = pgid if pgid and pgid > 0 else proc.pid
        try:
            os.killpg(target, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + TERMINATE_GRACE_S
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(target, signal.SIGKILL)
            except OSError:
                pass

    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    result["kill_ok"] = proc.poll() is not None
    if not result["kill_ok"]:
        result["survivors"].append(proc.pid)
    return result


def run(cmd, timeout_s, label: str, progress_path=None, cwd: str | None = None,
        proxy: str | None = None) -> dict:
    """Run a command with an owned process group, real exit code and a trail."""
    timeout_s = _validate_timeout(timeout_s)

    creationflags = 0
    popen_kwargs = {}
    if os.name == "nt":
        creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP
                         | subprocess.CREATE_NO_WINDOW)
    else:
        # Own session => own process group => killpg can never hit the caller.
        popen_kwargs["start_new_session"] = True

    command_id = "%s-%d" % (label, int(time.time() * 1000))
    started = time.monotonic()
    with open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=devnull, cwd=cwd, env=child_env(proxy),
                                creationflags=creationflags, **popen_kwargs)

    pgid = None
    if os.name != "nt":
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid

    out_chunks: list = []
    err_chunks: list = []
    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, out_chunks), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_chunks), daemon=True),
    ]
    for t in threads:
        t.start()

    timed_out = False
    kill_info = {"kill_attempted": False, "kill_ok": True, "survivors": []}
    deadline = started + timeout_s
    last_progress = started
    while proc.poll() is None:
        now = time.monotonic()
        if now - last_progress >= PROGRESS_INTERVAL_S:
            _write_progress(progress_path, {
                "command_id": command_id, "label": label, "phase": "running",
                "elapsed_s": round(now - started, 1), "timeout_s": timeout_s,
                "at_utc": _utc(),
            })
            last_progress = now
        if now >= deadline:
            timed_out = True
            kill_info = terminate_owned_group(proc, pgid)
            break
        time.sleep(0.2)

    for t in threads:
        t.join(timeout=5)
    rc = proc.poll()
    result = {
        "command_id": command_id,
        "label": label,
        "argv": list(cmd),
        "cwd": cwd,
        "exit_code": rc,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 1),
        "stdout": b"".join(out_chunks).decode("utf-8", "replace"),
        "stderr": b"".join(err_chunks).decode("utf-8", "replace"),
    }
    result.update(kill_info)
    _write_progress(progress_path, {
        "command_id": command_id, "label": label,
        "phase": "timed_out" if timed_out else "done",
        "exit_code": rc, "elapsed_s": result["duration_s"],
        "kill_ok": kill_info.get("kill_ok"), "at_utc": _utc(),
    })
    return result


def _utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_progress(path, record: dict) -> None:
    """Append-only LOCAL progress trail. Command progress, not research progress."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ repo safety


def resolve_repo(repo: str, timeout_s: float = 60.0, progress_path=None) -> str:
    """Resolve and validate the repository before any remote write."""
    res = run(build_git_command(["rev-parse", "--show-toplevel"], repo=repo,
                                use_gh_helper=False),
              timeout_s, "git-resolve-repo", progress_path)
    if res["exit_code"] != 0 or not (res["stdout"] or "").strip():
        raise SyncError("not a git repository: %r (%s)"
                        % (repo, (res["stderr"] or "").strip()[:200]))
    return res["stdout"].strip()


def remote_slug(repo: str, remote: str = "origin",
                timeout_s: float = 60.0, progress_path=None) -> str:
    res = run(build_git_command(["remote", "get-url", remote], repo=repo,
                                use_gh_helper=False),
              timeout_s, "git-remote-url", progress_path)
    url = (res["stdout"] or "").strip()
    slug = url[:-4] if url.endswith(".git") else url
    for marker in ("github.com/", "github.com:"):
        if marker in slug:
            slug = slug.split(marker, 1)[1]
    return slug.strip("/")


def local_head(repo: str, timeout_s: float = 60.0, progress_path=None) -> str:
    res = run(build_git_command(["rev-parse", "HEAD"], repo=repo,
                                use_gh_helper=False),
              timeout_s, "git-head", progress_path)
    return (res["stdout"] or "").strip()


def validate_push_target(repo: str, remote: str, branch: str, expected_sha: str,
                         must_be_task_branch: bool = True,
                         timeout_s: float = 60.0, progress_path=None) -> dict:
    """Everything that must hold BEFORE a remote write."""
    problems = []
    ref = "refs/heads/" + branch
    if must_be_task_branch and not ref.startswith(ALLOWED_REF_PREFIX):
        problems.append("ref %r is outside the allowed prefix %r"
                        % (branch, ALLOWED_REF_PREFIX))
    slug = remote_slug(repo, remote, timeout_s, progress_path)
    if slug and slug != EXPECTED_REPO_SLUG:
        problems.append("remote %r points at %r, expected %r"
                        % (remote, slug, EXPECTED_REPO_SLUG))
    head = local_head(repo, timeout_s, progress_path)
    if expected_sha and head != expected_sha:
        problems.append("HEAD %s does not match the commit to publish %s"
                        % (head, expected_sha))
    return {"ok": not problems, "problems": problems, "remote_slug": slug,
            "head": head, "branch": branch}


# ------------------------------------------------------------------ identity


def remote_branch_sha(repo: str, remote: str, branch: str,
                      timeout_s: float = 60.0, progress_path=None) -> dict:
    """Read the remote ref through the SAME wrapper (P1-C).

    Uses the verified credential configuration, so the old helper list can never
    come back on the verification path, and it is bounded by the same timeout.
    """
    res = run(build_git_command(["ls-remote", remote, "refs/heads/" + branch],
                                repo=repo),
              timeout_s, "git-ls-remote", progress_path)
    sha = ""
    if res["exit_code"] == 0:
        lines = (res["stdout"] or "").strip().splitlines()
        if lines:
            sha = lines[0].split()[0]
    return {"sha": sha, "exit_code": res["exit_code"], "timed_out": res["timed_out"],
            "error": "" if res["exit_code"] == 0 else (res["stderr"] or "")[:200]}


def pr_view(repo: str, pr: int, timeout_s: float = 60.0, progress_path=None) -> dict:
    """PR metadata via gh through the wrapper. Errors are reported, not masked."""
    res = run([GH, "pr", "view", str(pr), "--json",
               "number,state,isDraft,baseRefName,headRefName,headRefOid,url"],
              timeout_s, "gh-pr-view", progress_path, cwd=repo)
    if res["exit_code"] != 0:
        return {"ok": False, "error": (res["stderr"] or "").strip()[:200],
                "timed_out": res["timed_out"], "exit_code": res["exit_code"]}
    try:
        data = json.loads(res["stdout"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "unparsable gh output: %s" % exc}
    data["ok"] = True
    return data


def pr_list_for_branch(repo: str, branch: str, timeout_s: float = 60.0,
                       progress_path=None) -> dict:
    """List PRs for the head branch. An API failure is an ERROR, not an empty list."""
    res = run([GH, "pr", "list", "--head", branch, "--state", "all", "--json",
               "number,isDraft,state,baseRefName,url,headRefOid"],
              timeout_s, "gh-pr-list", progress_path, cwd=repo)
    if res["exit_code"] != 0:
        return {"ok": False, "error": (res["stderr"] or "").strip()[:200],
                "timed_out": res["timed_out"], "exit_code": res["exit_code"],
                "prs": []}
    try:
        prs = json.loads(res["stdout"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "unparsable gh output: %s" % exc, "prs": []}
    return {"ok": True, "prs": prs if isinstance(prs, list) else []}


# ------------------------------------------------------------------ verdicts


def git_sync_verdict(local: str, remote_sha: str) -> bool:
    """Two-way: local HEAD vs remote branch ref. Empty never counts as equal."""
    return bool(local) and bool(remote_sha) and local == remote_sha


def three_way_equal(a: str, b: str, c: str) -> bool:
    return bool(a) and bool(b) and bool(c) and a == b == c


def do_push(repo: str, branch: str, remote: str = "origin",
            timeout_s: float = 120.0, progress_path=None, pr: int | None = None,
            dry_run: bool = False, proxy: str | None = None,
            expected_sha: str | None = None, must_be_task_branch: bool = True) -> dict:
    """Standard `git push`, then grade GIT_SYNC and, only with a PR, PR_SYNC."""
    if dry_run:
        return {"dry_run": True,
                "command": build_git_command(
                    ["push", "--dry-run", remote,
                     "refs/heads/%s:refs/heads/%s" % (branch, branch)], repo=repo)}

    resolved = resolve_repo(repo, timeout_s, progress_path)
    head_before = local_head(resolved, timeout_s, progress_path)
    guard = validate_push_target(resolved, remote, branch,
                                 expected_sha or head_before,
                                 must_be_task_branch, timeout_s, progress_path)
    if not guard["ok"]:
        return {"exit_code": None, "timed_out": False, "git_sync_ok": False,
                "pr_sync": "PR_NOT_CHECKED", "three_way_equal": False,
                "blocked_before_write": True, "problems": guard["problems"],
                "local_head": guard["head"], "remote_branch_sha": "",
                "pr_head_sha": None}

    push = run(build_git_command(
        ["push", remote, "refs/heads/%s:refs/heads/%s" % (branch, branch)],
        repo=resolved),
        timeout_s, "git-push", progress_path, proxy=proxy)

    local = local_head(resolved, timeout_s, progress_path)
    remote_info = remote_branch_sha(resolved, remote, branch, timeout_s, progress_path)
    remote_sha = remote_info["sha"]
    git_ok = git_sync_verdict(local, remote_sha)

    verdict = {
        "resolved_repo": resolved,
        "exit_code": push["exit_code"],
        "timed_out": push["timed_out"],
        "kill_ok": push.get("kill_ok"),
        "duration_s": push["duration_s"],
        "local_head": local,
        "remote_branch_sha": remote_sha,
        "remote_read_error": remote_info["error"],
        "git_sync_ok": git_ok,
        "stderr_tail": (push["stderr"].strip().splitlines()[-4:]
                        if push["stderr"] else []),
    }

    # A command error with the postcondition already satisfied is NOT plain
    # success, but it is useful to distinguish it from a real failure.
    verdict["postcondition_only"] = bool(git_ok and push["exit_code"] != 0)

    if pr is None:
        # Two-way agreement must NOT be reported as three-way (P1-D).
        verdict["pr_head_sha"] = None
        verdict["pr_sync"] = "PR_NOT_CHECKED"
        verdict["three_way_equal"] = None
        return verdict

    meta = pr_view(resolved, pr, timeout_s, progress_path)
    if not meta.get("ok"):
        verdict["pr_head_sha"] = None
        verdict["pr_sync"] = "PR_READ_ERROR"
        verdict["pr_error"] = meta.get("error")
        verdict["three_way_equal"] = False
        return verdict
    verdict["pr_head_sha"] = meta.get("headRefOid")
    verdict["pr_state"] = meta.get("state")
    verdict["pr_base"] = meta.get("baseRefName")
    verdict["three_way_equal"] = three_way_equal(local, remote_sha,
                                                 meta.get("headRefOid") or "")
    verdict["pr_sync"] = ("PR_SYNC_PASS" if verdict["three_way_equal"]
                          else "PR_SYNC_FAIL")
    return verdict


def publish(repo: str, branch: str, title: str, body_file: str, base: str,
            remote: str = "origin", timeout_s: float = 120.0, progress_path=None,
            proxy: str | None = None) -> dict:
    """Push, then create-or-reuse a PR for this branch and VERIFY its head.

    Idempotent and honest: it does NOT create commits (the caller commits), it
    refuses to reuse a PR that is closed, non-draft, on a different base or
    pointing at a different head, and it will not create a PR when the listing
    errored.
    """
    push = do_push(repo, branch, remote=remote, timeout_s=timeout_s,
                   progress_path=progress_path, proxy=proxy)
    if push.get("blocked_before_write"):
        return {**push, "ok": False, "stage": "precondition"}
    if push.get("exit_code") != 0:
        status = ("POSTCONDITION_CONFIRMED_WITH_COMMAND_ERROR"
                  if push.get("git_sync_ok") else "PUSH_FAILED")
        return {**push, "ok": False, "stage": "push", "status": status}
    if not push.get("git_sync_ok"):
        return {**push, "ok": False, "stage": "git_sync",
                "status": "GIT_SYNC_FAIL"}

    listing = pr_list_for_branch(push["resolved_repo"], branch, timeout_s,
                                 progress_path)
    if not listing["ok"]:
        # An API failure is an error, NOT "no PR exists" (P1-D).
        return {**push, "ok": False, "stage": "pr_list", "status": "PR_LIST_API_ERROR",
                "error": listing.get("error")}

    reusable = None
    for cand in listing["prs"]:
        if (cand.get("state") == "OPEN" and cand.get("baseRefName") == base
                and cand.get("isDraft")):
            reusable = cand
            break
    if listing["prs"] and reusable is None:
        return {"ok": False, "stage": "pr_reuse", "status": "EXISTING_PR_INCOMPATIBLE",
                "found": [{"number": c.get("number"), "state": c.get("state"),
                           "draft": c.get("isDraft"), "base": c.get("baseRefName")}
                          for c in listing["prs"]], **push}

    if reusable:
        number, url, stage = reusable["number"], reusable["url"], "pr_reused"
    else:
        created = run([GH, "pr", "create", "--draft", "--base", base,
                       "--head", branch, "--title", title, "--body-file", body_file],
                      timeout_s, "gh-pr-create", progress_path,
                      cwd=push["resolved_repo"])
        if created["exit_code"] != 0:
            return {**push, "ok": False, "stage": "pr_create",
                    "status": "PR_CREATE_FAILED",
                    "error": (created["stderr"] or "").strip()[:200]}
        out = (created["stdout"] or "").strip()
        url = out.splitlines()[-1] if out else ""
        try:
            number = int(url.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            return {**push, "ok": False, "stage": "pr_create",
                    "status": "PR_CREATE_UNPARSABLE", "raw": url[:200]}
        stage = "pr_created"

    # Never trust create/reuse: re-read and verify the ACTUAL head (P1-D).
    meta = pr_view(push["resolved_repo"], number, timeout_s, progress_path)
    if not meta.get("ok"):
        return {**push, "ok": False, "stage": "pr_verify",
                "status": "PR_VERIFY_API_ERROR",
                "pr": number, "url": url, "error": meta.get("error")}
    problems = []
    if meta.get("headRefOid") != push["local_head"]:
        problems.append("pr head %s != local head %s"
                        % (meta.get("headRefOid"), push["local_head"]))
    if meta.get("baseRefName") != base:
        problems.append("pr base %s != expected %s" % (meta.get("baseRefName"), base))
    if meta.get("state") != "OPEN" or not meta.get("isDraft"):
        problems.append("pr must stay OPEN and Draft, got state=%s draft=%s"
                        % (meta.get("state"), meta.get("isDraft")))
    if problems:
        return {**push, "ok": False, "stage": "pr_verify", "status": "PR_VERIFY_FAILED",
                "problems": problems, "pr": number, "url": url}

    return {**push, "ok": True, "stage": stage, "status": "PR_SYNC_PASS", "pr": number,
            "url": url, "pr_head_sha": meta.get("headRefOid"),
            "three_way_equal": three_way_equal(push["local_head"],
                                               push["remote_branch_sha"],
                                               meta.get("headRefOid") or "")}


# ------------------------------------------------------------------ entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="git_sync.py",
                                 description="Verified standard git/gh sync wrapper.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--proxy", default=None,
                    help="explicit local proxy; otherwise inherited environment only")
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
                "allowed_ref_prefix": ALLOWED_REF_PREFIX,
                "expected_repo_slug": EXPECTED_REPO_SLUG,
                "explicit_proxy_configured": bool(resolve_proxy(args.proxy)),
                "git_version": run([GIT, "--version"], 30,
                                   "git-version")["stdout"].strip()}
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return 0

    try:
        if args.command == "push":
            out = do_push(args.repo, args.branch, remote=args.remote,
                          timeout_s=args.timeout, progress_path=args.progress,
                          pr=args.pr, proxy=args.proxy)
        else:
            resolved = resolve_repo(args.repo, args.timeout, args.progress)
            local = local_head(resolved, args.timeout, args.progress)
            remote_sha = remote_branch_sha(resolved, args.remote, args.branch,
                                           args.timeout, args.progress)["sha"]
            if args.pr is None:
                out = {"local_head": local, "remote_branch_sha": remote_sha,
                       "pr_head_sha": None, "pr_sync": "PR_NOT_CHECKED",
                       "git_sync_ok": git_sync_verdict(local, remote_sha),
                       "three_way_equal": None}
            else:
                meta = pr_view(resolved, args.pr, args.timeout, args.progress)
                pr_sha = meta.get("headRefOid") if meta.get("ok") else ""
                out = {"local_head": local, "remote_branch_sha": remote_sha,
                       "pr_head_sha": pr_sha or None,
                       "pr_sync": ("PR_SYNC_PASS" if meta.get("ok") and
                                   three_way_equal(local, remote_sha, pr_sha)
                                   else "PR_SYNC_FAIL"),
                       "git_sync_ok": git_sync_verdict(local, remote_sha),
                       "three_way_equal": three_way_equal(local, remote_sha, pr_sha)}
    except SyncError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False,
                         indent=2))
        return 4

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out.get("timed_out"):
        return 3
    if out.get("three_way_equal") is True or out.get("git_sync_ok") is True:
        return 0
    return 3


if __name__ == "__main__":
    sys.exit(main())
