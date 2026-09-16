#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin, verified wrapper around **standard** git/gh for PokerSense development.

Not a Git implementation and not a publishing platform: it only invokes the real
`git` and `gh` binaries with a configuration verified on this machine, and it
refuses to report success without the applicable proof.

Round 1 (BRIDGE/GITHUB-001)
    * credential helper ordering: `-c credential.helper=X` only APPENDS, so the
      pre-existing Git Credential Manager helper still ran first and blocked.
      Reset the list (`-c credential.helper=`) before adding the gh helper.

Round 2 (review GITHUB-001-R1)
    * owned process groups, repo binding, a single execution path for every
      network step, and honest two-way vs three-way verdicts.

Round 3 (review GITHUB-001-R2)
    * **one execution context**: the proxy (and timeout/progress) travels with
      the context to EVERY call — push, ls-remote, pr view/list/create, verify.
      A proxy passed only to the push is what produced "push fine, verify cannot
      connect".
    * **fail-closed pre-write validation**: an unreadable or empty remote
      identity is a rejection, not a skipped check; the remote HOST and repo path
      are parsed, not substring-matched; HEAD, the full source ref and the
      approved SHA must all agree *before* the push runs.
    * **the CLI entry owns the verdict**: `main()` no longer treats "SHA happens
      to be equal" as success when the command failed, and with `--pr` it can no
      longer degrade to a two-way result. `verify` keeps its read errors.

It never forces anything: no `--force`, no `reset`, no `rebase`, no `clean`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse

GIT = "git"
GH = "gh"

HELPER_RESET = ["-c", "credential.helper="]
GH_HELPER = ["-c", "credential.helper=!gh auth git-credential"]

PROGRESS_INTERVAL_S = 20.0
TERMINATE_GRACE_S = 15.0

# Pushed branches in this repository must be task branches, never main.
ALLOWED_REF_PREFIX = "refs/heads/codex/"

EXPECTED_HOST = "github.com"
EXPECTED_REPO_SLUG = "xiaoyangpeng1994-create/PokerSense"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SyncError(RuntimeError):
    """Configuration or precondition failure. Never silently swallowed."""


def is_sha(value) -> bool:
    return bool(value) and bool(_HEX40.match(str(value).strip()))


# ------------------------------------------------------------------ context


class Ctx:
    """One execution context per invocation.

    Carries repo / proxy / timeout / progress so a single configuration reaches
    every child process. It never writes to the global environment.
    """

    __slots__ = ("repo", "proxy", "timeout_s", "progress_path")

    def __init__(self, repo: str, proxy: str | None = None,
                 timeout_s: float = 120.0, progress_path: str | None = None):
        self.repo = repo
        self.proxy = proxy
        self.timeout_s = timeout_s
        self.progress_path = progress_path

    def derive(self, **kw) -> "Ctx":
        data = {"repo": self.repo, "proxy": self.proxy,
                "timeout_s": self.timeout_s, "progress_path": self.progress_path}
        data.update(kw)
        return Ctx(**data)


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

    LIMITATION (stated, not hidden): `survivors` only proves the group leader is
    gone. It does NOT prove every descendant exited. Do not claim full-tree
    termination on the strength of this field.
    """
    result = {"kill_attempted": True, "kill_ok": False, "survivors": [],
              "kill_scope": "group-leader-only"}
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
                "proxy_configured": bool(resolve_proxy(proxy)), "at_utc": _utc(),
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


def command_ok(res: dict) -> bool:
    """A child command only counts as OK when it exited cleanly and was reaped."""
    return (res.get("exit_code") == 0 and not res.get("timed_out")
            and res.get("kill_ok") is not False)


# ------------------------------------------------------------------ repo identity


def resolve_repo(ctx: Ctx) -> str:
    """Resolve and validate the repository before any remote write."""
    res = run(build_git_command(["rev-parse", "--show-toplevel"], repo=ctx.repo,
                                use_gh_helper=False),
              ctx.timeout_s, "git-resolve-repo", ctx.progress_path,
              proxy=ctx.proxy)
    top = (res["stdout"] or "").strip()
    if not command_ok(res) or not top:
        raise SyncError("not a git repository: %r (%s)"
                        % (ctx.repo, (res["stderr"] or "").strip()[:200]))
    return top


def parse_remote(url: str):
    """Parse a remote URL into (host, slug). Raises when it cannot be trusted."""
    raw = (url or "").strip()
    if not raw:
        raise SyncError("remote url is empty")
    if "://" not in raw:
        # scp-like: git@github.com:owner/repo.git
        host, _, path = raw.partition(":")
        host = host.rsplit("@", 1)[-1]   # scp-like user@host
        if not host or not path:
            raise SyncError("unrecognised remote url: %r" % raw)
    else:
        parts = urllib.parse.urlsplit(raw)
        host = parts.hostname or ""
        path = parts.path or ""
    slug = path.strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    if not host or "/" not in slug:
        raise SyncError("unrecognised remote url: %r" % raw)
    return host.lower(), slug


def read_remote_identity(ctx: Ctx, remote: str) -> dict:
    """Read and STRICTLY validate the remote identity (fail closed)."""
    res = run(build_git_command(["remote", "get-url", remote], repo=ctx.repo,
                                use_gh_helper=False),
              ctx.timeout_s, "git-remote-url", ctx.progress_path,
              proxy=ctx.proxy)
    url = (res["stdout"] or "").strip()
    if not command_ok(res) or not url:
        return {"ok": False, "host": "", "slug": "", "url": "",
                "error": "cannot read remote %r: %s"
                         % (remote, (res["stderr"] or "no output").strip()[:160])}
    try:
        host, slug = parse_remote(url)
    except SyncError as exc:
        return {"ok": False, "host": "", "slug": "", "url": url, "error": str(exc)}
    return {"ok": True, "host": host, "slug": slug, "url": url, "error": ""}


def read_ref_sha(ctx: Ctx, ref: str) -> dict:
    """Resolve a full ref (e.g. refs/heads/codex/x) to a valid commit id."""
    res = run(build_git_command(["rev-parse", "--verify", "--quiet", ref],
                                repo=ctx.repo, use_gh_helper=False),
              ctx.timeout_s, "git-rev-parse", ctx.progress_path, proxy=ctx.proxy)
    sha = (res["stdout"] or "").strip()
    if not command_ok(res) or not is_sha(sha):
        return {"ok": False, "sha": "", "ref": ref,
                "error": "cannot resolve %s: %s"
                         % (ref, (res["stderr"] or "no valid object id").strip()[:160])}
    return {"ok": True, "sha": sha, "ref": ref, "error": ""}


def validate_push_target(ctx: Ctx, remote: str, branch: str, approved_sha: str,
                         must_be_task_branch: bool = True,
                         require_head_match: bool = True) -> dict:
    """Everything that must hold BEFORE a remote write. Fails closed."""
    problems = []
    ref = "refs/heads/" + branch
    if must_be_task_branch and not ref.startswith(ALLOWED_REF_PREFIX):
        problems.append("ref %r is outside the allowed prefix %r"
                        % (ref, ALLOWED_REF_PREFIX))

    ident = read_remote_identity(ctx, remote)
    if not ident["ok"]:
        problems.append(ident["error"])
    else:
        if ident["host"] != EXPECTED_HOST:
            problems.append("remote host %r is not %r" % (ident["host"], EXPECTED_HOST))
        if ident["slug"] != EXPECTED_REPO_SLUG:
            problems.append("remote repository %r is not %r"
                            % (ident["slug"], EXPECTED_REPO_SLUG))

    source = read_ref_sha(ctx, ref)
    if not source["ok"]:
        problems.append(source["error"])
    else:
        if not is_sha(approved_sha):
            problems.append("approved SHA %r is not a commit id" % (approved_sha,))
        elif source["sha"] != approved_sha:
            # The reviewed failure: HEAD was compared with itself while the push
            # wrote a different local branch.
            problems.append("source ref %s is %s but the approved commit is %s"
                            % (ref, source["sha"], approved_sha))

    head = read_ref_sha(ctx, "HEAD")
    if not head["ok"]:
        problems.append(head["error"])
    elif require_head_match and source["ok"] and head["sha"] != source["sha"]:
        problems.append("HEAD %s does not match the source ref %s (%s)"
                        % (head["sha"], ref, source["sha"]))

    return {"ok": not problems, "problems": problems,
            "host": ident.get("host", ""), "slug": ident.get("slug", ""),
            "source_ref": ref, "source_sha": source.get("sha", ""),
            "head_sha": head.get("sha", ""), "branch": branch}


# ------------------------------------------------------------------ remote reads


def remote_branch_sha(ctx: Ctx, remote: str, branch: str) -> dict:
    """Read the remote ref through the SAME context (proxy/timeout/credentials)."""
    res = run(build_git_command(["ls-remote", remote, "refs/heads/" + branch],
                                repo=ctx.repo),
              ctx.timeout_s, "git-ls-remote", ctx.progress_path, proxy=ctx.proxy)
    sha = ""
    if command_ok(res):
        lines = (res["stdout"] or "").strip().splitlines()
        if lines:
            sha = lines[0].split()[0]
    ok = command_ok(res) and is_sha(sha)
    return {"ok": ok, "sha": sha if ok else "",
            "exit_code": res["exit_code"], "timed_out": res["timed_out"],
            "error": "" if ok else ("ls-remote failed: %s"
                                    % (res["stderr"] or "no output").strip()[:160])}


def pr_view(ctx: Ctx, pr: int) -> dict:
    """PR metadata via gh through the same context. Errors are reported."""
    res = run([GH, "pr", "view", str(pr), "--json",
               "number,state,isDraft,baseRefName,headRefName,headRefOid,url,"
               "headRepositoryOwner,headRepository"],
              ctx.timeout_s, "gh-pr-view", ctx.progress_path, cwd=ctx.repo,
              proxy=ctx.proxy)
    if not command_ok(res):
        return {"ok": False, "error": (res["stderr"] or "").strip()[:200],
                "timed_out": res["timed_out"], "exit_code": res["exit_code"]}
    try:
        data = json.loads(res["stdout"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "unparsable gh output: %s" % exc}
    data["ok"] = True
    return data


def pr_list_for_branch(ctx: Ctx, branch: str) -> dict:
    """List PRs for the head branch. An API failure is an ERROR, not an empty list."""
    res = run([GH, "pr", "list", "--head", branch, "--state", "all", "--json",
               "number,isDraft,state,baseRefName,url,headRefOid"],
              ctx.timeout_s, "gh-pr-list", ctx.progress_path, cwd=ctx.repo,
              proxy=ctx.proxy)
    if not command_ok(res):
        return {"ok": False, "error": (res["stderr"] or "").strip()[:200],
                "timed_out": res["timed_out"], "exit_code": res["exit_code"],
                "prs": []}
    try:
        prs = json.loads(res["stdout"])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "unparsable gh output: %s" % exc, "prs": []}
    return {"ok": True, "prs": prs if isinstance(prs, list) else []}


# ------------------------------------------------------------------ verdicts


def git_sync_verdict(local_sha: str, remote_sha: str) -> bool:
    """Two-way: local source ref vs remote branch ref. Empty never equals."""
    return is_sha(local_sha) and is_sha(remote_sha) and local_sha == remote_sha


def three_way_equal(a: str, b: str, c: str) -> bool:
    return is_sha(a) and is_sha(b) and is_sha(c) and a == b == c


def check_pr_sync(ctx: Ctx, pr: int, branch: str, base: str,
                  local_sha: str) -> dict:
    """Validate the PR at the CLI entry, not only inside publish()."""
    meta = pr_view(ctx, pr)
    if not meta.get("ok"):
        return {"pr_sync": "PR_READ_ERROR", "pr_error": meta.get("error"),
                "pr_head_sha": None, "problems": ["PR API read failed"]}
    problems = []
    head = meta.get("headRefOid") or ""
    if head != local_sha:
        problems.append("pr head %s != local %s" % (head or "?", local_sha))
    if meta.get("headRefName") and meta.get("headRefName") != branch:
        problems.append("pr head branch %s != requested %s"
                        % (meta.get("headRefName"), branch))
    if meta.get("baseRefName") != base:
        problems.append("pr base %s != expected %s" % (meta.get("baseRefName"), base))
    if meta.get("state") != "OPEN":
        problems.append("pr state %s is not OPEN" % meta.get("state"))
    if not meta.get("isDraft"):
        problems.append("pr must stay Draft")
    if not three_way_equal(local_sha, local_sha, head):
        problems.append("pr head is not the commit being published")
    return {"pr_sync": "PR_SYNC_PASS" if not problems else "PR_SYNC_FAIL",
            "pr_head_sha": head or None, "pr_state": meta.get("state"),
            "pr_base": meta.get("baseRefName"),
            "pr_head_branch": meta.get("headRefName"), "problems": problems,
            "pr_error": ""}


def do_push(ctx: Ctx, branch: str, remote: str = "origin",
            pr: int | None = None, base: str = "main",
            dry_run: bool = False, expected_sha: str | None = None,
            must_be_task_branch: bool = True,
            require_head_match: bool = True) -> dict:
    """Standard `git push`, then grade GIT_SYNC and, only with a PR, PR_SYNC."""
    if dry_run:
        return {"dry_run": True,
                "command": build_git_command(
                    ["push", "--dry-run", remote,
                     "refs/heads/%s:refs/heads/%s" % (branch, branch)],
                    repo=ctx.repo)}

    resolved = resolve_repo(ctx)
    ctx = ctx.derive(repo=resolved)

    source = read_ref_sha(ctx, "refs/heads/" + branch)
    approved = expected_sha or source.get("sha") or ""
    guard = validate_push_target(ctx, remote, branch, approved,
                                 must_be_task_branch, require_head_match)
    if not guard["ok"]:
        return {"exit_code": None, "timed_out": False, "command_ok": False,
                "git_sync_ok": False, "pr_sync": "PR_NOT_CHECKED",
                "three_way_equal": False, "blocked_before_write": True,
                "problems": guard["problems"],
                "approved_sha": approved,
                "source_ref": guard["source_ref"],
                "source_sha": guard["source_sha"],
                "local_head": guard["head_sha"], "remote_branch_sha": "",
                "pr_head_sha": None, "remote_read_error": "not attempted"}

    push = run(build_git_command(
        ["push", remote, "refs/heads/%s:refs/heads/%s" % (branch, branch)],
        repo=resolved),
        ctx.timeout_s, "git-push", ctx.progress_path, proxy=ctx.proxy)

    local = read_ref_sha(ctx, "refs/heads/" + branch)
    remote_info = remote_branch_sha(ctx, remote, branch)
    git_ok = git_sync_verdict(local.get("sha", ""), remote_info.get("sha", ""))
    cmd_ok = command_ok(push)

    verdict = {
        "resolved_repo": resolved,
        "exit_code": push["exit_code"],
        "timed_out": push["timed_out"],
        "kill_ok": push.get("kill_ok"),
        "command_ok": cmd_ok,
        "duration_s": push["duration_s"],
        "proxy_configured": bool(resolve_proxy(ctx.proxy)),
        "local_head": local.get("sha", ""),
        "source_ref": "refs/heads/" + branch,
        "remote_branch_sha": remote_info.get("sha", ""),
        "remote_read_error": remote_info.get("error", ""),
        "git_sync_ok": git_ok,
        "stderr_tail": (push["stderr"].strip().splitlines()[-4:]
                        if push["stderr"] else []),
    }
    # A command error whose postcondition is already satisfied is NOT plain
    # success: keep it visible instead of letting the equal SHA hide it.
    verdict["postcondition_only"] = bool(git_ok and not cmd_ok)

    if pr is None:
        verdict["pr_head_sha"] = None
        verdict["pr_sync"] = "PR_NOT_CHECKED"
        verdict["three_way_equal"] = None
        return verdict

    prc = check_pr_sync(ctx, pr, branch, base, local.get("sha", ""))
    verdict.update(prc)
    verdict["three_way_equal"] = (prc["pr_sync"] == "PR_SYNC_PASS")
    return verdict


def publish(ctx: Ctx, branch: str, title: str, body_file: str, base: str,
            remote: str = "origin") -> dict:
    """Push, then create-or-reuse a PR for this branch and VERIFY its head.

    Idempotent and honest: it does NOT create commits (the caller commits).
    """
    push = do_push(ctx, branch, remote=remote, base=base)
    if push.get("blocked_before_write"):
        return {**push, "ok": False, "stage": "precondition"}
    if push.get("exit_code") != 0 or not push.get("command_ok", False):
        status = ("POSTCONDITION_CONFIRMED_WITH_COMMAND_ERROR"
                  if push.get("git_sync_ok") else "PUSH_FAILED")
        return {**push, "ok": False, "stage": "push", "status": status}
    if not push.get("git_sync_ok"):
        return {**push, "ok": False, "stage": "git_sync", "status": "GIT_SYNC_FAIL"}

    listing = pr_list_for_branch(ctx, branch)
    if not listing["ok"]:
        # An API failure is an error, NOT "no PR exists".
        return {**push, "ok": False, "stage": "pr_list",
                "status": "PR_LIST_API_ERROR", "error": listing.get("error")}

    reusable = None
    for cand in listing["prs"]:
        if (cand.get("state") == "OPEN" and cand.get("baseRefName") == base
                and cand.get("isDraft")):
            reusable = cand
            break
    if listing["prs"] and reusable is None:
        return {**push, "ok": False, "stage": "pr_reuse",
                "status": "EXISTING_PR_INCOMPATIBLE",
                "found": [{"number": c.get("number"), "state": c.get("state"),
                           "draft": c.get("isDraft"), "base": c.get("baseRefName")}
                          for c in listing["prs"]]}

    if reusable:
        number, url, stage = reusable["number"], reusable["url"], "pr_reused"
    else:
        created = run([GH, "pr", "create", "--draft", "--base", base,
                       "--head", branch, "--title", title, "--body-file", body_file],
                      ctx.timeout_s, "gh-pr-create", ctx.progress_path,
                      cwd=ctx.repo, proxy=ctx.proxy)
        if not command_ok(created):
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

    prc = check_pr_sync(ctx, number, branch, base, push.get("local_head", ""))
    if prc["pr_sync"] != "PR_SYNC_PASS":
        return {**push, **prc, "ok": False, "stage": "pr_verify",
                "status": "PR_VERIFY_FAILED", "pr": number, "url": url}

    return {**push, **prc, "ok": True, "stage": stage,
            "status": "PR_SYNC_PASS", "pr": number, "url": url,
            "three_way_equal": True}


# ------------------------------------------------------------------ entry


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _exit_code(out: dict, pr_given: bool) -> int:
    """Single success criterion for the CLI entry.

    With --pr, a two-way result must never count as success; a command error or
    timeout must never be hidden by an equal SHA.
    """
    if out.get("timed_out"):
        return 3
    if out.get("kill_ok") is False:
        return 3
    if pr_given:
        if out.get("pr_sync") != "PR_SYNC_PASS":
            return 3
        return 0 if out.get("command_ok", True) else 3
    if not out.get("git_sync_ok"):
        return 3
    return 0 if out.get("command_ok", True) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="git_sync.py",
                                 description="Verified standard git/gh sync wrapper.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--base", default="main")
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
        _print({"git": GIT, "gh": GH,
                "helper_reset": HELPER_RESET, "gh_helper": GH_HELPER,
                "allowed_ref_prefix": ALLOWED_REF_PREFIX,
                "expected_host": EXPECTED_HOST,
                "expected_repo_slug": EXPECTED_REPO_SLUG,
                "explicit_proxy_configured": bool(resolve_proxy(args.proxy)),
                "git_version": run([GIT, "--version"], 30,
                                   "git-version")["stdout"].strip()})
        return 0

    ctx = Ctx(args.repo, proxy=args.proxy, timeout_s=args.timeout,
              progress_path=args.progress)
    try:
        if args.command == "push":
            out = do_push(ctx, args.branch, remote=args.remote, pr=args.pr,
                          base=args.base)
        else:
            resolved = resolve_repo(ctx)
            ctx = ctx.derive(repo=resolved)
            local = read_ref_sha(ctx, "refs/heads/" + args.branch)
            remote = remote_branch_sha(ctx, args.remote, args.branch)
            prc = ({"pr_sync": "PR_NOT_CHECKED", "pr_head_sha": None,
                    "pr_error": "", "problems": []} if args.pr is None
                   else check_pr_sync(ctx, args.pr, args.branch, args.base,
                                      local.get("sha", "")))
            out = {"local_head": local.get("sha", ""),
                   "local_ref_error": local.get("error", ""),
                   "source_ref": "refs/heads/" + args.branch,
                   "remote_branch_sha": remote.get("sha", ""),
                   "remote_read_error": remote.get("error", ""),
                   "git_sync_ok": git_sync_verdict(local.get("sha", ""),
                                                   remote.get("sha", "")),
                   "proxy_configured": bool(resolve_proxy(ctx.proxy)), **prc}
            out["three_way_equal"] = (prc["pr_sync"] == "PR_SYNC_PASS"
                                      if args.pr is not None else None)
    except SyncError as exc:
        _print({"ok": False, "error": str(exc)})
        return 4

    _print(out)
    return _exit_code(out, args.pr is not None)


if __name__ == "__main__":
    sys.exit(main())
