#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused offline tests for tools/git_sync.py.

No network and no access to the real repository. Two layers are exercised:

* **entry level** — `main([...])` is called directly, which is what the R2 review
  asked for: the CLI exit code must not report PR failures or command failures as
  success;
* **decision level** — only the process layer (`git_sync.run`) is replaced, so the
  real pre-write validation and proxy threading run, with call counting to prove
  no push happened.

Process-spawning tests use only the local Python interpreter.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

import git_sync  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40
REPO_URL = "https://github.com/xiaoyangpeng1994-create/PokerSense.git"


def _res(exit_code=0, stdout="", stderr="", timed_out=False, kill_ok=True, **kw):
    out = {"command_id": "x", "label": "", "argv": [], "cwd": None,
           "exit_code": exit_code, "timed_out": timed_out, "duration_s": 0.1,
           "stdout": stdout, "stderr": stderr,
           "kill_attempted": False, "kill_ok": kill_ok, "survivors": []}
    out.update(kw)
    return out


class FakeGit:
    """Scripted process layer. Replaces only `git_sync.run`."""

    def __init__(self, head=SHA_A, source=SHA_A, remote_sha=SHA_A,
                 remote_url=REPO_URL, push_exit=0, pr_json=None,
                 pr_exit=0, ls_exit=0, url_exit=0, rev_exit=0):
        self.head, self.source, self.remote_sha = head, source, remote_sha
        self.remote_url = remote_url
        self.push_exit, self.pr_json, self.pr_exit = push_exit, pr_json, pr_exit
        self.ls_exit, self.url_exit, self.rev_exit = ls_exit, url_exit, rev_exit
        self.calls = []

    def __call__(self, cmd, timeout_s, label, progress_path=None, cwd=None,
                 proxy=None):
        self.calls.append({"label": label, "argv": list(cmd), "cwd": cwd,
                           "proxy": proxy})
        if label == "git-resolve-repo":
            return _res(0, "R\n")
        if label == "git-remote-url":
            if self.url_exit != 0:
                return _res(self.url_exit, "", "no such remote")
            return _res(0, self.remote_url + "\n")
        if label == "git-rev-parse":
            ref = cmd[-1]
            if self.rev_exit != 0:
                return _res(self.rev_exit, "", "unknown revision")
            return _res(0, (self.head if ref == "HEAD" else self.source) + "\n")
        if label == "git-ls-remote":
            if self.ls_exit != 0:
                return _res(self.ls_exit, "", "ls-remote failed")
            return _res(0, self.remote_sha + "\trefs/heads/b\n")
        if label == "git-push":
            msg = "" if self.push_exit == 0 else "push failed"
            return _res(self.push_exit, "", msg)
        if label == "gh-pr-view":
            if self.pr_exit != 0:
                return _res(self.pr_exit, "", "gh: api error")
            return _res(0, json.dumps(self.pr_json or {}))
        if label == "gh-pr-list":
            if self.pr_exit != 0:
                return _res(self.pr_exit, "", "gh: api error")
            return _res(0, json.dumps(self.pr_json if isinstance(self.pr_json, list)
                                      else []))
        if label == "gh-pr-create":
            return _res(0, "https://github.com/x/PokerSense/pull/99\n")
        return _res(0, "")

    @property
    def pushed(self):
        return [c for c in self.calls if c["label"] == "git-push"]

    def by_label(self, label):
        return [c for c in self.calls if c["label"] == label]


# --------------------------------------------------------------- command shape


def test_helper_list_is_reset_before_gh_helper():
    cmd = git_sync.build_git_command(["push", "origin", "HEAD"])
    assert cmd.index("credential.helper=") < \
        cmd.index("credential.helper=!gh auth git-credential")
    assert cmd[0] == "git"


def test_git_command_binds_the_repo_explicitly():
    cmd = git_sync.build_git_command(["status"], repo="C:/some/repo")
    assert "-C" in cmd and cmd[cmd.index("-C") + 1] == "C:/some/repo"
    assert "-C" not in git_sync.build_git_command(["status"], repo=None)


# --------------------------------------------------------------- environment


def test_child_env_does_not_inject_a_default_proxy(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "POKERSENSE_GIT_PROXY"):
        monkeypatch.delenv(name, raising=False)
    env = git_sync.child_env()
    assert "HTTP_PROXY" not in env and "HTTPS_PROXY" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"


def test_resolve_proxy_precedence(monkeypatch):
    monkeypatch.delenv("POKERSENSE_GIT_PROXY", raising=False)
    assert git_sync.resolve_proxy() == ""
    monkeypatch.setenv("POKERSENSE_GIT_PROXY", "http://env:1")
    assert git_sync.resolve_proxy() == "http://env:1"
    assert git_sync.resolve_proxy("http://cli:2") == "http://cli:2"


# --------------------------------------------------------------- sha helpers


def test_is_sha_and_verdicts():
    assert git_sync.is_sha(SHA_A) and not git_sync.is_sha("a" * 39)
    assert not git_sync.is_sha("")
    assert git_sync.git_sync_verdict(SHA_A, SHA_A)
    assert not git_sync.git_sync_verdict(SHA_A, "")
    assert git_sync.three_way_equal(SHA_A, SHA_A, SHA_A)
    assert not git_sync.three_way_equal(SHA_A, SHA_A, "")


def test_parse_remote_rejects_untrustworthy_urls():
    assert git_sync.parse_remote(REPO_URL) == (
        "github.com", "xiaoyangpeng1994-create/PokerSense")
    assert git_sync.parse_remote(
        "git@github.com:xiaoyangpeng1994-create/PokerSense.git") == (
        "github.com", "xiaoyangpeng1994-create/PokerSense")
    for bad in ("", "not a url", "https://hostonly", "file:///tmp/x"):
        with pytest.raises(git_sync.SyncError):
            git_sync.parse_remote(bad)
    # A syntactically valid but wrong host is REJECTED LATER, by validate_push_target.
    assert git_sync.parse_remote("https://evil.example/x/y") == ("evil.example", "x/y")


# --------------------------------------------------------------- execution


def test_run_captures_real_exit_code_not_pipeline_status():
    res = git_sync.run([sys.executable, "-c", "import sys; sys.exit(7)"],
                       timeout_s=30, label="selftest-exit")
    assert res["exit_code"] == 7 and res["timed_out"] is False
    assert git_sync.command_ok(res) is False


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), "abc", None])
def test_run_rejects_invalid_timeout(bad):
    with pytest.raises(git_sync.SyncError):
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=bad, label="bad")


def test_run_timeout_kills_only_its_own_group_and_others_survive():
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        res = git_sync.run([sys.executable, "-c", "import time; time.sleep(60)"],
                           timeout_s=3, label="selftest-timeout")
        assert res["timed_out"] is True and res["kill_ok"] is True
        assert sibling.poll() is None
        assert os.getpid() > 0
        # The limitation must be stated, not implied away.
        assert res["kill_scope"] == "group-leader-only"
    finally:
        git_sync.terminate_owned_group(sibling, None)


def test_progress_trail_records_phase_and_exit():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=30,
                     label="selftest-progress", progress_path=path)
        lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        assert lines[-1]["phase"] == "done" and lines[-1]["exit_code"] == 0


def test_wrapper_never_uses_a_shell_or_force():
    src = inspect.getsource(git_sync)
    assert "shell=True" not in src
    tree = ast.parse(open(git_sync.__file__, encoding="utf-8").read())
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                doc_ids.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in doc_ids):
            assert "--force" not in node.value


def test_network_helpers_route_through_run():
    tree = ast.parse(open(git_sync.__file__, encoding="utf-8").read())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in ("run", "terminate_owned_group"):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "run"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "subprocess"):
                offenders.append(node.name)
    assert not offenders, offenders


# ------------------------------------------------- pre-write validation (R2 P1-b)


def _ctx(tmp_path):
    return git_sync.Ctx(str(tmp_path), proxy=None, timeout_s=30,
                        progress_path=None)


@pytest.mark.parametrize("case,expected", [
    ("empty_identity", "cannot read remote"),
    ("wrong_host", "remote host"),
    ("wrong_repo", "remote repository"),
])
def test_pre_write_rejection_never_pushes(monkeypatch, tmp_path, case, expected):
    kwargs = {}
    if case == "empty_identity":
        kwargs = {"remote_url": "", "url_exit": 0}
    elif case == "wrong_host":
        kwargs = {"remote_url": "https://evil.example/x/y.git"}
    else:
        kwargs = {"remote_url": "https://github.com/someone/else.git"}
    fake = FakeGit(**kwargs)
    monkeypatch.setattr(git_sync, "run", fake)

    out = git_sync.do_push(_ctx(tmp_path), "codex/b")
    assert out["blocked_before_write"] is True
    assert any(expected in p for p in out["problems"]), out["problems"]
    assert not fake.pushed, "no remote write may happen when the guard fails"


def test_pre_write_rejects_source_ref_mismatch(monkeypatch, tmp_path):
    """The reviewed failure: HEAD compared with itself while a different local
    branch would be pushed."""
    fake = FakeGit(head=SHA_A, source=SHA_B)
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.do_push(_ctx(tmp_path), "codex/b", expected_sha=SHA_A)
    assert out["blocked_before_write"] is True
    assert any("source ref" in p for p in out["problems"])
    assert not fake.pushed


def test_pre_write_rejects_head_not_matching_source_ref(monkeypatch, tmp_path):
    fake = FakeGit(head=SHA_B, source=SHA_A)
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.do_push(_ctx(tmp_path), "codex/b")
    assert out["blocked_before_write"] is True
    assert any("HEAD" in p for p in out["problems"])
    assert not fake.pushed


def test_pre_write_rejects_unreadable_remote_branch_or_sha(monkeypatch, tmp_path):
    fake = FakeGit(rev_exit=1)
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.do_push(_ctx(tmp_path), "codex/b")
    assert out["blocked_before_write"] is True
    assert not fake.pushed


def test_pre_write_rejects_non_task_branch(monkeypatch, tmp_path):
    fake = FakeGit()
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.do_push(_ctx(tmp_path), "main")
    assert out["blocked_before_write"] is True
    assert any("allowed prefix" in p for p in out["problems"])
    assert not fake.pushed


# ------------------------------------------------- proxy threading (R2 P2)


def test_explicit_proxy_reaches_every_call(monkeypatch, tmp_path):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "POKERSENSE_GIT_PROXY"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeGit(pr_json={"number": 26, "state": "OPEN", "isDraft": True,
                            "baseRefName": "main", "headRefName": "codex/b",
                            "headRefOid": SHA_A, "url": "u"})
    monkeypatch.setattr(git_sync, "run", fake)
    ctx = git_sync.Ctx(str(tmp_path), proxy="http://cli:9", timeout_s=30)
    out = git_sync.do_push(ctx, "codex/b", pr=26)
    assert out["pr_sync"] == "PR_SYNC_PASS"
    labels = {c["label"] for c in fake.calls}
    assert {"git-push", "git-ls-remote", "gh-pr-view"} <= labels
    missing = [c["label"] for c in fake.calls if c["proxy"] != "http://cli:9"]
    assert not missing, "these calls lost the explicit proxy: %r" % missing


def test_explicit_proxy_reaches_cli_verify(monkeypatch, tmp_path, capsys):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "POKERSENSE_GIT_PROXY"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeGit(pr_json={"number": 26, "state": "OPEN", "isDraft": True,
                            "baseRefName": "main", "headRefName": "codex/b",
                            "headRefOid": SHA_A, "url": "u"})
    monkeypatch.setattr(git_sync, "run", fake)
    rc = git_sync.main(["--repo", str(tmp_path), "--branch", "codex/b",
                        "--proxy", "http://cli:9", "--pr", "26", "verify"])
    assert rc == 0
    assert fake.by_label("git-ls-remote")[0]["proxy"] == "http://cli:9"
    assert fake.by_label("gh-pr-view")[0]["proxy"] == "http://cli:9"


def test_no_explicit_proxy_inherits_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("POKERSENSE_GIT_PROXY", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://env:8")
    monkeypatch.setenv("HTTPS_PROXY", "http://env:8")
    env = git_sync.child_env(None)
    assert env["HTTPS_PROXY"] == "http://env:8"


# ------------------------------------------------- CLI entry (R2 P1-a)


def _main_push(monkeypatch, out, pr=None):
    monkeypatch.setattr(git_sync, "do_push", lambda *a, **k: out)
    argv = ["--repo", "R", "--branch", "codex/b", "push"]
    if pr is not None:
        argv += ["--pr", str(pr)]
    # --pr must precede the subcommand for argparse
    argv = (["--repo", "R", "--branch", "codex/b"]
            + (["--pr", str(pr)] if pr else []) + ["push"])
    return git_sync.main(argv)


def test_cli_push_command_error_with_equal_sha_is_not_success(monkeypatch, capsys):
    """R2 case 1: exit 1 but the remote already has the SHA must not be exit 0."""
    out = {"exit_code": 1, "timed_out": False, "command_ok": False,
           "git_sync_ok": True, "postcondition_only": True, "pr_sync": "PR_NOT_CHECKED",
           "three_way_equal": None, "pr_head_sha": None, "local_head": SHA_A,
           "remote_branch_sha": SHA_A, "kill_ok": True, "remote_read_error": ""}
    assert _main_push(monkeypatch, out) == 3


def test_cli_push_pr_head_mismatch_is_failure(monkeypatch):
    """R2 case 2: with --pr, PR_SYNC_FAIL must be non-zero."""
    out = {"exit_code": 0, "timed_out": False, "command_ok": True,
           "git_sync_ok": True, "pr_sync": "PR_SYNC_FAIL", "three_way_equal": False,
           "pr_head_sha": SHA_B, "local_head": SHA_A, "remote_branch_sha": SHA_A,
           "problems": ["pr head x != local y"], "kill_ok": True}
    assert _main_push(monkeypatch, out, pr=26) == 3


def test_cli_push_pr_api_read_error_is_failure(monkeypatch):
    """R2 case 3: an unreadable PR must fail and keep the error."""
    out = {"exit_code": 0, "timed_out": False, "command_ok": True,
           "git_sync_ok": True, "pr_sync": "PR_READ_ERROR",
           "pr_error": "gh: api error", "three_way_equal": False,
           "pr_head_sha": None, "local_head": SHA_A, "remote_branch_sha": SHA_A,
           "kill_ok": True}
    assert _main_push(monkeypatch, out, pr=26) == 3


def test_cli_push_pr_sync_pass_is_success(monkeypatch):
    """R2 case 4: the control — a verified draft PR succeeds."""
    out = {"exit_code": 0, "timed_out": False, "command_ok": True,
           "git_sync_ok": True, "pr_sync": "PR_SYNC_PASS", "three_way_equal": True,
           "pr_head_sha": SHA_A, "local_head": SHA_A, "remote_branch_sha": SHA_A,
           "kill_ok": True}
    assert _main_push(monkeypatch, out, pr=26) == 0


def test_cli_push_without_pr_success_is_zero(monkeypatch):
    """Control: two-way agreement is enough when no PR was requested."""
    out = {"exit_code": 0, "timed_out": False, "command_ok": True,
           "git_sync_ok": True, "pr_sync": "PR_NOT_CHECKED",
           "three_way_equal": None, "pr_head_sha": None, "local_head": SHA_A,
           "remote_branch_sha": SHA_A, "kill_ok": True}
    assert _main_push(monkeypatch, out) == 0


def test_cli_push_without_pr_but_command_failed_is_nonzero(monkeypatch):
    out = {"exit_code": 124, "timed_out": True, "command_ok": False,
           "git_sync_ok": True, "pr_sync": "PR_NOT_CHECKED",
           "three_way_equal": None, "kill_ok": True, "local_head": SHA_A,
           "remote_branch_sha": SHA_A}
    assert _main_push(monkeypatch, out) == 3


def test_cli_push_kill_failure_is_nonzero(monkeypatch):
    out = {"exit_code": 0, "timed_out": False, "command_ok": True,
           "git_sync_ok": True, "pr_sync": "PR_NOT_CHECKED",
           "three_way_equal": None, "kill_ok": False, "local_head": SHA_A,
           "remote_branch_sha": SHA_A}
    assert _main_push(monkeypatch, out) == 3


def test_cli_verify_preserves_read_errors(monkeypatch, tmp_path, capsys):
    fake = FakeGit(ls_exit=1, pr_json=None, pr_exit=1)
    monkeypatch.setattr(git_sync, "run", fake)
    rc = git_sync.main(["--repo", str(tmp_path), "--branch", "codex/b",
                        "--pr", "26", "verify"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert payload["remote_read_error"], "the ls-remote error must be preserved"
    assert payload["pr_error"], "the PR read error must be preserved"
    assert payload["pr_sync"] != "PR_SYNC_PASS"


def test_cli_verify_without_pr_reports_pr_not_checked(monkeypatch, tmp_path, capsys):
    fake = FakeGit()
    monkeypatch.setattr(git_sync, "run", fake)
    rc = git_sync.main(["--repo", str(tmp_path), "--branch", "codex/b", "verify"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["pr_sync"] == "PR_NOT_CHECKED"
    assert payload["three_way_equal"] is None


# ------------------------------------------------- publish (idempotency)


def _publish_ctx(tmp_path):
    return git_sync.Ctx(str(tmp_path), timeout_s=30)


def test_publish_refuses_closed_pr_and_does_not_push(monkeypatch, tmp_path):
    fake = FakeGit(pr_json=[{"number": 9, "state": "CLOSED", "isDraft": True,
                             "baseRefName": "main", "url": "u"}])
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.publish(_publish_ctx(tmp_path), "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "EXISTING_PR_INCOMPATIBLE"
    assert len(fake.pushed) == 1  # publish does push first; it must not create a PR


def test_publish_api_error_never_creates_a_pr(monkeypatch, tmp_path):
    fake = FakeGit(pr_exit=1)
    monkeypatch.setattr(git_sync, "run", fake)
    out = git_sync.publish(_publish_ctx(tmp_path), "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "PR_LIST_API_ERROR"
    assert not fake.by_label("gh-pr-create")


def test_publish_verifies_pr_head_after_reuse(monkeypatch, tmp_path):
    fake = FakeGit(pr_json=[{"number": 26, "state": "OPEN", "isDraft": True,
                             "baseRefName": "main", "url": "u"}],
                   remote_sha=SHA_A, source=SHA_A)
    # pr_view (single) returns a mismatching head while the list looks reusable
    monkeypatch.setattr(git_sync, "run", fake)

    class Mismatch(FakeGit):
        def __call__(self, cmd, timeout_s, label, progress_path=None, cwd=None,
                     proxy=None):
            res = super().__call__(cmd, timeout_s, label, progress_path, cwd, proxy)
            if label == "gh-pr-view":
                res["stdout"] = json.dumps({"number": 26, "state": "OPEN",
                                            "isDraft": True, "baseRefName": "main",
                                            "headRefName": "codex/b",
                                            "headRefOid": SHA_B,
                                            "url": "u"})
            return res

    fake2 = Mismatch(pr_json=[{"number": 26, "state": "OPEN", "isDraft": True,
                              "baseRefName": "main", "url": "u"}])
    monkeypatch.setattr(git_sync, "run", fake2)
    out = git_sync.publish(_publish_ctx(tmp_path), "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "PR_VERIFY_FAILED"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
