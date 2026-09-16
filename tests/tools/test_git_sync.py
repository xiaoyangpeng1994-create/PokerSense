#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused offline tests for tools/git_sync.py.

No network and no access to the real repository. Process-spawning tests use only
the local Python interpreter; gh/git behaviour is scripted by monkeypatching
`git_sync.run`, which is also how the verdict logic is exercised.
"""

import ast
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


# --------------------------------------------------------------- command shape


def test_helper_list_is_reset_before_gh_helper():
    """`-c` appends, so the empty reset must come first or GCM still runs."""
    cmd = git_sync.build_git_command(["push", "origin", "HEAD"])
    assert cmd.index("credential.helper=") < \
        cmd.index("credential.helper=!gh auth git-credential")
    assert cmd[0] == "git"
    assert cmd[-3:] == ["push", "origin", "HEAD"]


def test_git_command_binds_the_repo_explicitly():
    """P1-B: every git call must be anchored to the resolved repo."""
    cmd = git_sync.build_git_command(["status"], repo="C:/some/repo")
    assert "-C" in cmd
    assert cmd[cmd.index("-C") + 1] == "C:/some/repo"
    assert "-C" not in git_sync.build_git_command(["status"], repo=None)


def test_reset_only_when_gh_helper_disabled():
    cmd = git_sync.build_git_command(["status"], use_gh_helper=False)
    assert "credential.helper=" in cmd
    assert not any("gh auth git-credential" in p for p in cmd)


# --------------------------------------------------------------- environment


def test_child_env_does_not_inject_a_default_proxy(monkeypatch):
    """Alignment 1: no hardcoded loopback proxy for machines that lack one."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "POKERSENSE_GIT_PROXY"):
        monkeypatch.delenv(name, raising=False)
    env = git_sync.child_env()
    assert "HTTP_PROXY" not in env and "HTTPS_PROXY" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"


def test_child_env_uses_explicit_proxy_only(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "POKERSENSE_GIT_PROXY"):
        monkeypatch.delenv(name, raising=False)
    assert git_sync.child_env(proxy="http://127.0.0.1:9")["HTTPS_PROXY"] == \
        "http://127.0.0.1:9"


def test_resolve_proxy_reads_env_var(monkeypatch):
    monkeypatch.setenv("POKERSENSE_GIT_PROXY", "http://127.0.0.1:9")
    assert git_sync.resolve_proxy() == "http://127.0.0.1:9"
    assert git_sync.resolve_proxy("http://x:1") == "http://x:1"


# --------------------------------------------------------------- sha verdicts


def test_three_way_equal_requires_identical_non_empty():
    assert git_sync.three_way_equal(SHA_A, SHA_A, SHA_A)
    assert not git_sync.three_way_equal(SHA_A, SHA_A, SHA_B)
    assert not git_sync.three_way_equal(SHA_A, "", SHA_A)
    assert not git_sync.three_way_equal("", "", "")


def test_git_sync_verdict_is_two_way_only():
    assert git_sync.git_sync_verdict(SHA_A, SHA_A)
    assert not git_sync.git_sync_verdict(SHA_A, "")
    assert not git_sync.git_sync_verdict("", SHA_A)


# --------------------------------------------------------------- execution


def test_run_captures_real_exit_code_not_pipeline_status():
    res = git_sync.run([sys.executable, "-c", "import sys; sys.exit(7)"],
                       timeout_s=30, label="selftest-exit")
    assert res["exit_code"] == 7
    assert res["timed_out"] is False


def test_run_captures_stdout_and_stderr():
    code = "import sys;print('OUT-LINE');print('ERR-LINE',file=sys.stderr)"
    res = git_sync.run([sys.executable, "-c", code], timeout_s=30, label="selftest-io")
    assert res["exit_code"] == 0
    assert "OUT-LINE" in res["stdout"] and "ERR-LINE" in res["stderr"]


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), "abc", None])
def test_run_rejects_invalid_timeout(bad):
    """P1-A: non-finite / non-positive deadlines must be refused up front."""
    with pytest.raises(git_sync.SyncError):
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=bad, label="bad-timeout")


def test_run_timeout_kills_only_its_own_group_and_others_survive():
    """P1-A: the caller and an unrelated sibling must both outlive the kill."""
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        res = git_sync.run([sys.executable, "-c", "import time; time.sleep(60)"],
                           timeout_s=3, label="selftest-timeout")
        assert res["timed_out"] is True
        assert res["kill_attempted"] is True
        assert res["kill_ok"] is True, "owned group must be fully terminated"
        assert sibling.poll() is None, "unrelated sibling must survive"
        assert os.getpid() > 0  # if our own group had been killed we would be gone
    finally:
        git_sync.terminate_owned_group(sibling, None)


def test_run_establishes_its_own_session():
    """Guard the mechanism, not just the observed outcome."""
    import inspect
    src = inspect.getsource(git_sync.run)
    if os.name == "nt":
        assert "CREATE_NEW_PROCESS_GROUP" in src
    else:
        assert "start_new_session" in src


# --------------------------------------------------------------- progress


def test_progress_trail_is_append_only_and_not_a_heartbeat():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "progress.jsonl")
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=30,
                     label="selftest-progress", progress_path=path)
        lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        assert lines and lines[-1]["phase"] == "done"
        assert lines[-1]["exit_code"] == 0
        assert all("command_id" in r and "elapsed_s" in r for r in lines)


def test_progress_interval_is_within_required_bounds():
    assert 15 <= git_sync.PROGRESS_INTERVAL_S <= 30


def test_timeout_records_phase_and_nonzero_exit():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        res = git_sync.run([sys.executable, "-c", "import time; time.sleep(30)"],
                           timeout_s=2, label="selftest-timeout-regression",
                           progress_path=path)
        assert res["timed_out"] is True
        assert res["exit_code"] != 0
        phases = [json.loads(x)["phase"] for x in open(path, encoding="utf-8")
                  if x.strip()]
        assert phases[-1] == "timed_out"


def test_progress_writer_creates_missing_directories():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "deeper", "progress.jsonl")
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=30,
                     label="selftest-progress-dir", progress_path=path)
        assert os.path.exists(path)


# --------------------------------------------------------------- safety


def test_dry_run_push_is_non_destructive_and_offline():
    out = git_sync.do_push(".", "some-branch", dry_run=True)
    assert "--dry-run" in out["command"]
    assert "push" in out["command"]
    assert "--force" not in out["command"]


def test_constructed_commands_contain_no_destructive_verbs():
    cmds = [git_sync.build_git_command(["push", "origin", "HEAD"]),
            git_sync.build_git_command(["status"]),
            git_sync.build_git_command(["ls-remote", "origin"])]
    joined = " ".join(" ".join(c) for c in cmds)
    for bad in ("--force", "reset", "rebase", "clean"):
        assert bad not in joined


def test_module_code_never_requests_force_flags():
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


def test_wrapper_never_uses_a_shell():
    import inspect
    assert "shell=True" not in inspect.getsource(git_sync)


def test_network_helpers_route_through_run():
    """P1-C: no bare subprocess.run for git/gh outside run/terminate."""
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
    assert not offenders, "these functions bypass the wrapper: %r" % offenders


# --------------------------------------------------------------- push target


def _patch_push(monkeypatch, push_exit=0, local=SHA_A, remote=SHA_A, guard_ok=True):
    calls = []

    def fake_run(cmd, timeout_s, label, progress_path=None, cwd=None, proxy=None):
        calls.append({"label": label, "argv": list(cmd), "cwd": cwd})
        return {"command_id": label, "label": label, "argv": list(cmd), "cwd": cwd,
                "exit_code": push_exit, "timed_out": False, "duration_s": 0.1,
                "stdout": "", "stderr": "", "kill_attempted": False,
                "kill_ok": True, "survivors": []}

    monkeypatch.setattr(git_sync, "run", fake_run)
    monkeypatch.setattr(git_sync, "resolve_repo", lambda *a, **k: "R")
    monkeypatch.setattr(git_sync, "local_head", lambda *a, **k: local)
    monkeypatch.setattr(git_sync, "remote_slug",
                        lambda *a, **k: git_sync.EXPECTED_REPO_SLUG)
    monkeypatch.setattr(git_sync, "remote_branch_sha",
                        lambda *a, **k: {"sha": remote, "exit_code": 0,
                                         "timed_out": False, "error": ""})
    monkeypatch.setattr(git_sync, "validate_push_target",
                        lambda *a, **k: {"ok": guard_ok,
                                         "problems": [] if guard_ok else ["blocked"],
                                         "remote_slug": git_sync.EXPECTED_REPO_SLUG,
                                         "head": local, "branch": "b"})
    return calls


def test_push_without_pr_is_never_reported_as_three_way(monkeypatch):
    _patch_push(monkeypatch)
    out = git_sync.do_push("R", "b")
    assert out["pr_sync"] == "PR_NOT_CHECKED"
    assert out["pr_head_sha"] is None
    assert out["three_way_equal"] is None
    assert out["git_sync_ok"] is True


def test_push_command_error_with_equal_sha_is_postcondition_only(monkeypatch):
    _patch_push(monkeypatch, push_exit=1, local=SHA_A, remote=SHA_A)
    out = git_sync.do_push("R", "b")
    assert out["git_sync_ok"] is True
    assert out["postcondition_only"] is True
    assert out["three_way_equal"] is None
    assert out["exit_code"] == 1


def test_push_blocked_before_any_remote_write(monkeypatch):
    calls = _patch_push(monkeypatch, guard_ok=False)
    out = git_sync.do_push("R", "b")
    assert out["blocked_before_write"] is True
    assert not any(c["label"] == "git-push" for c in calls)


def test_validate_push_target_rejection_paths(monkeypatch):
    monkeypatch.setattr(git_sync, "remote_slug", lambda *a, **k: "someone/else")
    monkeypatch.setattr(git_sync, "local_head", lambda *a, **k: SHA_A)

    bad_ref = git_sync.validate_push_target("R", "origin", "main", SHA_A)
    assert not bad_ref["ok"]
    assert any("allowed prefix" in p for p in bad_ref["problems"])
    assert any("expected" in p for p in bad_ref["problems"])

    bad_sha = git_sync.validate_push_target("R", "origin", "codex/x", SHA_B)
    assert any("does not match" in p for p in bad_sha["problems"])


# --------------------------------------------------------------- gh helpers


def test_remote_branch_sha_uses_verified_helper_and_repo(monkeypatch):
    seen = {}

    def fake_run(cmd, timeout_s, label, progress_path=None, cwd=None, proxy=None):
        seen["argv"] = list(cmd)
        return {"exit_code": 0, "timed_out": False,
                "stdout": SHA_A + "\trefs/heads/b\n", "stderr": "", "kill_ok": True}

    monkeypatch.setattr(git_sync, "run", fake_run)
    info = git_sync.remote_branch_sha("R", "origin", "b")
    assert info["sha"] == SHA_A
    assert git_sync.HELPER_RESET[1] in seen["argv"]
    assert any("gh auth git-credential" in p for p in seen["argv"])
    assert "-C" in seen["argv"]


def test_pr_view_reports_error_instead_of_empty(monkeypatch):
    monkeypatch.setattr(git_sync, "run", lambda *a, **k: {
        "exit_code": 1, "timed_out": False, "stdout": "", "stderr": "auth required",
        "kill_ok": True})
    out = git_sync.pr_view("R", 26)
    assert out["ok"] is False
    assert "auth required" in out["error"]


def test_pr_list_api_error_is_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(git_sync, "run", lambda *a, **k: {
        "exit_code": 1, "timed_out": False, "stdout": "", "stderr": "rate limited",
        "kill_ok": True})
    out = git_sync.pr_list_for_branch("R", "b")
    assert out["ok"] is False
    assert out["error"]


# --------------------------------------------------------------- publish


def _fake_publish_env(monkeypatch, prs, pr_meta, push_exit=0, pr_list_ok=True):
    calls = []
    monkeypatch.setattr(git_sync, "do_push", lambda *a, **k: {
        "exit_code": push_exit, "timed_out": False, "git_sync_ok": True,
        "local_head": SHA_A, "remote_branch_sha": SHA_A, "resolved_repo": "R",
        "duration_s": 0.1, "pr_sync": "PR_NOT_CHECKED", "three_way_equal": None})
    monkeypatch.setattr(git_sync, "pr_list_for_branch", lambda *a, **k: prs)

    def fake_pr_view(repo, pr, *a, **k):
        calls.append({"pr_view": pr})
        return pr_meta

    monkeypatch.setattr(git_sync, "pr_view", fake_pr_view)

    def fake_run(cmd, *a, **k):
        calls.append({"argv": list(cmd)})
        return {"exit_code": 0, "timed_out": False, "stdout": "https://x/26",
                "stderr": "", "kill_ok": True}

    monkeypatch.setattr(git_sync, "run", fake_run)
    calls.append({"pr_list_ok": pr_list_ok})
    return calls


def test_publish_refuses_closed_existing_pr(monkeypatch):
    prs = {"ok": True, "prs": [{"number": 9, "state": "CLOSED", "isDraft": True,
                               "baseRefName": "main", "url": "u"}]}
    _fake_publish_env(monkeypatch, prs, {"ok": True, "headRefOid": SHA_A,
                                         "state": "OPEN", "isDraft": True,
                                         "baseRefName": "main"})
    out = git_sync.publish("R", "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "EXISTING_PR_INCOMPATIBLE"


def test_publish_does_not_create_pr_when_listing_errors(monkeypatch):
    calls = _fake_publish_env(monkeypatch,
                              {"ok": False, "error": "rate limited", "prs": []},
                              {"ok": True, "headRefOid": SHA_A},
                              pr_list_ok=False)
    out = git_sync.publish("R", "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "PR_LIST_API_ERROR"
    created = [c for c in calls if c.get("argv", [None, None])[1:2] == ["pr"]
               and "create" in c.get("argv", [])]
    assert not created, "an API error must never be turned into a new PR"


def test_publish_fails_when_pr_head_does_not_match(monkeypatch):
    prs = {"ok": True, "prs": [{"number": 26, "state": "OPEN", "isDraft": True,
                               "baseRefName": "main", "url": "u"}]}
    _fake_publish_env(monkeypatch, prs, {"ok": True, "headRefOid": SHA_B,
                                         "state": "OPEN", "isDraft": True,
                                         "baseRefName": "main"})
    out = git_sync.publish("R", "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "PR_VERIFY_FAILED"
    assert any("pr head" in p for p in out["problems"])


def test_publish_rejects_non_draft_existing_pr(monkeypatch):
    prs = {"ok": True, "prs": [{"number": 26, "state": "OPEN", "isDraft": False,
                               "baseRefName": "main", "url": "u"}]}
    _fake_publish_env(monkeypatch, prs, {"ok": True, "headRefOid": SHA_A})
    out = git_sync.publish("R", "codex/b", "t", "f", "main")
    assert out["ok"] is False
    assert out["status"] == "EXISTING_PR_INCOMPATIBLE"


def test_publish_succeeds_only_with_verified_draft_pr(monkeypatch):
    prs = {"ok": True, "prs": [{"number": 26, "state": "OPEN", "isDraft": True,
                               "baseRefName": "main", "url": "u"}]}
    _fake_publish_env(monkeypatch, prs, {"ok": True, "headRefOid": SHA_A,
                                         "state": "OPEN", "isDraft": True,
                                         "baseRefName": "main"})
    out = git_sync.publish("R", "codex/b", "t", "f", "main")
    assert out["ok"] is True
    assert out["status"] == "PR_SYNC_PASS"
    assert out["stage"] == "pr_reused"
    assert out["three_way_equal"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
