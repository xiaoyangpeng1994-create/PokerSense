#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused, offline tests for tools/git_sync.py.

No network and no repository access: everything here is either a pure function,
an environment check, or a subprocess of the local Python interpreter.
"""

import ast
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

import git_sync  # noqa: E402


# --------------------------------------------------------------- command shape


def test_helper_list_is_reset_before_gh_helper():
    """The whole point: `-c` appends, so the reset must come first.

    Without the empty `credential.helper=` the pre-existing Git Credential
    Manager helper still runs first and blocks the push.
    """
    cmd = git_sync.build_git_command(["push", "origin", "HEAD"])
    reset_at = cmd.index("credential.helper=")
    gh_at = cmd.index("credential.helper=!gh auth git-credential")
    assert reset_at < gh_at
    assert cmd[0] == "git"
    assert cmd[-3:] == ["push", "origin", "HEAD"]


def test_reset_only_when_gh_helper_disabled():
    cmd = git_sync.build_git_command(["status"], use_gh_helper=False)
    assert "credential.helper=" in cmd
    assert not any("gh auth git-credential" in part for part in cmd)


def test_child_env_disables_interactive_prompts():
    env = git_sync.child_env()
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GCM_INTERACTIVE"] == "never"
    assert env.get("HTTPS_PROXY"), "proxy must be present for github.com on this host"


# --------------------------------------------------------------- sha verdicts


def test_three_way_equal_requires_identical_non_empty():
    sha = "a" * 40
    assert git_sync.three_way_equal(sha, sha, sha)
    assert not git_sync.three_way_equal(sha, sha, "b" * 40)
    assert not git_sync.three_way_equal(sha, "", sha)
    assert not git_sync.three_way_equal("", "", "")


# --------------------------------------------------------------- real exit code


def test_run_captures_real_exit_code_not_pipeline_status():
    """A pipeline would report the last element's status; this must not."""
    res = git_sync.run([sys.executable, "-c", "import sys; sys.exit(7)"],
                       timeout_s=30, label="selftest-exit")
    assert res["exit_code"] == 7
    assert res["timed_out"] is False


def test_run_captures_stdout_and_stderr():
    code = ("import sys;print('OUT-LINE');print('ERR-LINE',file=sys.stderr)")
    res = git_sync.run([sys.executable, "-c", code], timeout_s=30, label="selftest-io")
    assert res["exit_code"] == 0
    assert "OUT-LINE" in res["stdout"]
    assert "ERR-LINE" in res["stderr"]


def test_run_times_out_and_kills_child_tree():
    res = git_sync.run([sys.executable, "-c", "import time; time.sleep(30)"],
                       timeout_s=2, label="selftest-timeout")
    assert res["timed_out"] is True
    assert res["duration_s"] < 20


# --------------------------------------------------------------- progress trail


def test_progress_trail_is_append_only_and_not_a_heartbeat():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "progress.jsonl")
        git_sync.run([sys.executable, "-c", "pass"], timeout_s=30,
                     label="selftest-progress", progress_path=path)
        lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        assert lines, "a finished command must leave a trail"
        assert lines[-1]["phase"] == "done"
        assert lines[-1]["exit_code"] == 0
        assert all("command_id" in rec and "elapsed_s" in rec for rec in lines)


def test_progress_written_regularly_while_running():
    """A hang must be visible as a trail, so the interval must be honoured."""
    assert git_sync.PROGRESS_INTERVAL_S <= 30, "issue requires 15-30s progress"
    assert git_sync.PROGRESS_INTERVAL_S >= 15


def test_constructed_commands_contain_no_destructive_verbs():
    """Guard on the commands the wrapper can actually build.

    The module docstring legitimately names the verbs it refuses to use, so this
    checks the *built argv* rather than the prose.
    """
    cmds = [
        git_sync.build_git_command(["push", "origin", "HEAD"]),
        git_sync.build_git_command(["status"]),
        git_sync.build_git_command(["ls-remote", "origin"]),
    ]
    joined = " ".join(" ".join(c) for c in cmds)
    for bad in ("--force", "reset", "rebase", "clean"):
        assert bad not in joined, "destructive verb in built command: %s" % bad


def test_module_code_never_requests_force_flags():
    """No string literal used as a git argument may be a force flag."""
    tree = ast.parse(open(git_sync.__file__, encoding="utf-8").read())
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                doc_ids.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in doc_ids):
            assert "--force" not in node.value


def test_timeout_records_phase_and_nonzero_exit():
    """Regression: a timed-out command must be distinguishable from a clean run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.jsonl")
        res = git_sync.run([sys.executable, "-c", "import time; time.sleep(30)"],
                           timeout_s=2, label="selftest-timeout-regression",
                           progress_path=path)
        assert res["timed_out"] is True
        assert res["exit_code"] != 0
        phases = [json.loads(x)["phase"] for x in open(path, encoding="utf-8")
                  if x.strip()]
        assert "timed_out" in phases
        assert phases[-1] == "timed_out"


def test_dry_run_push_is_non_destructive_and_offline():
    """`--dry-run` must be the default way to inspect a push without writing."""
    out = git_sync.do_push(".", "some-branch", dry_run=True)
    cmd = out["command"]
    assert "--dry-run" in cmd
    assert "push" in cmd
    assert "--force" not in cmd


def test_wrapper_never_uses_a_shell():
    """No shell=True anywhere: argv is passed directly so quoting cannot drift."""
    import inspect
    src = inspect.getsource(git_sync)
    assert "shell=True" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
