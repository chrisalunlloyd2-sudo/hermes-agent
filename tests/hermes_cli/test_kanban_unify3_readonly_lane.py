"""UNIFY-3 dispatcher tests: read-only lane, unblocker-first ordering, legacy parity."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def kb_env(monkeypatch, tmp_path):
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    for prof in ("alpha", "beta", "default"):
        os.makedirs(test_home / "profiles" / prof, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    # The kanban board resolution honours these machine-flow pins; a test run
    # inside a dispatcher-owned workspace would otherwise read the LIVE board.
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_connect
    from hermes_cli import kanban_db_dispatch
    # Patch AFTER the purge+import: the dispatcher gate imports
    # ``hermes_cli.profiles.profile_exists`` at call time from whatever module
    # instance is current, so the patch must land on the post-purge module
    # (an autouse fixture that patches before ``kb_env`` runs is undone by the
    # purge and every spawn is bucketed ``skipped_nonspawnable``).
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    yield kanban_db, kanban_db_connect, kanban_db_dispatch


def _fake_spawn(*args, **kwargs):
    return 12345


def _run(kbd, kbc, **kwargs):
    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True, **kwargs)


def _fresh(kb):
    kb.create_board(slug="default", name="Test")


def _set_running(kb, kbc, task_id):
    # Fake a live claim: claim_expires in the future so the reclaim phase
    # (release_stale_claims) treats the row as an unexpired claim, and
    # last_heartbeat_at fresh so the alive-PID claim-extension path keeps it
    # (a bare worker_pid=999999 with no heartbeat would eventually be
    # treated as a wedged worker and requeued, changing lane maths).
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock='t:1', "
                "worker_pid=999999, "
                "last_heartbeat_at=strftime('%s','now'), "
                "claim_expires=strftime('%s','now') + 3600 "
                "WHERE id=?",
                (task_id,),
            )


def test_legacy_parity_no_ro_cap_single_pool(kb_env):
    """Lane disabled: read-only cards share the heavy pool; saturation
    early-returns exactly like pre-UNIFY-3."""
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        ro = kb.create_task(conn, title="audit", assignee="alpha", read_only=True)
        hw = kb.create_task(conn, title="write code", assignee="alpha")
    _set_running(kb, kbc, hw)
    res = _run(kbd, kbc, max_in_progress=1)  # pool full
    assert res.spawned == []  # whole tick skipped (legacy early-return)


def test_ro_lane_runs_auditors_while_heavy_saturated(kb_env):
    """Lane enabled: saturated heavy pool does NOT stop a read-only spawn."""
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        hw = kb.create_task(conn, title="write code", assignee="alpha")
        ro1 = kb.create_task(conn, title="audit1", assignee="alpha", read_only=True)
        ro2 = kb.create_task(conn, title="audit2", assignee="alpha", read_only=True)
    _set_running(kb, kbc, hw)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress=1, max_in_progress_read_only=1,
        )
    spawned_ids = [s[0] for s in res.spawned]
    assert ro1 in spawned_ids
    assert ro2 not in spawned_ids  # ro cap of 1 already consumed
    assert hw not in spawned_ids


def test_ro_cap_respected_and_heavy_blocked_when_full(kb_env):
    """Both lanes at cap → nothing new spawns; heavy cards deferred."""
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        hw = kb.create_task(conn, title="write", assignee="alpha")
        ro = kb.create_task(conn, title="audit", assignee="alpha", read_only=True)
        hw2 = kb.create_task(conn, title="write2", assignee="alpha")
        ro2 = kb.create_task(conn, title="audit2", assignee="alpha", read_only=True)
    _set_running(kb, kbc, hw)
    _set_running(kb, kbc, ro)
    res = _run(kbd, kbc, max_in_progress=1, max_in_progress_read_only=1)
    assert res.spawned == []


def test_unblocker_first_ordering(kb_env):
    """A ready card whose finish unblocks children gets spawned before an
    older, unbounded leaf card with equal priority."""
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        leaf = kb.create_task(conn, title="old leaf", assignee="alpha")
        parent = kb.create_task(conn, title="parent of two", assignee="alpha")
        c1 = kb.create_task(conn, title="child1", assignee="alpha", parents=[parent])
        c2 = kb.create_task(conn, title="child2", assignee="alpha", parents=[parent])
    # leaf is older (created first) but unblocks nobody; parent unblocks 2 children
    res = _run(kbd, kbc, max_spawn=1, max_in_progress=1)
    # parent must come first despite being created later
    assert res.spawned and res.spawned[0][0] == parent


def test_ro_counted_as_single_pool_when_lane_disabled(kb_env):
    """Lane disabled + pool has headroom: a read-only card spawns (counts as heavy),
    but draws only ONE slot: remaining budget honored."""
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        ro = kb.create_task(conn, title="audit", assignee="alpha", read_only=True)
    res = _run(kbd, kbc, max_in_progress=2)
    assert [s[0] for s in res.spawned] == [ro]


def test_lane_flags_persist_roundtrip(kb_env):
    kb, kbc, kbd = kb_env
    _fresh(kb)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="auditor", assignee="alpha", read_only=True)
        task = kb.get_task(conn, tid)
    assert task.read_only is True
    # legacy creation path (no flag) stays heavy
    with kbc.connect_closing() as conn:
        t2 = kb.get_task(conn, kb.create_task(conn, title="builder", assignee="alpha"))
    assert t2.read_only is False
