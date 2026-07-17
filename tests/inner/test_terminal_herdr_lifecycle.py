"""Session-lifecycle tests for :class:`~omnigent.inner.terminal.HerdrBackend` (#11).

Drives the real ``HerdrBackend`` — the one that shells out to the herdr CLI —
against the deterministic, stdlib-only fake herdr in :mod:`tests.inner._fake_herdr`,
pointed to via :envvar:`OMNIGENT_HERDR_BIN` (encoded as a ``[python, script]``
launcher so it needs no ``.cmd``/PATHEXT shim). No real herdr binary or session
is ever touched.

Coverage mirrors the lifecycle portion of
:class:`~tests.inner.terminal_backend_conformance.BackendConformanceSuite`
(launch→ALIVE, close→ENDPOINT_GONE, inner-exit→ENDPOINT_GONE, probe→UNKNOWN,
non-submitting paste, named-key delivery) plus the herdr-specific acceptance
criteria: the protocol/version gate, husk adopt/replace ordering, the
explicit-``--session`` enforcement, Windows path translation, CRLF stripping, and
geometry pinning. Opting into the shared conformance suite is deliberately left
to ticket #12; the fake is structured so that opt-in needs no rewrite.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from omnigent.inner.terminal import HerdrBackend, Liveness, TerminalLaunchRequest
from tests.inner import _fake_herdr
from tests.inner._fake_herdr import (
    EXIT_SENTINEL,
    SUBMIT_SENTINEL,
    key_marker,
    read_log,
    seed_workspace,
    state_file,
)

_FAKE_PATH = Path(_fake_herdr.__file__).resolve()


@dataclass
class _HerdrEnv:
    """Handle over an installed fake herdr: paths plus a backend factory."""

    tmp_path: Path
    state_dir: Path
    log_path: Path
    monkeypatch: pytest.MonkeyPatch

    def make_backend(self, target: str = "main", *, name: str = "herdr.sock") -> HerdrBackend:
        """Return a fresh, unlaunched backend bound to a private endpoint."""
        return HerdrBackend(socket_path=self.tmp_path / name, target=target)

    def load_state(self, backend: HerdrBackend) -> dict:
        """Return the fake's on-disk state for *backend*'s session."""
        return json.loads(state_file(self.state_dir, backend._session).read_text(encoding="utf-8"))

    def log(self) -> list[list[str]]:
        """Return every logged herdr invocation's argv, in call order."""
        return read_log(self.log_path)


@pytest.fixture
def herdr_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _HerdrEnv:
    """Install the fake herdr CLI and point the backend's env at it."""
    state_dir = tmp_path / "herdr-state"
    log_path = tmp_path / "herdr-invocations.log"
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, json.dumps([sys.executable, str(_FAKE_PATH)]))
    monkeypatch.setenv(_fake_herdr.STATE_DIR_ENV_VAR, str(state_dir))
    monkeypatch.setenv(_fake_herdr.LOG_ENV_VAR, str(log_path))
    monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, _fake_herdr.DEFAULT_PROTOCOL)
    monkeypatch.setenv(_fake_herdr.VERSION_ENV_VAR, _fake_herdr.DEFAULT_VERSION)
    return _HerdrEnv(
        tmp_path=tmp_path, state_dir=state_dir, log_path=log_path, monkeypatch=monkeypatch
    )


def _request(
    command: list[str], *, size: tuple[int, int] = (80, 24), cwd: str = "."
) -> TerminalLaunchRequest:
    """Build a launch request against the fake."""
    return TerminalLaunchRequest(command=command, cwd=cwd, env={}, size=size)


# ---------------------------------------------------------------------------
# Lifecycle: launch / liveness / close (the conformance-suite lifecycle shape)
# ---------------------------------------------------------------------------


async def test_launch_makes_endpoint_alive(herdr_env: _HerdrEnv) -> None:
    """After launch, both liveness probes report the endpoint ALIVE."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.liveness() == Liveness.ALIVE
        assert backend.liveness_sync() == Liveness.ALIVE
    finally:
        await backend.close()


async def test_close_makes_endpoint_gone(herdr_env: _HerdrEnv) -> None:
    """close() tears the workspace down; liveness then reports ENDPOINT_GONE."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    await backend.close()
    assert await backend.liveness() == Liveness.ENDPOINT_GONE
    assert backend.liveness_sync() == Liveness.ENDPOINT_GONE


async def test_inner_exit_is_endpoint_gone_no_keep_alive(herdr_env: _HerdrEnv) -> None:
    """An inner process that exits destroys the pane → ENDPOINT_GONE."""
    backend = herdr_env.make_backend()
    await backend.launch(_request([EXIT_SENTINEL]))
    try:
        assert await backend.liveness() == Liveness.ENDPOINT_GONE
    finally:
        await backend.close()


async def test_inner_exit_ignores_keep_alive(herdr_env: _HerdrEnv) -> None:
    """herdr has no remain-on-exit: keep_alive_after_exit cannot preserve a dead
    endpoint, so an inner exit is still ENDPOINT_GONE (never INNER_EXITED)."""
    backend = herdr_env.make_backend()
    request = TerminalLaunchRequest(
        command=[EXIT_SENTINEL], cwd=".", env={}, keep_alive_after_exit=True
    )
    await backend.launch(request)
    try:
        assert await backend.liveness() == Liveness.ENDPOINT_GONE
    finally:
        await backend.close()


async def test_liveness_degrades_to_unknown_when_probe_cannot_run(herdr_env: _HerdrEnv) -> None:
    """A probe that cannot spawn the CLI degrades to UNKNOWN, not ENDPOINT_GONE."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    # Point the binary at a path that cannot be spawned: the probe cannot run.
    herdr_env.monkeypatch.setenv(
        HerdrBackend.BIN_ENV_VAR, str(herdr_env.tmp_path / "no-such-herdr")
    )
    assert await backend.liveness() == Liveness.UNKNOWN
    assert backend.liveness_sync() == Liveness.UNKNOWN


# ---------------------------------------------------------------------------
# Input + capture
# ---------------------------------------------------------------------------


async def test_capture_returns_pane_content(herdr_env: _HerdrEnv) -> None:
    """A snapshot returns the pane screen, sync and async alike."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("hello herdr 123")
        assert "hello herdr 123" in await backend.capture()
        assert "hello herdr 123" in backend.capture_sync()
    finally:
        await backend.close()


async def test_send_text_multiline_is_not_submitted(herdr_env: _HerdrEnv) -> None:
    """A multi-line paste lands intact (verbatim) and is NOT submitted."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        pasted = "first line\nsecond line\nthird line"
        await backend.send_text(pasted)
        snapshot = await backend.capture()
        assert pasted in snapshot
        assert SUBMIT_SENTINEL not in snapshot
    finally:
        await backend.close()


async def test_enter_key_submits_and_chords_translate(herdr_env: _HerdrEnv) -> None:
    """Enter submits (unlike a paste); a C- chord reaches the pane as plus-notation."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("run me")
        assert SUBMIT_SENTINEL not in await backend.capture()
        await backend.send_keys(["C-c", "Enter"])
        snapshot = await backend.capture()
        assert key_marker("ctrl+c") in snapshot  # C-c → ctrl+c
        assert SUBMIT_SENTINEL in snapshot
    finally:
        await backend.close()


async def test_unsupported_keys_are_skipped(herdr_env: _HerdrEnv) -> None:
    """A key herdr cannot express is skipped rather than sent as a wrong key."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_keys(["Home", "Escape"])
        snapshot = await backend.capture()
        assert key_marker("Home") not in snapshot
        assert key_marker("Escape") in snapshot
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Version / protocol gate (acceptance criterion 2)
# ---------------------------------------------------------------------------


def test_ensure_available_passes_on_supported_protocol(herdr_env: _HerdrEnv) -> None:
    """The gate passes when the fake reports protocol >= the minimum."""
    HerdrBackend.ensure_available()  # must not raise


def test_ensure_available_refuses_old_protocol(herdr_env: _HerdrEnv) -> None:
    """An unsupported protocol is refused loudly, naming found + required."""
    herdr_env.monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, "15")
    with pytest.raises(RuntimeError) as excinfo:
        HerdrBackend.ensure_available()
    message = str(excinfo.value)
    assert "protocol" in message.lower()
    assert "15" in message  # the protocol found
    assert str(HerdrBackend.MIN_PROTOCOL) in message  # the protocol required


def test_ensure_available_refuses_missing_binary(herdr_env: _HerdrEnv) -> None:
    """A missing/unspawnable binary is refused loudly, naming the override var."""
    herdr_env.monkeypatch.setenv(
        HerdrBackend.BIN_ENV_VAR, str(herdr_env.tmp_path / "no-such-herdr")
    )
    with pytest.raises(RuntimeError) as excinfo:
        HerdrBackend.ensure_available()
    message = str(excinfo.value)
    assert "herdr" in message.lower()
    assert HerdrBackend.BIN_ENV_VAR in message


# ---------------------------------------------------------------------------
# Husk adopt/replace (acceptance criterion 3) + orphan reaping
# ---------------------------------------------------------------------------


async def test_launch_adopts_and_replaces_same_label_husk(herdr_env: _HerdrEnv) -> None:
    """A restart leftover (same-label workspace) is replaced create-before-close."""
    backend = herdr_env.make_backend()
    husk_id = seed_workspace(herdr_env.state_dir, backend._session, backend._label)

    await backend.launch(_request(["sleep", "1000000"]))
    try:
        # The fresh workspace exists and is alive.
        assert await backend.liveness() == Liveness.ALIVE
        assert backend._workspace_id is not None and backend._workspace_id != husk_id
        # The husk is gone (replaced).
        state = herdr_env.load_state(backend)
        assert husk_id not in state["workspaces"]
        assert backend._workspace_id in state["workspaces"]

        # Create-before-close: the fresh `workspace create` was logged BEFORE the
        # husk's `workspace close`.
        log = herdr_env.log()
        create_idx = next(i for i, a in enumerate(log) if a[2:4] == ["workspace", "create"])
        close_idx = next(
            i for i, a in enumerate(log) if a[2:4] == ["workspace", "close"] and husk_id in a
        )
        assert create_idx < close_idx
    finally:
        await backend.close()


async def test_close_reaps_leftover_same_label_workspaces(herdr_env: _HerdrEnv) -> None:
    """close() reaps stray same-label husks of the session, not just its own."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    # Stage a stray same-label husk that appeared after launch.
    stray_id = seed_workspace(herdr_env.state_dir, backend._session, backend._label)

    await backend.close()

    state = herdr_env.load_state(backend)
    assert stray_id not in state["workspaces"]
    assert backend._workspace_id not in state["workspaces"]


async def test_close_is_quiet_when_workspace_already_gone(herdr_env: _HerdrEnv) -> None:
    """close() on an already-reaped workspace succeeds quietly (idempotent)."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    await backend.close()
    await backend.close()  # second close must not raise


# ---------------------------------------------------------------------------
# Explicit-session enforcement (acceptance criterion 4)
# ---------------------------------------------------------------------------


async def test_every_invocation_carries_explicit_omnigent_session(herdr_env: _HerdrEnv) -> None:
    """EVERY herdr invocation leads with an explicit, omnigent-scoped --session.

    Never a bare subcommand (which would target ``default`` = the user's live
    panes) and never ``default``. Exercises a full lifecycle plus the version
    probe, then inspects the fake's invocation log.
    """
    HerdrBackend.ensure_available()  # logs the probe invocation
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    await backend.send_text("x")
    await backend.send_keys(["Enter"])
    await backend.capture()
    await backend.liveness()
    await backend.close()

    log = herdr_env.log()
    assert log, "expected at least one logged herdr invocation"
    for argv in log:
        assert "--session" in argv, f"invocation without --session: {argv}"
        session_value = argv[argv.index("--session") + 1]
        assert session_value.startswith("omnigent-"), session_value
        assert session_value != "default"


# ---------------------------------------------------------------------------
# Launch verb reconciliation + Windows path translation + CRLF (acceptance criterion 5)
# ---------------------------------------------------------------------------


async def test_launch_does_not_emit_nonexistent_geometry_flags(herdr_env: _HerdrEnv) -> None:
    """Reconciled launch never passes ``--cols``/``--rows`` (real herdr rejects them).

    Geometry pinning is dropped in #13: real herdr's ``workspace``/``tab``/``agent``
    creation verbs take no absolute geometry (``pane resize`` is relative), so a
    ``--cols``/``--rows`` at spawn would be an unknown-flag error. The request
    size is accepted but not forwarded as those flags.
    """
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"], size=(123, 45)))
    try:
        for argv in herdr_env.log():
            assert "--cols" not in argv, argv
            assert "--rows" not in argv, argv
    finally:
        await backend.close()


async def test_launch_runs_inner_command_via_agent_start(herdr_env: _HerdrEnv) -> None:
    """The inner argv is spawned with ``agent start ... -- <argv>`` (the real verb).

    ``tab create --command`` does not exist in real herdr; the reconciled launch
    uses ``agent start`` and the pane it creates hosts the inner command.
    """
    backend = herdr_env.make_backend()
    await backend.launch(_request(["my-inner-cli", "--flag"]))
    try:
        log = herdr_env.log()
        start = next(a for a in log if a[2:4] == ["agent", "start"])
        # The argv rides after the terminal ``--`` marker.
        assert start[start.index("--") + 1 :] == ["my-inner-cli", "--flag"]
        assert "--workspace" in start
        # The resolved pane hosts that command in the fake's state.
        pane = herdr_env.load_state(backend)["panes"][backend._pane_id]
        assert pane["command"] == ["my-inner-cli", "--flag"]
    finally:
        await backend.close()


async def test_launch_threads_env_as_agent_start_flags(herdr_env: _HerdrEnv) -> None:
    """Every ``request.env`` pair reaches the pane as ``--env`` before the ``--``.

    This is the codex-on-Windows dependency: the pane needs ``CODEX_HOME`` (and
    the optional Databricks pair) to find its private per-session config.
    """
    backend = herdr_env.make_backend()
    env = {"CODEX_HOME": "C:\\codex-home", "DATABRICKS_HOST": "https://x"}
    request = TerminalLaunchRequest(
        command=["codex", "--remote", "ws://127.0.0.1:9"], cwd=".", env=env
    )
    await backend.launch(request)
    try:
        # The fake recorded exactly the threaded env on the pane.
        pane = herdr_env.load_state(backend)["panes"][backend._pane_id]
        assert pane["env"] == env
        # Every ``--env`` flag precedes the ``--`` inner-argv marker.
        start = next(a for a in herdr_env.log() if a[2:4] == ["agent", "start"])
        dash = start.index("--")
        env_positions = [i for i, tok in enumerate(start) if tok == "--env"]
        assert env_positions, "expected --env flags"
        assert all(pos < dash for pos in env_positions)
    finally:
        await backend.close()


async def test_launch_starts_session_server_before_socket_verbs(herdr_env: _HerdrEnv) -> None:
    """launch() brings up the session's headless server before any socket verb.

    A named herdr session's server does not auto-start (the first socket verb
    would fail with an OS NotFound), so launch must start it via ``--session <s>
    server`` before ``workspace create``.
    """
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        log = herdr_env.log()
        assert any(a[2:3] == ["server"] for a in log), "expected a 'server' bring-up"
        server_idx = next(i for i, a in enumerate(log) if a[2:3] == ["server"])
        create_idx = next(i for i, a in enumerate(log) if a[2:4] == ["workspace", "create"])
        assert server_idx < create_idx  # server up before the first mutating verb
        assert herdr_env.load_state(backend)["server_running"] is True
    finally:
        await backend.close()


async def test_ensure_server_is_idempotent_when_already_running(herdr_env: _HerdrEnv) -> None:
    """A redundant server-ensure on a live session spawns no second server."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        before = sum(1 for a in herdr_env.log() if a[2:3] == ["server"])
        await backend._ensure_server_running()  # already up → fast-path, no spawn
        after = sum(1 for a in herdr_env.log() if a[2:3] == ["server"])
        assert after == before
    finally:
        await backend.close()


async def test_launch_fails_clearly_when_server_cannot_start(herdr_env: _HerdrEnv) -> None:
    """A session server that never comes up fails launch with an actionable error."""
    herdr_env.monkeypatch.setenv(_fake_herdr.SERVER_REFUSE_ENV_VAR, "1")
    herdr_env.monkeypatch.setattr(HerdrBackend, "_SERVER_READY_TIMEOUT_S", 0.6)
    herdr_env.monkeypatch.setattr(HerdrBackend, "_SERVER_POLL_INTERVAL_S", 0.1)
    backend = herdr_env.make_backend()
    with pytest.raises(RuntimeError) as excinfo:
        await backend.launch(_request(["sleep", "1000000"]))
    assert "server" in str(excinfo.value).lower()


async def test_launch_with_empty_env_emits_no_env_flags(herdr_env: _HerdrEnv) -> None:
    """An empty ``request.env`` threads zero ``--env`` flags (POSIX-neutral)."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        start = next(a for a in herdr_env.log() if a[2:4] == ["agent", "start"])
        assert "--env" not in start
        assert herdr_env.load_state(backend)["panes"][backend._pane_id]["env"] == {}
    finally:
        await backend.close()


async def test_launch_env_value_with_equals_survives(herdr_env: _HerdrEnv) -> None:
    """An env value containing ``=`` is delivered intact (split on first ``=``)."""
    backend = herdr_env.make_backend()
    token = "header.payload=extra=padding"
    request = TerminalLaunchRequest(
        command=["codex"], cwd=".", env={"DATABRICKS_CODEX_TOKEN": token}
    )
    await backend.launch(request)
    try:
        pane = herdr_env.load_state(backend)["panes"][backend._pane_id]
        assert pane["env"]["DATABRICKS_CODEX_TOKEN"] == token
    finally:
        await backend.close()


async def test_launch_translates_cwd_to_windows_path(herdr_env: _HerdrEnv) -> None:
    """A forward-slash cwd is translated to a Windows-native path for herdr."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"], cwd="C:/Users/dev/proj"))
    try:
        pane = next(iter(herdr_env.load_state(backend)["panes"].values()))
        assert pane["cwd"] == "C:\\Users\\dev\\proj"
    finally:
        await backend.close()


async def test_capture_strips_crlf(herdr_env: _HerdrEnv) -> None:
    """The pane read (emitted with CRLF by herdr) is normalized to LF."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("alpha\nbeta")
        snapshot = await backend.capture()
        assert "\r" not in snapshot
        assert "alpha\nbeta" in snapshot
    finally:
        await backend.close()


def test_to_windows_path_unit() -> None:
    """Windows path translation is host-independent and deterministic."""
    assert HerdrBackend._to_windows_path("C:/Users/dev/proj") == "C:\\Users\\dev\\proj"
    assert HerdrBackend._to_windows_path("sub/dir") == "sub\\dir"


def test_normalize_newlines_unit() -> None:
    """CRLF and lone CR both collapse to LF."""
    assert HerdrBackend._normalize_newlines("a\r\nb\rc\n") == "a\nb\nc\n"


def test_translate_key_unit() -> None:
    """Key translation: chords → plus-notation, unsupported → skipped."""
    assert HerdrBackend._translate_key("C-c") == "ctrl+c"
    assert HerdrBackend._translate_key("M-x") == "alt+x"
    assert HerdrBackend._translate_key("Enter") == "Enter"
    assert HerdrBackend._translate_key("Escape") == "Escape"
    assert HerdrBackend._translate_key("Home") is None


def test_session_and_label_are_omnigent_scoped(herdr_env: _HerdrEnv) -> None:
    """Derived session/label are omnigent-scoped (never collide with default)."""
    backend = herdr_env.make_backend()
    assert backend._session.startswith("omnigent-")
    assert backend._session != "default"
    assert backend._label.startswith("omnigent-")
    # Distinct endpoints derive distinct sessions.
    other = herdr_env.make_backend(name="other.sock")
    assert other._session != backend._session


def test_command_prefix_parses_json_and_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    """OMNIGENT_HERDR_BIN accepts a plain path or a JSON launcher array."""
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, "/opt/herdr")
    assert HerdrBackend._command_prefix() == ["/opt/herdr"]
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, json.dumps(["py", "fake.py"]))
    assert HerdrBackend._command_prefix() == ["py", "fake.py"]
    monkeypatch.delenv(HerdrBackend.BIN_ENV_VAR, raising=False)
    assert HerdrBackend._command_prefix() == [HerdrBackend.DEFAULT_BIN]


def test_construct_for_instance_builds_bound_backend(tmp_path: Path) -> None:
    """The construction hook returns a herdr backend bound to the endpoint."""
    backend = HerdrBackend.construct_for_instance(socket_path=tmp_path / "h.sock", target="main")
    assert isinstance(backend, HerdrBackend)
    assert backend._target == "main"
