"""The tmux backend's opt-in to the parametrized conformance suite.

Drives the real :class:`~omnigent.inner.terminal.TmuxBackend` — the one that
shells out to ``tmux`` — against the scripted fake ``tmux`` binary in
:mod:`tests.inner._fake_tmux`, so the full backend contract is exercised in CI
with no real tmux installed. This is the reference example for how a future
backend (e.g. herdr) opts in: one :class:`ConformanceAdapter` plus one
:class:`BackendConformanceSuite` subclass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnigent._platform import IS_WINDOWS
from omnigent.inner.terminal import TerminalBackend, TerminalLaunchRequest, TmuxBackend
from tests.inner._fake_tmux import (
    EXIT_SENTINEL,
    FINAL_SCREEN_MARKER,
    SUBMIT_SENTINEL,
    install_fake_tmux,
    key_marker,
)
from tests.inner.terminal_backend_conformance import BackendConformanceSuite, ConformanceAdapter


class TmuxConformanceAdapter(ConformanceAdapter):
    """Runs the conformance suite against ``TmuxBackend`` + the fake ``tmux``."""

    backend_cls = TmuxBackend
    submit_sentinel = SUBMIT_SENTINEL
    final_screen_marker = FINAL_SCREEN_MARKER
    preserves_dead_endpoint = True  # tmux remain-on-exit keeps the dead pane

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        """Put the fake ``tmux`` first on PATH and bind a backend to a socket.

        The fake dir is prepended to ``os.environ`` PATH (used by the backend's
        liveness / send / capture subprocesses, which pass no ``env``) so it
        wins over any real tmux on the host.
        """
        bindir = tmp_path / "fakebin"
        install_fake_tmux(bindir)
        existing = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{existing}")
        return TmuxBackend(socket_path=tmp_path / "tmux.sock", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        """Build a launch request; the fake ignores cwd but honors the command.

        ``env`` is a copy of the current environment, whose PATH already carries
        the fake dir (set in :meth:`make_backend`) — the launch subprocess is
        the one call that resolves ``tmux`` against the request's ``env``.
        """
        return TerminalLaunchRequest(
            command=command,
            cwd=os.getcwd(),
            env=dict(os.environ),
            keep_alive_after_exit=keep_alive_after_exit,
        )

    def alive_command(self) -> list[str]:
        """A non-sentinel command the fake treats as a live inner process."""
        return ["sleep", "1000000"]

    def exiting_command(self) -> list[str]:
        """The sentinel command the fake treats as an immediate inner exit."""
        return [EXIT_SENTINEL]

    def key_marker(self, key: str) -> str:
        """The fake's observable pane token for a named key."""
        return key_marker(key)

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Empty PATH so the backend cannot find ``tmux`` → the probe can't run.

        The tmux liveness probes spawn ``tmux`` without passing ``env``, so they
        resolve against ``os.environ`` PATH; pointing it at a tmux-free dir makes
        ``create_subprocess_exec`` raise ``FileNotFoundError`` (an ``OSError``),
        which the backend maps to ``UNKNOWN``.
        """
        empty = tmp_path / "emptybin"
        empty.mkdir(exist_ok=True)
        monkeypatch.setenv("PATH", str(empty))


@pytest.mark.skipif(
    IS_WINDOWS,
    reason=(
        "The tmux backend is POSIX-only (TmuxBackend.platforms == {'posix'}) and its "
        "scripted fake is a bare-name shell shim; Windows CreateProcess cannot dispatch "
        "a bare-name script, so this suite runs on POSIX CI only."
    ),
)
class TestTmuxConformance(BackendConformanceSuite):
    """The tmux backend must pass the full backend conformance suite."""

    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        """Provide the tmux adapter to the inherited conformance tests."""
        return TmuxConformanceAdapter()
