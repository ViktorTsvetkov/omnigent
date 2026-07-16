"""The in-process :class:`FakeBackend`'s opt-in to the conformance suite.

Drives the full-protocol, in-memory
:class:`~tests.inner.fake_terminal_backend.FakeBackend` through the same
backend-agnostic
:class:`~tests.inner.terminal_backend_conformance.BackendConformanceSuite` the
tmux backend runs — one :class:`ConformanceAdapter` plus one suite subclass, per
that module's docstring.

Unlike the tmux opt-in (which needs a POSIX shell shim on ``PATH`` and therefore
skips on Windows), the fake is pure in-process Python, so this suite runs
everywhere — giving the terminal-backend program its first *Windows-native*
conformance coverage. There is deliberately no ``skipif`` here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.inner.terminal import TerminalBackend, TerminalLaunchRequest
from tests.inner.fake_terminal_backend import (
    EXIT_SENTINEL,
    FINAL_SCREEN_MARKER,
    SUBMIT_SENTINEL,
    FakeBackend,
    key_marker,
)
from tests.inner.terminal_backend_conformance import BackendConformanceSuite, ConformanceAdapter


class FakeConformanceAdapter(ConformanceAdapter):
    """Runs the conformance suite against the in-process ``FakeBackend``."""

    backend_cls = FakeBackend
    submit_sentinel = SUBMIT_SENTINEL
    final_screen_marker = FINAL_SCREEN_MARKER
    # The in-process fake keeps the dead endpoint and its final screen (it is
    # trivial to preserve in memory), so it reports INNER_EXITED under keep-alive
    # — the same verdict tmux's remain-on-exit yields.
    preserves_dead_endpoint = True

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        """Return a fresh, unlaunched fake — no ``PATH`` or binary to install."""
        del monkeypatch
        return FakeBackend(socket_path=tmp_path / "fake.sock", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        """Build a launch request; the fake hosts the command in memory."""
        return TerminalLaunchRequest(
            command=command,
            cwd=".",
            env={},
            keep_alive_after_exit=keep_alive_after_exit,
        )

    def alive_command(self) -> list[str]:
        """A non-sentinel command the fake treats as a live inner process."""
        return ["sleep", "1000000"]

    def exiting_command(self) -> list[str]:
        """The sentinel command the fake treats as an immediate inner exit."""
        return [EXIT_SENTINEL]

    def key_marker(self, key: str) -> str:
        """The fake's observable screen token for a named key."""
        return key_marker(key)

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Make the fake's liveness probe report UNKNOWN (probe cannot run)."""
        del monkeypatch, tmp_path
        assert isinstance(backend, FakeBackend)
        backend.break_liveness_probe()


class TestFakeConformance(BackendConformanceSuite):
    """The in-process fake backend must pass the full conformance suite.

    Runs on every platform, including native Windows.
    """

    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        """Provide the fake adapter to the inherited conformance tests."""
        return FakeConformanceAdapter()
