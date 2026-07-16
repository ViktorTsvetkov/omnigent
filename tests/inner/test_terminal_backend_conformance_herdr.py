"""The real :class:`HerdrBackend`'s opt-in to the shared conformance suite (#12).

Drives the CLI-shelling ``HerdrBackend`` through the same backend-agnostic
:class:`~tests.inner.terminal_backend_conformance.BackendConformanceSuite` the
tmux and in-process-fake backends run — one :class:`ConformanceAdapter` plus one
suite subclass, per that module's docstring — against the deterministic,
stdlib-only fake herdr in :mod:`tests.inner._fake_herdr` (pointed to via
:envvar:`OMNIGENT_HERDR_BIN`). No real herdr binary or session is ever touched.

Like the fake-backend opt-in and unlike the tmux one, this constructs the
backend directly (never through the platform-gated factory), so it runs and must
pass on BOTH native Windows and POSIX — there is deliberately no ``skipif``.

herdr has no ``remain-on-exit``, so it declares ``preserves_dead_endpoint =
False``: an inner exit is :attr:`Liveness.ENDPOINT_GONE`, not INNER_EXITED, and
the suite's keep-alive test asserts that verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from omnigent.inner.terminal import HerdrBackend, TerminalBackend, TerminalLaunchRequest
from tests.inner import _fake_herdr
from tests.inner._fake_herdr import EXIT_SENTINEL, SUBMIT_SENTINEL, key_marker
from tests.inner.terminal_backend_conformance import BackendConformanceSuite, ConformanceAdapter

_FAKE_PATH = Path(_fake_herdr.__file__).resolve()


class HerdrConformanceAdapter(ConformanceAdapter):
    """Runs the conformance suite against the real, CLI-shelling ``HerdrBackend``."""

    backend_cls = HerdrBackend
    submit_sentinel = SUBMIT_SENTINEL
    # herdr cannot preserve a dead endpoint (no remain-on-exit): an inner exit is
    # ENDPOINT_GONE, so the final-screen marker is never consulted.
    final_screen_marker = ""
    preserves_dead_endpoint = False

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        """Install the fake herdr CLI and return a fresh, unlaunched backend."""
        monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, json.dumps([sys.executable, str(_FAKE_PATH)]))
        monkeypatch.setenv(_fake_herdr.STATE_DIR_ENV_VAR, str(tmp_path / "herdr-state"))
        monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, _fake_herdr.DEFAULT_PROTOCOL)
        monkeypatch.setenv(_fake_herdr.VERSION_ENV_VAR, _fake_herdr.DEFAULT_VERSION)
        return HerdrBackend(socket_path=tmp_path / "herdr.sock", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        """Build a launch request against the fake."""
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
        """The fake's screen token for *key* AFTER the backend's translation.

        The backend translates a neutral key into herdr's syntax before the fake
        echoes it (e.g. ``C-c`` → ``ctrl+c`` → ``<ctrl+c>``), so the expected
        marker must be built from the translated token, not the neutral name.
        """
        translated = HerdrBackend._translate_key(key)
        assert translated is not None, f"conformance sample key {key!r} is unsupported by herdr"
        return key_marker(translated)

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Point the herdr binary at a nonexistent path so probes cannot spawn."""
        del backend, tmp_path
        monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, str(_FAKE_PATH.parent / "no-such-herdr"))


class TestHerdrConformance(BackendConformanceSuite):
    """The real herdr backend must pass the full conformance suite.

    Runs on every platform (direct construction bypasses the Windows-only
    platform gate), including POSIX CI.
    """

    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        """Provide the herdr adapter to the inherited conformance tests."""
        return HerdrConformanceAdapter()
