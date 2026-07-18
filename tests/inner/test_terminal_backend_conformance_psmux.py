"""Native-Windows psmux opt-in to the shared backend conformance suite."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from omnigent.inner.psmux_backend import PsmuxBackend
from omnigent.inner.terminal import TerminalBackend, TerminalLaunchRequest
from tests.inner import _fake_psmux
from tests.inner._fake_psmux import (
    EXIT_SENTINEL,
    FINAL_SCREEN_MARKER,
    SUBMIT_SENTINEL,
    key_marker,
)
from tests.inner.terminal_backend_conformance import BackendConformanceSuite, ConformanceAdapter

_FAKE_PATH = Path(_fake_psmux.__file__).resolve()


class PsmuxConformanceAdapter(ConformanceAdapter):
    """Drive the CLI-shelling backend through a deterministic psmux double."""

    backend_cls = PsmuxBackend
    submit_sentinel = SUBMIT_SENTINEL
    final_screen_marker = FINAL_SCREEN_MARKER
    preserves_dead_endpoint = True

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        monkeypatch.setenv(
            PsmuxBackend.BIN_ENV_VAR,
            json.dumps([sys.executable, str(_FAKE_PATH)]),
        )
        monkeypatch.setenv(_fake_psmux.STATE_DIR_ENV_VAR, str(tmp_path / "psmux-state"))
        return PsmuxBackend(socket_path=tmp_path / "psmux.sock", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        return TerminalLaunchRequest(
            command=command,
            cwd=os.getcwd(),
            env=dict(os.environ),
            size=(120, 30),
            keep_alive_after_exit=keep_alive_after_exit,
        )

    def alive_command(self) -> list[str]:
        return ["sleep", "1000000"]

    def exiting_command(self) -> list[str]:
        return [EXIT_SENTINEL]

    def key_marker(self, key: str) -> str:
        return key_marker(key)

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        del backend
        monkeypatch.setenv(PsmuxBackend.BIN_ENV_VAR, str(tmp_path / "no-such-psmux.exe"))


@pytest.mark.windows_only
class TestPsmuxConformance(BackendConformanceSuite):
    """The native psmux backend must pass the shared conformance contract."""

    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        return PsmuxConformanceAdapter()
