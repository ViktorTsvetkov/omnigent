"""Shared terminal-backend conformance suite for native Windows ConPTY."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from omnigent.inner.conpty_backend import ConptyBackend
from omnigent.inner.terminal import TerminalBackend, TerminalLaunchRequest
from tests.inner.terminal_backend_conformance import (
    BackendConformanceSuite,
    ConformanceAdapter,
)

_CHILD = Path(__file__).with_name("_conpty_conformance_child.py")


class ConptyConformanceAdapter(ConformanceAdapter):
    backend_cls = ConptyBackend
    submit_sentinel = "CONPTY_SUBMITTED"
    final_screen_marker = "CONPTY_FINAL_SCREEN"
    sample_keys = ("Tab", "BSpace")

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        del monkeypatch
        return ConptyBackend(socket_path=tmp_path / "conpty", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        return TerminalLaunchRequest(
            command=command,
            cwd=str(_CHILD.parent),
            env=dict(__import__("os").environ),
            size=(120, 40),
            keep_alive_after_exit=keep_alive_after_exit,
        )

    def alive_command(self) -> list[str]:
        return [sys.executable, str(_CHILD)]

    def exiting_command(self) -> list[str]:
        return [sys.executable, str(_CHILD), "--exit"]

    def key_marker(self, key: str) -> str:
        return f"<KEY:{key}>"

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        del monkeypatch, tmp_path
        backend._reader_error = OSError("probe unavailable")  # type: ignore[attr-defined]


@pytest.mark.windows_only
class TestConptyConformance(BackendConformanceSuite):
    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        return ConptyConformanceAdapter()
