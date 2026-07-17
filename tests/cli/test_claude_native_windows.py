"""Windows-native Claude hosting tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import pytest

import omnigent.claude_native as claude_native
import omnigent.cli as cli_mod
import omnigent.inner.terminal as terminal_mod
from omnigent.claude_native import PreparedClaudeTerminal
from omnigent.inner.terminal import HerdrBackend
from tests.inner import _fake_herdr

_FAKE_PATH = Path(_fake_herdr.__file__).resolve()


def _prepared(socket_path: Path | None) -> PreparedClaudeTerminal:
    return PreparedClaudeTerminal(
        session_id="conv_abc123",
        terminal_id="terminal_claude_main",
        bridge_dir=Path("unused"),
        reattached=False,
        tmux_socket=socket_path,
        tmux_target="main",
    )


def test_availability_gate_accepts_herdr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(terminal_mod, "IS_WINDOWS", True)
    monkeypatch.setenv(
        HerdrBackend.BIN_ENV_VAR, json.dumps([sys.executable, str(_FAKE_PATH)])
    )
    monkeypatch.setenv(_fake_herdr.STATE_DIR_ENV_VAR, str(tmp_path / "herdr-state"))
    monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, _fake_herdr.DEFAULT_PROTOCOL)

    cli_mod._ensure_native_terminal_backend_available("claude")


def test_availability_gate_reports_missing_herdr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(terminal_mod, "IS_WINDOWS", True)
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, str(tmp_path / "missing-herdr"))

    with pytest.raises(click.ClickException, match=r"claude.*herdr"):
        cli_mod._ensure_native_terminal_backend_available("claude")


def test_preflight_skips_only_tmux_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_native, "IS_WINDOWS", True)
    monkeypatch.setattr(
        claude_native.shutil,
        "which",
        lambda command: "C:/bin/claude.exe" if command == "claude" else None,
    )

    claude_native._preflight_local_tools("claude")


def test_preflight_still_requires_tmux_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_native, "IS_WINDOWS", False)
    monkeypatch.setattr(
        claude_native.shutil,
        "which",
        lambda command: "/usr/bin/claude" if command == "claude" else None,
    )

    with pytest.raises(click.ClickException, match="tmux"):
        claude_native._preflight_local_tools("claude")


def test_guidance_names_herdr_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    socket_path = tmp_path / "tmux.sock"

    claude_native._print_herdr_local_attach_guidance(_prepared(socket_path))

    err = capsys.readouterr().err
    assert "herdr" in err.lower()
    assert HerdrBackend.workspace_label_for(socket_path, "main") in err
    assert "Web UI" in err


def test_windows_attach_degrades_without_pty_and_keeps_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claude_native, "IS_WINDOWS", True)

    async def unexpected_attach(**_kwargs: object) -> None:
        pytest.fail("the POSIX terminal attach must not run on Windows")

    async def unexpected_close(**_kwargs: object) -> None:
        pytest.fail("the runner-owned terminal must stay alive on Windows")

    monkeypatch.setattr(claude_native, "_attach_with_reconnect", unexpected_attach)
    monkeypatch.setattr(claude_native, "_close_claude_terminal", unexpected_close)

    outcome = asyncio.run(
        claude_native._attach_with_transcript_forwarder(
            base_url="http://127.0.0.1:8000",
            headers={},
            prepared=_prepared(None),
            agent_name="Claude Code",
            attach_url="ws://127.0.0.1:8000/attach",
            attach=unexpected_attach,
            run_transcript_forwarder=False,
        )
    )

    assert outcome is claude_native._AttachOutcome.DETACHED
