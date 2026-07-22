"""Windows-native codex hosting tests (#13).

Covers the CLI-side pieces that let ``omnigent codex`` run on native Windows by
hosting the codex CLI in a herdr pane instead of tmux:

* the rejection → backend-availability check
  (:func:`omnigent.cli_common.ensure_native_terminal_backend_available`), which
  replaces the hard Windows rejection for the hosting-only codex harness; and
* the local-attach herdr-UI guidance
  (:func:`omnigent.codex_native._print_herdr_local_attach_guidance`), the
  Windows degrade for the POSIX-only local PTY attach.

Both are driven against the deterministic fake herdr in
:mod:`tests.inner._fake_herdr` (via :envvar:`OMNIGENT_HERDR_BIN`) or with no
binary at all — no real herdr session is touched. The platform is simulated with
``monkeypatch`` so the tests run on any host.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import pytest

import omnigent.cli_common as cli_common_mod
import omnigent.codex_native as codex_native
import omnigent.inner.terminal as terminal_mod
from omnigent.codex_native import (
    PreparedCodexTerminal,
    _preflight_local_tools,
    _print_herdr_local_attach_guidance,
)
from omnigent.inner.terminal import HerdrBackend
from tests.inner import _fake_herdr

_FAKE_PATH = Path(_fake_herdr.__file__).resolve()


def _force_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate native Windows for both the CLI gate and backend selection."""
    monkeypatch.setattr(cli_common_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(terminal_mod, "IS_WINDOWS", True)


def _install_fake_herdr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the herdr backend at the stdlib fake (protocol 16 → available)."""
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, json.dumps([sys.executable, str(_FAKE_PATH)]))
    monkeypatch.setenv(_fake_herdr.STATE_DIR_ENV_VAR, str(tmp_path / "herdr-state"))
    monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, _fake_herdr.DEFAULT_PROTOCOL)


# ---------------------------------------------------------------------------
# Availability check (replaces the hard Windows rejection)
# ---------------------------------------------------------------------------


def test_availability_check_passes_when_backend_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On Windows with a usable herdr backend, the codex gate proceeds (no raise)."""
    _force_windows(monkeypatch)
    _install_fake_herdr(monkeypatch, tmp_path)
    # Must not raise — codex can be hosted in a herdr pane.
    cli_common_mod.ensure_native_terminal_backend_available("codex")


def test_availability_check_uses_conpty_when_herdr_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing/unspawnable herdr backend fails with a clear, actionable error."""
    _force_windows(monkeypatch)
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, str(tmp_path / "no-such-herdr"))
    cli_common_mod.ensure_native_terminal_backend_available("codex")


def test_availability_check_is_noop_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On POSIX the gate is a no-op even with an unusable herdr bin (byte-identical)."""
    monkeypatch.setattr(cli_common_mod, "IS_WINDOWS", False)
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, str(tmp_path / "no-such-herdr"))
    # Returns without raising — the POSIX path never gained an early gate.
    cli_common_mod.ensure_native_terminal_backend_available("codex")


# ---------------------------------------------------------------------------
# Local-tools preflight (the run_codex_native entry gate)
# ---------------------------------------------------------------------------


def test_preflight_is_noop_on_windows_without_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows the codex preflight skips the tmux requirement (herdr hosts codex).

    This is the entry-path gate the live run first hit: ``_preflight_local_tools``
    runs at the top of ``run_codex_native``, before any of the Windows attach /
    availability branches, so it must not reject a Windows host for lacking tmux.
    """
    monkeypatch.setattr(codex_native, "IS_WINDOWS", True)
    monkeypatch.setattr("omnigent.codex_native.shutil.which", lambda _name: None)
    # Must not raise even though tmux is absent.
    _preflight_local_tools()


def test_preflight_still_requires_tmux_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """On POSIX the tmux requirement is byte-identical (still raises without tmux)."""
    monkeypatch.setattr(codex_native, "IS_WINDOWS", False)
    monkeypatch.setattr("omnigent.codex_native.shutil.which", lambda _name: None)
    with pytest.raises(click.ClickException) as excinfo:
        _preflight_local_tools()
    assert "tmux" in str(excinfo.value).lower()


def test_preflight_passes_on_posix_with_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    """On POSIX with tmux present the preflight passes (unchanged happy path)."""
    monkeypatch.setattr(codex_native, "IS_WINDOWS", False)
    monkeypatch.setattr("omnigent.codex_native.shutil.which", lambda _name: "/usr/bin/tmux")
    _preflight_local_tools()


# ---------------------------------------------------------------------------
# Local-attach herdr-UI guidance (Windows degrade for the POSIX PTY attach)
# ---------------------------------------------------------------------------


def _prepared(tmux_socket: Path | None) -> PreparedCodexTerminal:
    """Build a minimal prepared-terminal handle for the guidance printer."""
    return PreparedCodexTerminal(
        session_id="conv_abc123",
        terminal_id="terminal_codex_main",
        tmux_socket=tmux_socket,
        tmux_target="main",
        bridge_dir=Path("unused"),
        thread_id="thread_1",
        app_server_url="ws://127.0.0.1:9",
        app_server=None,
        event_client=None,
        reattached=False,
    )


def test_local_attach_guidance_names_conpty_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Windows guidance points at the herdr app and the exact workspace label."""
    socket_path = tmp_path / "tmux.sock"
    _print_herdr_local_attach_guidance(_prepared(socket_path))
    err = capsys.readouterr().err
    assert "ConPTY" in err


def test_local_attach_guidance_degrades_without_a_socket(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no socket to derive a label from, guidance still names the prefix."""
    _print_herdr_local_attach_guidance(_prepared(None))
    err = capsys.readouterr().err
    assert "ConPTY" in err
    assert "Web UI" in err


def test_windows_routes_through_websocket_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(codex_native, "IS_WINDOWS", True)
    monkeypatch.setattr(codex_native.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(codex_native.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(codex_native, "_direct_tmux_unavailable_reason", lambda _p: "conpty")

    async def fake_attach(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(codex_native, "_attach_with_reconnect", fake_attach)

    asyncio.run(
        codex_native._attach_terminal_resource(
            base_url="http://127.0.0.1:8000",
            headers={},
            prepared=_prepared(None),
            recover=None,
        )
    )

    assert len(calls) == 1
    assert calls[0]["attach"] is codex_native.attach_local_terminal
