"""Windows local-console attachment tests for native Pi."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import pytest

from omnigent import claude_native, pi_native


def _prepared(socket_path: Path | None) -> pi_native.PreparedPiTerminal:
    return pi_native.PreparedPiTerminal(
        session_id="conv_pi",
        terminal_id="terminal_pi_main",
        tmux_socket=socket_path,
        tmux_target="main" if socket_path is not None else None,
        reattached=False,
    )


def test_windows_uses_shared_websocket_attach_without_local_tmux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(pi_native, "IS_WINDOWS", True)

    async def fake_attach(url: str, *, headers: dict[str, str]) -> bool:
        calls.append((url, headers))
        return False

    monkeypatch.setattr(claude_native, "attach_local_terminal", fake_attach)

    asyncio.run(
        pi_native._attach_terminal_resource(
            base_url="http://127.0.0.1:8000",
            headers={"Authorization": "Bearer token"},
            prepared=_prepared(None),
        )
    )

    assert calls == [
        (
            "ws://127.0.0.1:8000/v1/sessions/conv_pi/resources/terminals/terminal_pi_main/attach",
            {"Authorization": "Bearer token"},
        )
    ]


def test_windows_prefers_reachable_tmux_socket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "tmux.sock"
    socket_path.touch()
    tmux_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(pi_native, "IS_WINDOWS", True)
    monkeypatch.setattr(pi_native.shutil, "which", lambda _name: "tmux")

    async def fake_tmux(path: Path, target: str) -> None:
        tmux_calls.append((path, target))

    async def fail_websocket(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("reachable tmux must remain the preferred transport")

    monkeypatch.setattr(pi_native, "_attach_direct_tmux", fake_tmux)
    monkeypatch.setattr(claude_native, "attach_local_terminal", fail_websocket)

    asyncio.run(
        pi_native._attach_terminal_resource(
            base_url="http://127.0.0.1:8000",
            headers={},
            prepared=_prepared(socket_path),
        )
    )

    assert tmux_calls == [(socket_path, "main")]


def test_windows_preflight_does_not_require_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pi_native, "IS_WINDOWS", True)
    monkeypatch.setattr(pi_native.shutil, "which", lambda _name: None)

    pi_native._preflight_local_tools()


def test_posix_missing_tmux_socket_keeps_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pi_native, "IS_WINDOWS", False)

    with pytest.raises(click.ClickException, match="requires direct tmux attach"):
        asyncio.run(
            pi_native._attach_terminal_resource(
                base_url="http://127.0.0.1:8000",
                headers={},
                prepared=_prepared(None),
            )
        )
