from __future__ import annotations

import asyncio
from pathlib import Path

import click
import pytest

from omnigent import claude_native, opencode_native
from omnigent.opencode_native import PreparedOpenCodeTerminal


def _prepared(socket: Path | None) -> PreparedOpenCodeTerminal:
    return PreparedOpenCodeTerminal(
        session_id="conv_opencode",
        terminal_id="terminal_opencode_main",
        tmux_socket=socket,
        tmux_target="main" if socket is not None else None,
        reattached=False,
    )


def test_windows_attach_uses_websocket_when_tmux_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.setattr(opencode_native, "IS_WINDOWS", True)

    async def fake_attach(url: str, *, headers: dict[str, str]) -> bool:
        calls.append((url, headers))
        return False

    monkeypatch.setattr(claude_native, "attach_local_terminal", fake_attach)

    asyncio.run(
        opencode_native._attach_terminal_resource(
            base_url="http://127.0.0.1:8000",
            headers={"Authorization": "Bearer token"},
            prepared=_prepared(None),
        )
    )

    assert calls == [
        (
            "ws://127.0.0.1:8000/v1/sessions/conv_opencode/resources/terminals/"
            "terminal_opencode_main/attach",
            {"Authorization": "Bearer token"},
        )
    ]


def test_windows_attach_prefers_reachable_tmux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    socket_path = tmp_path / "tmux.sock"
    socket_path.touch()
    tmux_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(opencode_native, "IS_WINDOWS", True)
    monkeypatch.setattr(opencode_native.shutil, "which", lambda _name: "tmux")

    async def fake_tmux(path: Path, target: str) -> None:
        tmux_calls.append((path, target))

    async def fail_websocket(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("reachable tmux must remain the preferred transport")

    monkeypatch.setattr(opencode_native, "_attach_direct_tmux", fake_tmux)
    monkeypatch.setattr(claude_native, "attach_local_terminal", fail_websocket)

    asyncio.run(
        opencode_native._attach_terminal_resource(
            base_url="http://127.0.0.1:8000",
            headers={},
            prepared=_prepared(socket_path),
        )
    )

    assert tmux_calls == [(socket_path, "main")]


def test_posix_missing_tmux_socket_keeps_existing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(opencode_native, "IS_WINDOWS", False)

    with pytest.raises(click.ClickException, match="requires direct tmux attach"):
        asyncio.run(
            opencode_native._attach_terminal_resource(
                base_url="http://127.0.0.1:8000",
                headers={},
                prepared=_prepared(None),
            )
        )
