"""Tests for the backend-neutral snapshot-polling WebSocket bridge."""

from __future__ import annotations

import pytest

from omnigent.terminals.snapshot_bridge import (
    _forward_snapshots_to_ws,
    _send_browser_input,
)


class _Backend:
    def __init__(self) -> None:
        self.snapshots = ["first", "first", "second"]
        self.text: list[str] = []
        self.keys: list[list[str]] = []

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        del scrollback
        assert ansi is True
        if not self.snapshots:
            raise RuntimeError("stop")
        return self.snapshots.pop(0)

    async def send_text(self, text: str) -> None:
        self.text.append(text)

    async def send_keys(self, keys: list[str]) -> None:
        self.keys.append(keys)


class _WebSocket:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)


@pytest.mark.asyncio
async def test_snapshot_forwarder_sends_only_changed_screens() -> None:
    backend = _Backend()
    websocket = _WebSocket()

    with pytest.raises(RuntimeError, match="stop"):
        await _forward_snapshots_to_ws(websocket, backend, poll_interval_s=0)  # type: ignore[arg-type]

    assert websocket.frames == [
        b"\x1b[0m\x1b[H\x1b[2Jfirst\x1b[0m",
        b"\x1b[0m\x1b[H\x1b[2Jsecond\x1b[0m",
    ]


@pytest.mark.asyncio
async def test_snapshot_forwarder_normalizes_frame_rows() -> None:
    backend = _Backend()
    backend.snapshots = ["first row\nsecond row\n"]
    websocket = _WebSocket()

    with pytest.raises(RuntimeError, match="stop"):
        await _forward_snapshots_to_ws(websocket, backend, poll_interval_s=0)  # type: ignore[arg-type]

    assert websocket.frames == [b"\x1b[0m\x1b[H\x1b[2Jfirst row\r\nsecond row\x1b[0m"]


@pytest.mark.asyncio
async def test_browser_input_routes_through_backend_primitives() -> None:
    backend = _Backend()

    await _send_browser_input(backend, b"hello\rworld\x03")  # type: ignore[arg-type]
    await _send_browser_input(backend, b"\x1b[A")  # type: ignore[arg-type]

    assert backend.text == ["hello", "world"]
    assert backend.keys == [["Enter"], ["C-c"], ["Up"]]
