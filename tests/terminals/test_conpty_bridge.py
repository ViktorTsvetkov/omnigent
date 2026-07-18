"""Tests for the native ConPTY streaming WebSocket bridge."""

from __future__ import annotations

import asyncio
import json

import pytest

from omnigent.terminals.conpty_bridge import bridge_conpty_to_websocket


class _Backend:
    def __init__(self) -> None:
        self.output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.output.put_nowait(b"\x1b[2;4Hready")
        self.inputs: list[bytes] = []
        self.sizes: list[tuple[int, int]] = []
        self.unsubscribed = False

    def subscribe_output(self) -> asyncio.Queue[bytes | None]:
        return self.output

    def unsubscribe_output(self, queue: asyncio.Queue[bytes | None]) -> None:
        assert queue is self.output
        self.unsubscribed = True

    async def write_input(self, data: bytes) -> None:
        self.inputs.append(data)

    async def set_size(self, cols: int, rows: int) -> None:
        self.sizes.append((cols, rows))


class _WebSocket:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = messages
        self.frames: list[bytes] = []
        self.output_sent = asyncio.Event()

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)
        self.output_sent.set()

    async def receive(self) -> dict[str, object]:
        await self.output_sent.wait()
        if self.messages:
            return self.messages.pop(0)
        return {"type": "websocket.disconnect"}


@pytest.mark.asyncio
async def test_bridge_streams_vt_input_and_resize() -> None:
    backend = _Backend()
    websocket = _WebSocket(
        [
            {"text": json.dumps({"type": "resize", "cols": 132, "rows": 43})},
            {"bytes": b"echo hi\r"},
        ]
    )

    await bridge_conpty_to_websocket(websocket, backend=backend)

    assert websocket.frames == [b"\x1b[2;4Hready"]
    assert backend.sizes == [(132, 43)]
    assert backend.inputs == [b"echo hi\r"]
    assert backend.unsubscribed


@pytest.mark.asyncio
async def test_bridge_drops_read_only_input_and_invalid_resize() -> None:
    backend = _Backend()
    websocket = _WebSocket(
        [
            {"text": json.dumps({"type": "resize", "cols": 0, "rows": 24})},
            {"bytes": b"blocked"},
        ]
    )

    await bridge_conpty_to_websocket(websocket, backend=backend, read_only=True)

    assert backend.sizes == []
    assert backend.inputs == []
