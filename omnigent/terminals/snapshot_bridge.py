"""Snapshot-polling terminal backend to browser WebSocket bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Callable

from fastapi import WebSocket, WebSocketDisconnect

from omnigent.inner.terminal import TerminalBackend

_POLL_INTERVAL_S = 0.1
_REDRAW_PREFIX = b"\x1b[0m\x1b[H\x1b[2J"
_CAPTURE_ROW_SEP_RE = re.compile(rb"(?<!\r)\n")


async def _forward_snapshots_to_ws(
    websocket: WebSocket,
    backend: TerminalBackend,
    *,
    poll_interval_s: float = _POLL_INTERVAL_S,
) -> None:
    """Poll ANSI snapshots and send a redraw only when the screen changes."""
    previous: str | None = None
    while True:
        snapshot = await backend.capture(ansi=True)
        if snapshot != previous:
            previous = snapshot
            body = snapshot.encode("utf-8")
            body = body[:-1] if body.endswith(b"\n") else body
            normalized = _CAPTURE_ROW_SEP_RE.sub(b"\r\n", body)
            await websocket.send_bytes(_REDRAW_PREFIX + normalized + b"\x1b[0m")
        await asyncio.sleep(poll_interval_s)


async def _send_browser_input(backend: TerminalBackend, data: bytes) -> None:
    """Route xterm input bytes through backend-neutral send primitives."""
    escape_key = {
        b"\x1b[A": "Up",
        b"\x1b[B": "Down",
        b"\x1b[C": "Right",
        b"\x1b[D": "Left",
        b"\x1b[3~": "Delete",
    }.get(data)
    if escape_key is not None:
        await backend.send_keys([escape_key])
        return
    text = data.decode("utf-8", errors="replace")
    literal: list[str] = []

    async def flush() -> None:
        if literal:
            await backend.send_text("".join(literal))
            literal.clear()

    key_names = {"\r": "Enter", "\n": "Enter", "\t": "Tab", "\x7f": "Backspace", "\x1b": "Escape"}
    for char in text:
        key = key_names.get(char)
        if key is None and ord(char) < 32:
            key = f"C-{chr(ord(char) + 96)}"
        if key is None:
            literal.append(char)
            continue
        await flush()
        await backend.send_keys([key])
    await flush()


async def bridge_snapshot_to_websocket(
    websocket: WebSocket,
    *,
    backend: TerminalBackend,
    read_only: bool = False,
    on_client_interaction: Callable[[], None] | None = None,
) -> None:
    """Bridge a polling-only terminal backend over the existing WS protocol."""

    async def receive_input() -> None:
        while True:
            message = await websocket.receive()
            if on_client_interaction is not None:
                on_client_interaction()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("text") is not None:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    json.loads(message["text"])
            elif message.get("bytes") is not None and not read_only:
                await _send_browser_input(backend, message["bytes"])

    if on_client_interaction is not None:
        on_client_interaction()
    output_task = asyncio.create_task(_forward_snapshots_to_ws(websocket, backend))
    input_task = asyncio.create_task(receive_input())
    try:
        done, pending = await asyncio.wait(
            {output_task, input_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                task.result()
    finally:
        if on_client_interaction is not None:
            on_client_interaction()
