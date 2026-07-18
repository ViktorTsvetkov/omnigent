"""Native ConPTY byte-stream to browser WebSocket bridge."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Protocol, cast

from fastapi import WebSocket, WebSocketDisconnect

from omnigent.inner.terminal import TerminalBackend
from omnigent.terminals.ws_bridge import _forward_pty_to_ws


class _StreamingTerminalBackend(Protocol):
    def subscribe_output(self) -> asyncio.Queue[bytes | None]: ...

    def unsubscribe_output(self, queue: asyncio.Queue[bytes | None]) -> None: ...

    async def write_input(self, data: bytes) -> None: ...

    async def set_size(self, cols: int, rows: int) -> None: ...


async def bridge_conpty_to_websocket(
    websocket: WebSocket,
    *,
    backend: TerminalBackend,
    read_only: bool = False,
    on_client_interaction: Callable[[], None] | None = None,
) -> None:
    """Stream one ConPTY's VT output and browser input over the shared protocol."""
    streaming_backend = cast(_StreamingTerminalBackend, backend)
    output = streaming_backend.subscribe_output()

    async def receive_input() -> None:
        try:
            while True:
                message = await websocket.receive()
                if on_client_interaction is not None:
                    on_client_interaction()
                if message.get("type") == "websocket.disconnect":
                    return
                text = message.get("text")
                data = message.get("bytes")
                if text is not None:
                    try:
                        control = json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if isinstance(control, dict) and control.get("type") == "resize":
                        try:
                            cols = int(control["cols"])
                            rows = int(control["rows"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if cols > 0 and rows > 0:
                            await streaming_backend.set_size(cols, rows)
                elif data is not None and not read_only:
                    await streaming_backend.write_input(data)
        except WebSocketDisconnect:
            return

    if on_client_interaction is not None:
        on_client_interaction()
    output_task = asyncio.create_task(_forward_pty_to_ws(websocket, output))
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
        streaming_backend.unsubscribe_output(output)
        if on_client_interaction is not None:
            on_client_interaction()
