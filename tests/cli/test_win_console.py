"""Windows local-console attach tests."""

from __future__ import annotations

import asyncio
import ctypes
from types import SimpleNamespace

import pytest

import omnigent._win_console as win_console
import omnigent.claude_native as claude_native


class _FakeKernel32:
    def __init__(self) -> None:
        self.modes = {101: 0x0007, 102: 0x0001}
        self.set_calls: list[tuple[int, int]] = []

    def GetStdHandle(self, handle_id: int) -> int:
        return {-10: 101, -11: 102}[handle_id]

    def GetConsoleMode(self, handle: int, mode_ptr: object) -> int:
        ctypes.cast(mode_ptr, ctypes.POINTER(ctypes.c_uint32))[0] = self.modes[handle]
        return 1

    def SetConsoleMode(self, handle: int, mode: int) -> int:
        self.modes[handle] = mode
        self.set_calls.append((handle, mode))
        return 1


def test_console_modes_set_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel32 = _FakeKernel32()
    monkeypatch.setattr(win_console, "_kernel32", lambda: kernel32)

    state = win_console.enter_console_mode()

    assert state is not None
    assert kernel32.modes[101] == win_console.ENABLE_VIRTUAL_TERMINAL_INPUT
    assert kernel32.modes[102] == 0x0001 | win_console.ENABLE_VIRTUAL_TERMINAL_PROCESSING

    win_console.restore_console_mode(state)

    assert kernel32.modes == {101: 0x0007, 102: 0x0001}
    assert state not in win_console._active_states


def test_windows_platform_dispatches_console_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    state = SimpleNamespace(name="console-state")
    restored: list[object] = []
    monkeypatch.setattr(claude_native, "IS_WINDOWS", True)
    monkeypatch.setattr(win_console, "enter_console_mode", lambda: state)
    monkeypatch.setattr(win_console, "restore_console_mode", restored.append)

    saved = claude_native._enter_raw_mode(0)
    claude_native._restore_terminal(0, saved)
    signal_restore = claude_native._install_attach_signal_handlers(object(), 0)

    assert saved is state
    assert restored == [state]
    assert signal_restore.received_signal is None
    signal_restore.restore()


@pytest.mark.asyncio
async def test_windows_resize_poller_sends_changed_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = iter(
        [
            claude_native.os.terminal_size((80, 24)),
            claude_native.os.terminal_size((100, 40)),
            claude_native.os.terminal_size((100, 40)),
        ]
    )
    sent: list[str] = []
    ws = SimpleNamespace(send=lambda payload: _record(sent, payload))
    monkeypatch.setattr(claude_native.os, "isatty", lambda _fd: True)
    monkeypatch.setattr(claude_native.os, "get_terminal_size", lambda: next(sizes))

    task = asyncio.create_task(claude_native._poll_terminal_resize(ws, 0, interval_s=0))
    while not sent:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert '"cols": 100' in sent[0]
    assert '"rows": 40' in sent[0]


async def _record(items: list[str], payload: str) -> None:
    items.append(payload)
