"""Windows console mode support for interactive terminal attachment."""

from __future__ import annotations

import atexit
import ctypes
from dataclasses import dataclass
from typing import Any

STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
ENABLE_PROCESSED_INPUT = 0x0001
ENABLE_LINE_INPUT = 0x0002
ENABLE_ECHO_INPUT = 0x0004
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


@dataclass(frozen=True)
class ConsoleModeState:
    """Console handles and modes to restore after an interactive attach."""

    stdin_handle: int
    stdin_mode: int
    stdout_handle: int
    stdout_mode: int


_active_states: list[ConsoleModeState] = []


def _kernel32() -> Any:
    """Return the Win32 console API object."""
    return ctypes.windll.kernel32  # type: ignore[attr-defined]


def _get_mode(kernel32: Any, handle: int) -> int | None:
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return None
    return int(mode.value)


def enter_console_mode() -> ConsoleModeState | None:
    """Enable raw VT input and VT output, returning modes to restore."""
    kernel32 = _kernel32()
    stdin_handle = int(kernel32.GetStdHandle(STD_INPUT_HANDLE))
    stdout_handle = int(kernel32.GetStdHandle(STD_OUTPUT_HANDLE))
    stdin_mode = _get_mode(kernel32, stdin_handle)
    stdout_mode = _get_mode(kernel32, stdout_handle)
    if stdin_mode is None or stdout_mode is None:
        return None

    raw_input = stdin_mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT)
    raw_input |= ENABLE_VIRTUAL_TERMINAL_INPUT
    vt_output = stdout_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if not kernel32.SetConsoleMode(stdin_handle, raw_input):
        raise ctypes.WinError()
    if not kernel32.SetConsoleMode(stdout_handle, vt_output):
        kernel32.SetConsoleMode(stdin_handle, stdin_mode)
        raise ctypes.WinError()

    state = ConsoleModeState(stdin_handle, stdin_mode, stdout_handle, stdout_mode)
    _active_states.append(state)
    return state


def restore_console_mode(state: ConsoleModeState | None) -> None:
    """Restore a console state previously returned by :func:`enter_console_mode`."""
    if state is None:
        return
    kernel32 = _kernel32()
    kernel32.SetConsoleMode(state.stdin_handle, state.stdin_mode)
    kernel32.SetConsoleMode(state.stdout_handle, state.stdout_mode)
    if state in _active_states:
        _active_states.remove(state)


def _restore_active_console_modes() -> None:
    for state in reversed(_active_states.copy()):
        restore_console_mode(state)


atexit.register(_restore_active_console_modes)
