"""Shared terminal-backend conformance suite for native Windows ConPTY."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

from omnigent.inner.conpty_backend import ConptyBackend
from omnigent.inner.terminal import TerminalBackend, TerminalLaunchRequest
from tests.inner.terminal_backend_conformance import (
    BackendConformanceSuite,
    ConformanceAdapter,
)

_CHILD = Path(__file__).with_name("_conpty_conformance_child.py")


class _OutputProcess:
    def __init__(self, chunk: str) -> None:
        self._chunks = iter((chunk,))

    def read(self, size: int) -> str:
        del size
        try:
            return next(self._chunks)
        except StopIteration:
            raise EOFError from None

    def write(self, data: str) -> None:
        del data


class ConptyConformanceAdapter(ConformanceAdapter):
    backend_cls = ConptyBackend
    submit_sentinel = "CONPTY_SUBMITTED"
    final_screen_marker = "CONPTY_FINAL_SCREEN"
    sample_keys = ("Tab", "BSpace")

    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        del monkeypatch
        return ConptyBackend(socket_path=tmp_path / "conpty", target="main")

    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        return TerminalLaunchRequest(
            command=command,
            cwd=str(_CHILD.parent),
            env=dict(__import__("os").environ),
            size=(120, 40),
            keep_alive_after_exit=keep_alive_after_exit,
        )

    def alive_command(self) -> list[str]:
        return [sys.executable, str(_CHILD)]

    def exiting_command(self) -> list[str]:
        return [sys.executable, str(_CHILD), "--exit"]

    def key_marker(self, key: str) -> str:
        return f"<KEY:{key}>"

    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        del monkeypatch, tmp_path
        backend._reader_error = OSError("probe unavailable")  # type: ignore[attr-defined]


@pytest.mark.windows_only
class TestConptyConformance(BackendConformanceSuite):
    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        return ConptyConformanceAdapter()


@pytest.mark.windows_only
def test_read_output_strips_only_osc_color_queries(tmp_path: Path) -> None:
    backend = ConptyBackend(socket_path=tmp_path / "conpty", target="main")
    backend._process = _OutputProcess(  # type: ignore[assignment]
        "before\x1b]10;?\x1b\\middle\x1b]11;?\x07"
        "\x1b]12;?\x1b\\after\x1b]10;rgb:1818/1818/1b1b\x1b\\"
    )

    backend._read_output()

    assert b"".join(backend._output_journal) == (
        b"beforemiddleafter\x1b]10;rgb:1818/1818/1b1b\x1b\\"
    )


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_osc_color_queries_are_not_replayed_on_resubscribe(tmp_path: Path) -> None:
    backend = ConptyBackend(socket_path=tmp_path / "conpty", target="main")
    backend._process = _OutputProcess(  # type: ignore[assignment]
        "ready\x1b]10;?\x1b\\\x1b]11;?\x07"
    )
    backend._read_output()

    first = backend.subscribe_output()
    assert await asyncio.wait_for(first.get(), timeout=1) == b"ready"
    assert await asyncio.wait_for(first.get(), timeout=1) is None

    second = backend.subscribe_output()
    replay = await asyncio.wait_for(second.get(), timeout=1)
    assert replay == b"ready"
    assert b"]10;" not in replay
    assert b"]11;" not in replay


class _WriteRecordingProcess:
    """Minimal live ConPTY process stub that records writes."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def isalive(self) -> bool:
        return True

    def write(self, data: str) -> None:
        self.writes.append(data)


def _backend_with_recording_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ConptyBackend, _WriteRecordingProcess, list[int]]:
    """Build a backend whose capture-sync barrier is counted, not executed."""
    backend = ConptyBackend(socket_path=tmp_path / "conpty", target="main")
    process = _WriteRecordingProcess()
    backend._process = process  # type: ignore[assignment]
    waits: list[int] = []
    monkeypatch.setattr(
        ConptyBackend,
        "_wait_for_output_locked",
        lambda self, generation: waits.append(generation),
    )
    return backend, process, waits


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_write_input_skips_the_capture_quiet_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive input must not wait for the reader, or typing serialises."""
    backend, process, waits = _backend_with_recording_process(tmp_path, monkeypatch)

    await backend.write_input(b"hello")

    assert process.writes == ["hello"]
    assert waits == []


@pytest.mark.windows_only
def test_send_text_and_keys_still_wait_for_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The automation path keeps the barrier so capture sees the new frame."""
    backend, process, waits = _backend_with_recording_process(tmp_path, monkeypatch)

    backend.send_text_sync("hi")
    assert len(waits) == 1

    backend.send_keys_sync(["Enter"])
    assert len(waits) == 2

    backend.paste_without_submit_sync("more")
    assert len(waits) == 3

    assert process.writes == ["\x1b[200~hi\x1b[201~", "\r", "\x1b[200~more\x1b[201~"]


@pytest.mark.windows_only
def test_quiet_wait_returns_promptly_once_output_settles(tmp_path: Path) -> None:
    """The quiet window stays the primary exit: a settled terminal returns at once."""
    backend = ConptyBackend(socket_path=tmp_path / "conpty", target="main")
    backend._output_generation = 1
    backend._last_output_at = time.monotonic() - 1.0

    started = time.monotonic()
    with backend._changed:
        backend._wait_for_output_locked(0)
    elapsed = time.monotonic() - started

    assert elapsed < 0.05


@pytest.mark.windows_only
def test_quiet_wait_is_bounded_when_output_never_settles(tmp_path: Path) -> None:
    """Output that never goes quiet must not block the caller indefinitely."""
    backend = ConptyBackend(socket_path=tmp_path / "conpty", target="main")
    backend._output_generation = 1
    backend._last_output_at = time.monotonic()

    stop = threading.Event()

    def _keep_emitting() -> None:
        """Bump the output clock faster than the quiet window, forever."""
        while not stop.is_set():
            with backend._changed:
                backend._last_output_at = time.monotonic()
                backend._changed.notify_all()
            time.sleep(backend._QUIET_WINDOW_S / 4)

    emitter = threading.Thread(target=_keep_emitting, daemon=True)
    emitter.start()
    try:
        started = time.monotonic()
        with backend._changed:
            backend._wait_for_output_locked(0)
        elapsed = time.monotonic() - started
    finally:
        stop.set()
        emitter.join(timeout=5)

    # It must have actually hit the cap (not exited early via the quiet
    # window), and it must not have run away past it.
    assert elapsed >= backend._QUIET_WAIT_CAP_S * 0.75
    assert elapsed < 0.6
