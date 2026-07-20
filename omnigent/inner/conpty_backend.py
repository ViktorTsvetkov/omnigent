"""Native Windows terminal backend built on ConPTY via pywinpty."""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from .terminal import (
    TERMINAL_TRANSPORT_STREAM,
    Liveness,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalLaunchRequest,
    register_terminal_backend,
)


class ConptyBackend(TerminalBackend):
    """Host one native Windows process in an isolated ConPTY."""

    name = "conpty"
    capabilities = TerminalBackendCapabilities(
        native_popup=False,
        start_on_attach=False,
        native_busy_state=False,
        push_events=True,
        control_mode_attach=False,
        attach_transports=frozenset({TERMINAL_TRANSPORT_STREAM}),
        status_line=False,
    )
    platforms = frozenset({"windows"})
    _OUTPUT_JOURNAL_LIMIT: ClassVar[int] = 8 * 1024 * 1024
    # How long the child's output must stay quiet before a write is
    # considered "settled" and a following capture will see its frame.
    _QUIET_WINDOW_S: ClassVar[float] = 0.02
    # Hard ceiling on that settle wait, so output that never goes quiet
    # cannot block the caller indefinitely.
    _QUIET_WAIT_CAP_S: ClassVar[float] = 0.2
    _OSC_COLOR_QUERY: ClassVar[re.Pattern[str]] = re.compile(r"\x1b\]1[012];\?(?:\x07|\x1b\\)")

    _KEYS: ClassVar[dict[str, str]] = {
        "Enter": "\r",
        "Escape": "\x1b",
        "Tab": "\t",
        "BTab": "\x1b[Z",
        "BSpace": "\x7f",
        "Backspace": "\x7f",
        "Up": "\x1b[A",
        "Down": "\x1b[B",
        "Right": "\x1b[C",
        "Left": "\x1b[D",
        "Home": "\x1b[H",
        "End": "\x1b[F",
        "PPage": "\x1b[5~",
        "PageUp": "\x1b[5~",
        "NPage": "\x1b[6~",
        "PageDown": "\x1b[6~",
        "IC": "\x1b[2~",
        "Insert": "\x1b[2~",
        "DC": "\x1b[3~",
        "Delete": "\x1b[3~",
    }

    @classmethod
    def ensure_available(cls) -> None:
        """Fail loudly when the Windows ConPTY binding is unavailable."""
        try:
            import winpty  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The conpty terminal backend requires pywinpty on Windows. "
                "Install omnigent with its Windows runtime dependencies."
            ) from exc

    @classmethod
    def construct_for_instance(cls, *, socket_path: Path, target: str) -> TerminalBackend:
        return cls(socket_path=socket_path, target=target)

    def __init__(self, *, socket_path: Path, target: str = "main") -> None:
        self._socket_path = socket_path
        self._target = target
        self._process: Any | None = None
        self._screen: Any | None = None
        self._stream: Any | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._output_generation = 0
        self._handshake_complete = False
        self._last_output_at = 0.0
        self._eof = False
        self._closed = False
        self._keep_alive_after_exit = False
        self._reader_error: BaseException | None = None
        self._output_journal: deque[bytes] = deque()
        self._output_journal_size = 0
        self._output_subscribers: set[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[bytes | None]]
        ] = set()

    async def launch(self, request: TerminalLaunchRequest) -> None:
        await asyncio.to_thread(self._launch_sync, request)

    def _launch_sync(self, request: TerminalLaunchRequest) -> None:
        self.ensure_available()
        import pyte
        import winpty

        cols, rows = request.size
        screen = pyte.HistoryScreen(cols, rows, history=max(1, request.scrollback))
        process = winpty.PtyProcess.spawn(
            request.command,
            cwd=request.cwd,
            env=request.env,
            dimensions=(rows, cols),
        )
        with self._lock:
            if self._process is not None and not self._closed:
                process.close(force=True)
                raise RuntimeError("ConPTY terminal is already running")
            self._process = process
            self._screen = screen
            self._stream = pyte.Stream(screen)
            self._closed = False
            self._keep_alive_after_exit = request.keep_alive_after_exit
            self._reader_error = None
            self._output_generation = 0
            self._handshake_complete = False
            self._last_output_at = 0.0
            self._eof = False
            self._output_journal.clear()
            self._output_journal_size = 0
            self._reader = threading.Thread(
                target=self._read_output,
                name=f"omnigent-conpty-{self._target}",
                daemon=True,
            )
            self._reader.start()
            self._wait_for_output_locked(0)

    def _read_output(self) -> None:
        import winpty

        while True:
            with self._lock:
                process = self._process
                closed = self._closed
            if process is None or closed:
                return
            try:
                chunk = process.read(4096)
            except EOFError:
                with self._changed:
                    self._eof = True
                    self._publish_output_locked(None)
                    self._changed.notify_all()
                return
            except (OSError, RuntimeError, winpty.WinptyError) as exc:
                with self._changed:
                    if not self._closed and process.isalive():
                        self._reader_error = exc
                    else:
                        self._eof = True
                        self._publish_output_locked(None)
                    self._changed.notify_all()
                return
            if chunk:
                with self._changed:
                    handshake = "\x1b[c" in chunk
                    if handshake:
                        # ConPTY asks its host for primary device attributes
                        # before releasing the child's normal output stream.
                        process.write("\x1b[?1;2c")
                        self._handshake_complete = True
                    browser_output = self._OSC_COLOR_QUERY.sub("", chunk.replace("\x1b[c", ""))
                    if browser_output:
                        encoded = browser_output.encode("utf-8")
                        self._append_output_locked(encoded)
                        self._publish_output_locked(encoded)
                    if self._stream is not None:
                        self._stream.feed(chunk)
                        if self._handshake_complete and not handshake:
                            self._output_generation += 1
                            self._last_output_at = time.monotonic()
                        self._changed.notify_all()

    async def liveness(self) -> Liveness:
        return self.liveness_sync()

    def liveness_sync(self) -> Liveness:
        with self._lock:
            process = self._process
            if self._closed or process is None:
                return Liveness.ENDPOINT_GONE
            if self._reader_error is not None:
                return Liveness.UNKNOWN
            try:
                alive = process.isalive()
            except (OSError, RuntimeError):
                return Liveness.UNKNOWN
            if alive and not self._eof:
                return Liveness.ALIVE
            return Liveness.INNER_EXITED if self._keep_alive_after_exit else Liveness.ENDPOINT_GONE

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._changed:
            process = self._process
            self._closed = True
            self._publish_output_locked(None)
            self._changed.notify_all()
        if process is not None:
            with contextlib.suppress(Exception):
                process.close(force=True)

    async def send_text(self, text: str) -> None:
        await asyncio.to_thread(self.send_text_sync, text)

    def send_text_sync(self, text: str) -> None:
        # Bracketed paste keeps embedded newlines as editable content instead of
        # turning each line into a separate submission in terminal applications.
        self._write(f"\x1b[200~{text}\x1b[201~")

    async def send_keys(self, keys: Sequence[str]) -> None:
        await asyncio.to_thread(self.send_keys_sync, keys)

    async def write_input(self, data: bytes) -> None:
        """
        Write interactive terminal input without paste-mode decoration.

        Deliberately skips :meth:`_wait_for_output_locked`. That barrier
        exists so an automation caller that writes and then calls
        :meth:`capture_sync` observes the resulting frame — the
        interactive byte stream has no such read-after-write, and paying
        it here froze typing: the attach bridge writes one frame at a
        time and does not read the next one off the WebSocket until the
        write completes, so a ~31 ms per-frame barrier capped input at
        ~32 frames/s. A paste or a fast burst then backlogged in the
        socket buffer for seconds with no echo, and landed all at once.

        :param data: Raw input bytes to write to the ConPTY, e.g.
            ``b"ls\\r"``.
        :returns: None once the bytes have been handed to the ConPTY.
        """
        await asyncio.to_thread(
            self._write,
            data.decode("utf-8", errors="replace"),
            wait_for_output=False,
        )

    async def set_size(self, cols: int, rows: int) -> None:
        """Resize the ConPTY without blocking the event loop."""
        await asyncio.to_thread(self.resize, cols, rows)

    def subscribe_output(self) -> asyncio.Queue[bytes | None]:
        """Subscribe to the VT stream, replaying retained output first."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        with self._lock:
            for chunk in self._output_journal:
                queue.put_nowait(chunk)
            if self._eof or self._closed:
                queue.put_nowait(None)
            else:
                self._output_subscribers.add((loop, queue))
        return queue

    def unsubscribe_output(self, queue: asyncio.Queue[bytes | None]) -> None:
        """Remove a VT stream subscriber."""
        with self._lock:
            self._output_subscribers = {
                subscriber for subscriber in self._output_subscribers if subscriber[1] is not queue
            }

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        encoded = "".join(sequence for key in keys if (sequence := self._translate_key(key)))
        if encoded:
            self._write(encoded)

    def paste_without_submit_sync(self, text: str) -> None:
        self.send_text_sync(text)

    def kill_session_sync(self) -> None:
        with self._changed:
            process = self._process
            self._closed = True
            self._publish_output_locked(None)
            self._changed.notify_all()
        if process is None:
            raise RuntimeError("ConPTY terminal is not running")
        try:
            process.close(force=True)
        except Exception as exc:
            raise RuntimeError(f"failed to stop ConPTY terminal: {exc}") from exc

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        return self.capture_sync(ansi=ansi, scrollback=scrollback)

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        with self._lock:
            if self.liveness_sync() == Liveness.ENDPOINT_GONE:
                raise RuntimeError("ConPTY terminal endpoint is gone")
            screen = self._screen
            if screen is None:
                raise RuntimeError("ConPTY terminal has not been launched")
            rows = list(screen.history.top)[-max(0, scrollback) :] if scrollback else []
            rows.extend(screen.buffer[y] for y in range(screen.lines))
            rendered = [self._render_row(row, ansi=ansi) for row in rows]
        return "\n".join(rendered).rstrip()

    def resize(self, cols: int, rows: int) -> None:
        """Resize both the native pseudoconsole and its server-side screen."""
        if cols <= 0 or rows <= 0:
            raise ValueError("terminal dimensions must be positive")
        with self._lock:
            process = self._require_live_process()
            process.setwinsize(rows, cols)
            screen = self._screen
            if screen is None:
                raise RuntimeError("ConPTY terminal has not been launched")
            screen.resize(lines=rows, columns=cols)

    def _append_output_locked(self, chunk: bytes) -> None:
        self._output_journal.append(chunk)
        self._output_journal_size += len(chunk)
        while (
            self._output_journal_size > self._OUTPUT_JOURNAL_LIMIT
            and len(self._output_journal) > 1
        ):
            self._output_journal_size -= len(self._output_journal.popleft())

    def _publish_output_locked(self, chunk: bytes | None) -> None:
        for loop, queue in self._output_subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, chunk)
        if chunk is None:
            self._output_subscribers.clear()

    def _write(self, data: str, *, wait_for_output: bool = True) -> None:
        """
        Write *data* to the ConPTY, optionally synchronising with the reader.

        :param data: Text to write to the pseudoconsole.
        :param wait_for_output: When true (the default), block until the
            reader thread has incorporated the resulting output, so a
            following :meth:`capture_sync` observes the new frame. The
            interactive path (:meth:`write_input`) passes false: it has
            no read-after-write and the wait would serialise typing.
        :raises RuntimeError: If the ConPTY is gone or the write fails.
        """
        with self._changed:
            process = self._require_live_process()
            generation = self._output_generation
            try:
                process.write(data)
            except (EOFError, OSError, RuntimeError) as exc:
                raise RuntimeError(f"failed to write to ConPTY terminal: {exc}") from exc
            if wait_for_output:
                self._wait_for_output_locked(generation)

    def _wait_for_output_locked(self, generation: int) -> None:
        """Wait until the reader incorporates output produced by an operation."""
        changed = self._changed.wait_for(
            lambda: (
                self._output_generation > generation
                or self._eof
                or self._reader_error is not None
                or self._closed
            ),
            timeout=2.0,
        )
        if not changed:
            return
        # One PTY write can arrive as many tiny reads. Wait until the reader has
        # drained a complete burst so capture observes the resulting frame.
        #
        # The quiet window is the primary exit. The deadline is only a
        # backstop: output that never goes quiet (a child emitting chunks
        # closer together than the window, indefinitely) would otherwise
        # block the caller forever, since nothing else ends this loop.
        deadline = time.monotonic() + self._QUIET_WAIT_CAP_S
        while not (self._eof or self._reader_error is not None or self._closed):
            now = time.monotonic()
            quiet_for = now - self._last_output_at
            if quiet_for >= self._QUIET_WINDOW_S:
                return
            if now >= deadline:
                return
            self._changed.wait(timeout=min(self._QUIET_WINDOW_S - quiet_for, deadline - now))

    def _require_live_process(self) -> Any:
        process = self._process
        if self._closed or process is None or not process.isalive():
            raise RuntimeError("ConPTY terminal endpoint is gone")
        return process

    @classmethod
    def _translate_key(cls, key: str) -> str | None:
        if key in cls._KEYS:
            return cls._KEYS[key]
        match = __import__("re").fullmatch(r"C-(.)", key, flags=__import__("re").IGNORECASE)
        if match:
            char = match.group(1).upper()
            return chr(ord(char) & 0x1F)
        match = __import__("re").fullmatch(r"M-(.)", key, flags=__import__("re").IGNORECASE)
        if match:
            return "\x1b" + match.group(1)
        if len(key) == 1:
            return key
        return None

    @staticmethod
    def _render_row(row: Any, *, ansi: bool) -> str:
        cells = [row[x] for x in sorted(row)]
        if not ansi:
            return "".join(cell.data for cell in cells).rstrip()
        output: list[str] = []
        active: tuple[Any, ...] | None = None
        for cell in cells:
            style = (
                cell.fg,
                cell.bg,
                cell.bold,
                cell.italics,
                cell.underscore,
                cell.strikethrough,
                cell.reverse,
            )
            if style != active:
                output.append(ConptyBackend._sgr(style))
                active = style
            output.append(cell.data)
        if active is not None:
            output.append("\x1b[0m")
        return "".join(output).rstrip()

    @staticmethod
    def _sgr(style: tuple[Any, ...]) -> str:
        fg, bg, bold, italics, underscore, strikethrough, reverse = style
        codes = ["0"]
        codes.extend(
            code
            for enabled, code in (
                (bold, "1"),
                (italics, "3"),
                (underscore, "4"),
                (reverse, "7"),
                (strikethrough, "9"),
            )
            if enabled
        )
        colors = {
            "black": 0,
            "red": 1,
            "green": 2,
            "brown": 3,
            "blue": 4,
            "magenta": 5,
            "cyan": 6,
            "white": 7,
        }
        if fg in colors:
            codes.append(str(30 + colors[fg]))
        if bg in colors:
            codes.append(str(40 + colors[bg]))
        return f"\x1b[{';'.join(codes)}m"


register_terminal_backend(ConptyBackend)
