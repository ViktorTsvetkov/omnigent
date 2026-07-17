"""FakeBackend parallel proof — mirrors ``tests/inner/test_terminal.py``.

This module exercises the same :class:`~omnigent.inner.terminal.TerminalInstance`
behaviors that :mod:`tests.inner.test_terminal` proves with *ad-hoc tmux
stubbing* — patching ``asyncio.create_subprocess_exec`` to feed canned tmux
argv/stdout — but drives them through the in-process
:class:`~tests.inner.fake_terminal_backend.FakeBackend` instead. It is a
**parallel** module, not a rewrite: per the owner's strictly-additive rule the
original ``test_terminal.py`` stays byte-identical, and true replacement of its
ad-hoc stubbing awaits owner approval. The pairing is deliberate — each test
here names the ``test_terminal.py`` test it mirrors.

Why this is a better test double: the mirrored originals reach into
``TerminalInstance`` internals (monkeypatching ``_capture_pane_for_idle_or_none``
/ ``_pane_is_dead``, or swapping the whole ``asyncio`` module) to fake what a
multiplexer would do. Here the instance runs on a real, registered backend built
through the production construction hook
(:meth:`TerminalBackend.construct_for_instance`), so the seam itself is
exercised, no subprocess is patched, and the tests run unchanged on native
Windows.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import omnigent.inner.terminal as terminal_mod
from omnigent.inner.terminal import Liveness, TerminalInstance
from tests.inner.fake_terminal_backend import SUBMIT_SENTINEL, FakeBackend


def _instance_on_fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: object,
) -> tuple[TerminalInstance, FakeBackend]:
    """Build a ``TerminalInstance`` whose backend is a scriptable ``FakeBackend``.

    Registers the fake under ``"fake"`` with ``monkeypatch.setitem`` — so it
    never leaks into production selection, exactly as the selection tests do —
    then constructs an instance with that ``backend_name``. The instance builds
    its backend through the production construction hook, so ``instance._backend``
    is the fake the caller then scripts.

    :returns: The instance and its ``FakeBackend``.
    """
    monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", FakeBackend)
    instance = TerminalInstance(
        name="bash",
        session_key="s1",
        socket_path=tmp_path / "fake.sock",
        private_dir=tmp_path,
        backend_name="fake",
        **kwargs,  # type: ignore[arg-type]
    )
    backend = instance._backend
    assert isinstance(backend, FakeBackend)
    return instance, backend


# ---------------------------------------------------------------------------
# backend_capabilities — the seam the native cost popup degrades on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend_name", "expected_native_popup"),
    [("tmux", True), ("herdr", False)],
)
def test_backend_capabilities_expose_native_popup_flag(
    tmp_path: Path, backend_name: str, expected_native_popup: bool
) -> None:
    """A TerminalInstance exposes its backend's ``native_popup`` capability.

    The native cost-popup dispatch reads
    ``instance.backend_capabilities.native_popup`` to decide whether to render a
    pane overlay or degrade to the web approval card: tmux hosts the popup
    (``True``); herdr on Windows cannot (``False``), so the ASK verdict flows to
    the web card. Construction goes through the production hook and probes no
    binary, so this runs on any platform for both backends.
    """
    instance = TerminalInstance(
        name="codex",
        session_key="main",
        socket_path=tmp_path / "tmux.sock",
        private_dir=tmp_path,
        backend_name=backend_name,
    )
    assert instance.backend_capabilities.native_popup is expected_native_popup


# ---------------------------------------------------------------------------
# is_alive — mirrors test_is_alive_true_when_pane_live / _false_when_pane_dead
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_alive_true_when_pane_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_alive`` reports True when the backend reports the pane ALIVE.

    Mirrors ``test_terminal.test_is_alive_true_when_pane_live``, which fed a
    ``#{pane_dead}`` -> ``0`` stdout through a patched subprocess. Here the fake
    reports :attr:`Liveness.ALIVE` from its own state, no subprocess involved.
    """
    instance, _backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    assert await instance.is_alive() is True
    assert instance.running is True


@pytest.mark.asyncio
async def test_is_alive_false_when_pane_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_alive`` reports False (and clears ``running``) for a dead pane.

    Mirrors ``test_terminal.test_is_alive_false_when_pane_dead``: with the inner
    process exited but the endpoint kept, liveness is :attr:`Liveness.INNER_EXITED`
    — not ALIVE — so ``is_alive`` is False and flips ``running`` off so later
    pollers short-circuit. The original faked ``#{pane_dead}`` -> ``1``; here we
    script the verdict directly.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    backend.push_liveness([Liveness.INNER_EXITED])
    assert await instance.is_alive() is False
    assert instance.running is False


# ---------------------------------------------------------------------------
# send / read — mirrors the send-chunking + read behaviors of test_terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_delivers_text_then_submits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``send`` types literal text then submits it with Enter.

    Mirrors the delivery behavior of
    ``test_terminal.test_send_chunks_long_literal_text_under_tmux_command_cap``
    (the tmux-specific 16KB chunking is a backend concern the conformance suite
    covers; the instance-level contract is "text is delivered, then submitted").
    Through the fake the delivered stream is observable in memory: the pasted
    text landed, an Enter followed, and the submit sentinel proves it submitted.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)

    result = await instance.send(text="hello world")

    assert result == {"status": "sent"}
    assert backend.sent_text == ["hello world"]
    assert backend.sent_keys == [["Enter"]]
    screen = await backend.capture()
    assert "hello world" in screen
    assert SUBMIT_SENTINEL in screen  # the Enter submitted


@pytest.mark.asyncio
async def test_read_returns_pane_screen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``read`` returns the backend's captured screen, ANSI stripped.

    Mirrors the read path exercised throughout ``test_terminal.py``; here the
    screen is seeded in memory and read back without a ``capture-pane``
    subprocess.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    backend.set_screen("\x1b[31mred line\x1b[0m\nplain line")

    result = await instance.read()

    assert result["terminal"] == "bash:s1"
    assert result["screen"] == "red line\nplain line"  # ANSI stripped
    assert result["scrollback_lines"] == 0


@pytest.mark.asyncio
async def test_send_on_stopped_instance_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``send`` on a not-running instance returns an error without touching the backend.

    Mirrors the not-running guard the send/read paths share in ``test_terminal``.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=False)
    result = await instance.send(text="ignored")
    assert result == {"error": "Terminal is not running"}
    assert backend.sent_text == []


# ---------------------------------------------------------------------------
# Threaded idle watcher — mirrors the three watcher tests of test_terminal
# ---------------------------------------------------------------------------


def test_threaded_idle_watcher_reports_terminal_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The threaded watcher reports the endpoint's disappearance via on_exit.

    Mirrors ``test_terminal.test_threaded_idle_watcher_reports_terminal_exit``,
    which patched ``_capture_pane_for_idle_or_none`` to return ``None``. Here a
    scripted ``None`` snapshot makes the backend's capture raise, which the
    watcher reads as "host gone".
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    backend.push_snapshots([None])
    exited = threading.Event()

    instance.start_idle_watcher_thread(on_exit=exited.set, poll_interval_s=0.01)

    assert exited.wait(timeout=1.0)
    assert instance.running is False


def test_threaded_idle_watcher_keeps_last_pane_text_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last pane text survives so an exit can be diagnosed.

    Mirrors ``test_terminal.test_threaded_idle_watcher_keeps_last_pane_text_on_exit``:
    one real snapshot, then the host vanishes. The remembered snapshot is
    returned ANSI-stripped by ``last_pane_text``.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    backend.push_snapshots(["\x1b[31mstartup failed\x1b[0m\ntry config", None])
    exited = threading.Event()

    instance.start_idle_watcher_thread(on_exit=exited.set, poll_interval_s=0.01)

    assert exited.wait(timeout=1.0)
    assert instance.last_pane_text() == "startup failed\ntry config"


def test_threaded_idle_watcher_reports_exit_on_dead_pane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead pane (process exited, endpoint kept) fires on_exit.

    Mirrors ``test_terminal.test_threaded_idle_watcher_reports_exit_on_dead_pane``
    (issue #540): capture still succeeds against the surviving endpoint, but the
    liveness verdict is :attr:`Liveness.INNER_EXITED`, so the watcher reports the
    exit instead of mistaking the frozen final frame for an idle agent. The
    original patched ``_pane_is_dead``; here the fake's scripted verdict drives
    it through the real seam.
    """
    instance, backend = _instance_on_fake(tmp_path, monkeypatch, running=True)
    backend.push_snapshots(["claude exited: boom\nbye"])
    backend.push_liveness([Liveness.INNER_EXITED])
    exited = threading.Event()

    instance.start_idle_watcher_thread(on_exit=exited.set, poll_interval_s=0.01)

    assert exited.wait(timeout=1.0)
    assert instance.running is False
    assert instance.last_pane_text() == "claude exited: boom\nbye"
