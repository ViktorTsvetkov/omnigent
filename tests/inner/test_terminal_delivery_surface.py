"""Tests for the shared prompt-delivery surface (:class:`TerminalDelivery`).

Two layers, matching ticket #8's acceptance criteria:

1. **Fake-backend behavior** — the surface driven against
   :class:`~tests.inner.fake_terminal_backend.FakeBackend`, asserting the
   delivery dance's external behavior (paste never submits, submit verification
   commits/retries/raises, interrupt keys are delivered) with no multiplexer.
2. **tmux command-stream pinning** — :class:`~omnigent.inner.terminal.TmuxBackend`
   delivery methods with ``subprocess.run`` patched, pinning the exact ``tmux``
   argv each delivery op emits. This is the proof that migrating the seven native
   bridges onto the surface keeps their POSIX command stream byte-identical to
   the private helpers they used. The backend is built with a *string* socket so
   the argv is byte-exact on any platform (no ``Path`` round-trip), which is what
   real POSIX delivery produces.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.inner.terminal import (
    TerminalDelivery,
    TerminalLaunchRequest,
    TmuxBackend,
    _tmux_paste_payload_bytes,
    build_prompt_delivery,
)
from tests.inner.fake_terminal_backend import SUBMIT_SENTINEL, FakeBackend, key_marker

_SOCKET = "/tmp/example/tmux.sock"
_TARGET = "claude:0.0"


async def _launched_fake() -> FakeBackend:
    """Return a launched fake backend (endpoint alive, empty screen)."""
    backend = FakeBackend()
    await backend.launch(TerminalLaunchRequest(command=["sh"], cwd="/", env={}))
    return backend


# ─────────────────────────── fake-backend behavior (AC1) ────────────────────


async def test_paste_without_submit_does_not_submit() -> None:
    """A pasted draft lands on the screen but never carries a submit.

    The delivery-dance invariant: paste deposits an editable multi-line draft;
    only an explicit ``Enter`` submits. A regression that submitted on paste
    would fire the turn with a half-typed prompt.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.paste_without_submit("fix the bug\nin foo.py")

    assert backend.pasted_text == ["fix the bug\nin foo.py"]
    screen = delivery.snapshot()
    assert "fix the bug\nin foo.py" in screen
    # No submit happened — the fake appends SUBMIT_SENTINEL only on a named Enter.
    assert SUBMIT_SENTINEL not in screen


async def test_submit_and_verify_submits_once_when_draft_clears() -> None:
    """When the draft leaves the box after one Enter, exactly one Enter is sent.

    A slow-but-successful submit must not be double-tapped: the verify loop
    stops the instant the draft clears.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)
    # Draft present until the first Enter clears it; snapshots key off a marker.
    state = {"submitted": False}

    def draft_present(_pane: str) -> bool:
        return not state["submitted"]

    # Flip the state when the fake records an Enter.
    original_send = backend.send_keys_sync

    def tracking_send(keys: list[str]) -> None:
        original_send(keys)
        if "Enter" in keys:
            state["submitted"] = True

    backend.send_keys_sync = tracking_send  # type: ignore[method-assign]

    delivery.submit_and_verify(
        draft_present=draft_present,
        error_message="not delivered",
        poll_interval_s=0.0,
        commit_timeout_s=0.5,
        settle_s=0.0,
        verify_timeout_s=0.5,
        retry_interval_s=0.01,
    )

    # One Enter total: the initial submit; the draft cleared so no retry fired.
    assert backend.sent_keys.count(["Enter"]) == 1


async def test_submit_and_verify_retries_swallowed_enter() -> None:
    """A submit Enter swallowed into the paste burst is retried until it takes.

    Models the "typed but never sent" bug: the first Enter is folded into the
    draft; the verify loop must observe the draft still present and re-send.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)
    enters = {"n": 0}

    def draft_present(_pane: str) -> bool:
        # Draft is committed for the commit-wait, stays present through the
        # first (swallowed) Enter, and clears once the second Enter lands.
        return enters["n"] < 2

    original_send = backend.send_keys_sync

    def tracking_send(keys: list[str]) -> None:
        original_send(keys)
        if "Enter" in keys:
            enters["n"] += 1

    backend.send_keys_sync = tracking_send  # type: ignore[method-assign]

    delivery.submit_and_verify(
        draft_present=draft_present,
        error_message="not delivered",
        poll_interval_s=0.0,
        commit_timeout_s=0.5,
        settle_s=0.0,
        verify_timeout_s=1.0,
        retry_interval_s=0.0,
    )

    # Exactly two Enters: the swallowed one plus one retry.
    assert enters["n"] == 2


async def test_submit_and_verify_raises_when_draft_never_clears() -> None:
    """When the draft never leaves the box, the caller's error is raised.

    Returning success would complete an Omnigent turn with the message still
    sitting unsent — the failure must surface.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    with pytest.raises(RuntimeError, match="not delivered"):
        delivery.submit_and_verify(
            draft_present=lambda _pane: True,  # never clears
            error_message="not delivered",
            poll_interval_s=0.0,
            commit_timeout_s=0.05,
            settle_s=0.0,
            verify_timeout_s=0.1,
            retry_interval_s=0.0,
        )


async def test_submit_and_verify_submits_blind_when_draft_never_identifiable() -> None:
    """When the draft is never observed, submit once and do not verify.

    Whitespace-only content has no identifiable draft, so the surface submits
    blind (one Enter) rather than looping — the draft's absence can't prove a
    failed submit.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.submit_and_verify(
        draft_present=lambda _pane: False,  # never seen
        error_message="not delivered",
        poll_interval_s=0.0,
        commit_timeout_s=0.05,
        settle_s=0.0,
        verify_timeout_s=0.1,
        retry_interval_s=0.0,
    )

    assert backend.sent_keys.count(["Enter"]) == 1


async def test_interrupt_key_is_delivered() -> None:
    """A named interrupt key reaches the backend as that key."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.send_keys(["Escape"])

    assert backend.sent_keys == [["Escape"]]
    assert key_marker("Escape") in delivery.snapshot()


async def test_ctrl_c_interrupt_key_is_delivered() -> None:
    """A Ctrl-C chord reaches the backend verbatim in the neutral vocabulary."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.send_keys(["C-c"])

    assert backend.sent_keys == [["C-c"]]


async def test_kill_marks_endpoint_gone() -> None:
    """A hard kill tears the endpoint down (snapshot then reads empty)."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.kill()

    assert backend.killed is True
    assert delivery.snapshot() == ""


async def test_launch_native_popup_is_noop_without_capability() -> None:
    """The popup no-ops on a backend without the native-popup capability.

    The fake advertises no capabilities, so the web approval card remains the
    surface — the documented degradation.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)
    called = {"n": 0}
    backend.native_popup_launch = lambda **_kw: called.__setitem__("n", called["n"] + 1)  # type: ignore[method-assign]

    delivery.launch_native_popup(
        config_file=Path("/tmp/cfg.json"),
        session_id="conv_1",
        elicitation_id="elicit_1",
        message="continue?",
    )

    assert called["n"] == 0


# ─────────────────────────── tmux command-stream pinning ────────────────────


class _TmuxRecorder:
    """Patches ``subprocess.run`` to record tmux argv and simulate rc/output."""

    def __init__(self, capture_output: str = "") -> None:
        self.calls: list[list[str]] = []
        self.loaded_payloads: list[bytes] = []
        self._capture_output = capture_output
        self._rc = 0
        self._stderr = ""

    def fail_with(self, stderr: str) -> None:
        """Make every subsequent non-capture call exit non-zero with *stderr*."""
        self._rc = 1
        self._stderr = stderr

    def __call__(self, cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        if "capture-pane" in cmd:
            return SimpleNamespace(returncode=0, stdout=self._capture_output, stderr="")
        if "load-buffer" in cmd:
            self.loaded_payloads.append(Path(cmd[-1]).read_bytes())
        self.calls.append(cmd)
        return SimpleNamespace(returncode=self._rc, stdout="", stderr=self._stderr)


@pytest.fixture
def tmux_delivery(monkeypatch: pytest.MonkeyPatch) -> tuple[TerminalDelivery, _TmuxRecorder]:
    """A tmux-backed delivery surface with ``subprocess.run`` recorded.

    The backend is built with a *string* socket so the pinned argv is byte-exact
    on any platform (a ``Path`` would render backslashes on Windows).
    """
    recorder = _TmuxRecorder()
    monkeypatch.setattr("subprocess.run", recorder)
    backend = TmuxBackend(socket_path=_SOCKET, target=_TARGET)
    return TerminalDelivery(backend), recorder


def test_tmux_send_keys_pins_argv(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``send_keys`` emits one ``send-keys -t <target> <key>`` per key, no ``-f``."""
    delivery, recorder = tmux_delivery
    delivery.send_keys(["Escape"])
    assert recorder.calls == [["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "Escape"]]


def test_tmux_clear_composer_pins_two_calls(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """C-a then C-k clear the composer as two separate send-keys calls."""
    delivery, recorder = tmux_delivery
    delivery.send_keys(["C-a"])
    delivery.send_keys(["C-k"])
    assert recorder.calls == [
        ["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "C-a"],
        ["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "C-k"],
    ]


def test_tmux_type_literal_pins_argv(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``type_literal`` emits ``send-keys -l -t <target> <text>`` (slash-command path)."""
    delivery, recorder = tmux_delivery
    delivery.type_literal("/effort high")
    assert recorder.calls == [
        ["tmux", "-S", _SOCKET, "send-keys", "-l", "-t", _TARGET, "/effort high"],
    ]


def test_tmux_kill_pins_kill_session(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``kill`` emits ``kill-session -t <target>`` (not kill-server, not send-keys)."""
    delivery, recorder = tmux_delivery
    delivery.kill()
    assert recorder.calls == [["tmux", "-S", _SOCKET, "kill-session", "-t", _TARGET]]


@pytest.mark.parametrize(
    ("text", "expected_payload"),
    [
        ("hello", b"hello"),
        ("a\nb", b"a\rb"),
        ("a\r\nb", b"a\rb"),
        ("a\rb", b"a\rb"),
        ("a\tb\x1b\x07c", b"a\tbc"),
        ("café", "café".encode()),
    ],
    ids=["plain", "newline", "crlf", "cr", "controls", "utf8"],
)
def test_tmux_paste_pins_buffer_argv_and_payload(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
    text: str,
    expected_payload: bytes,
) -> None:
    """A paste loads the encoded payload then pastes it with -p -d markers.

    Pins the exact ``load-buffer`` / ``paste-buffer`` argv and the CR-encoded
    payload — the anthropics/claude-code#52126 multi-line-collapse guard.
    """
    delivery, recorder = tmux_delivery
    delivery.paste_without_submit(text)

    assert recorder.loaded_payloads == [expected_payload]
    assert recorder.calls[0][:6] == [
        "tmux",
        "-S",
        _SOCKET,
        "load-buffer",
        "-b",
        "omnigent-paste",
    ]
    assert recorder.calls[1] == [
        "tmux",
        "-S",
        _SOCKET,
        "paste-buffer",
        "-p",
        "-d",
        "-b",
        "omnigent-paste",
        "-t",
        _TARGET,
    ]


def test_tmux_paste_payload_matches_encoder() -> None:
    """The delivery payload equals :func:`_tmux_paste_payload_bytes` (the shared encoder)."""
    assert _tmux_paste_payload_bytes("a\nb\r\nc\td\x1b\x00e") == b"a\rb\rc\td" + b"e"


def test_tmux_snapshot_pins_capture_argv_and_returns_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``snapshot`` emits ``capture-pane -t <target> -p`` and returns its stdout."""
    recorder = _TmuxRecorder(capture_output="❯ ready")
    monkeypatch.setattr("subprocess.run", recorder)
    seen: list[list[str]] = []

    def recording_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        seen.append(cmd)
        return recorder(cmd, **kwargs)

    monkeypatch.setattr("subprocess.run", recording_run)
    delivery = TerminalDelivery(TmuxBackend(socket_path=_SOCKET, target=_TARGET))

    assert delivery.snapshot() == "❯ ready"
    assert seen == [["tmux", "-S", _SOCKET, "capture-pane", "-t", _TARGET, "-p"]]


def test_tmux_snapshot_returns_empty_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture failure yields ``""`` (the "not ready yet" signal), never raises."""

    def failing_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(returncode=1, stdout="", stderr="no server running")

    monkeypatch.setattr("subprocess.run", failing_run)
    delivery = TerminalDelivery(TmuxBackend(socket_path=_SOCKET, target=_TARGET))
    assert delivery.snapshot() == ""


def test_tmux_send_keys_raises_with_stderr_on_failure(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """A non-zero tmux exit propagates as RuntimeError carrying the stderr."""
    delivery, recorder = tmux_delivery
    recorder.fail_with("no server running on /tmp/example/tmux.sock")
    with pytest.raises(RuntimeError, match="no server running"):
        delivery.send_keys(["Enter"])


def test_full_delivery_dance_pins_command_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end paste-and-submit dance emits C-a, C-k, load, paste, Enter.

    Mirrors what the migrated ``inject_user_message`` drives: clear (two keys),
    bracketed paste (two buffer calls), then the submit Enter — in that order,
    with the exact tmux argv. The simulated TUI clears its box on Enter so the
    verify loop stops after one submit.
    """
    tui = {"pane": "❯ "}
    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        if "capture-pane" in cmd:
            return SimpleNamespace(returncode=0, stdout=tui["pane"], stderr="")
        if "paste-buffer" in cmd:
            tui["pane"] = "❯ [Pasted text #1 +1 line]"
        if cmd[-1] == "Enter":
            tui["pane"] = "❯ "
        recorded.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    delivery = TerminalDelivery(TmuxBackend(socket_path=_SOCKET, target=_TARGET))

    delivery.send_keys(["C-a"])
    delivery.send_keys(["C-k"])
    delivery.paste_without_submit("do the thing\n")
    delivery.submit_and_verify(
        draft_present=lambda pane: "[Pasted text" in pane,
        error_message="not delivered",
        poll_interval_s=0.0,
        commit_timeout_s=0.5,
        settle_s=0.0,
        verify_timeout_s=0.5,
        retry_interval_s=0.01,
    )

    ops = [c[3] for c in recorded]
    assert ops == ["send-keys", "send-keys", "load-buffer", "paste-buffer", "send-keys"]
    assert recorded[0][-1] == "C-a"
    assert recorded[1][-1] == "C-k"
    assert recorded[-1][-1] == "Enter"


def test_build_prompt_delivery_defaults_to_tmux() -> None:
    """The factory builds a tmux-backed surface by default (POSIX default backend)."""
    delivery = build_prompt_delivery(socket_path=_SOCKET, target=_TARGET)
    assert isinstance(delivery.backend, TmuxBackend)
