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
    Liveness,
    TerminalDelivery,
    TerminalLaunchRequest,
    TmuxBackend,
    TmuxDeliveryStyle,
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


async def test_is_alive_reflects_endpoint_liveness() -> None:
    """``is_alive`` is True while the endpoint exists and False once it is gone.

    The delivery fast-fail gate: a bridge probes this before injecting so a web
    message into an exited TUI raises a clear "restart" error instead of being
    typed into a dead pane.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    assert delivery.is_alive() is True

    backend.push_liveness([Liveness.ENDPOINT_GONE])
    assert delivery.is_alive() is False


async def test_is_alive_true_when_inner_exited_but_endpoint_kept() -> None:
    """A kept-but-dead pane (INNER_EXITED) still counts as a present endpoint."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    backend.push_liveness([Liveness.INNER_EXITED])
    assert delivery.is_alive() is True


async def test_send_keys_repeated_presses_key_n_times() -> None:
    """``send_keys_repeated`` delivers one key the requested number of times."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.send_keys_repeated("BSpace", 3)

    # The fake's default repeat expresses the burst as three presses in one call.
    assert delivery.snapshot().count(key_marker("BSpace")) == 3


async def test_send_keys_atomic_delivers_all_keys_in_one_call() -> None:
    """``send_keys_atomic`` delivers every key, recorded as a single call.

    The atomic multi-key primitive (goose's packed ``Down Down Enter``): the fake
    records one ``sent_keys`` entry holding the whole sequence, and every key's
    marker lands on the screen in order.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.send_keys_atomic(["Down", "Down", "Enter"])

    assert backend.sent_keys == [["Down", "Down", "Enter"]]
    screen = delivery.snapshot()
    assert screen.count(key_marker("Down")) == 2
    assert key_marker("Enter") in screen


async def test_send_keys_repeated_zero_count_is_noop() -> None:
    """A non-positive count sends nothing (an empty burst emits no key)."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.send_keys_repeated("BSpace", 0)

    assert backend.sent_keys == []
    assert key_marker("BSpace") not in delivery.snapshot()


async def test_submit_once_submits_exactly_one_enter_no_retry() -> None:
    """``submit_once`` sends a single Enter and never retries, even if the draft stays.

    Unlike :meth:`submit_and_verify`, this does not verify the draft left the box:
    the cursor/goose/kimi TUIs submit on one Enter and a second would submit
    twice, so a still-present draft must NOT trigger another Enter or an error.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.submit_once(
        draft_present=lambda _pane: True,  # never clears — would make verify retry/raise
        poll_interval_s=0.0,
        commit_timeout_s=0.05,
        settle_s=0.0,
    )

    assert backend.sent_keys.count(["Enter"]) == 1


async def test_submit_once_waits_for_commit_then_submits() -> None:
    """``submit_once`` waits for the draft to land, then submits one Enter.

    A submit key that arrives mid-paste is folded into the draft as a newline, so
    the commit wait polls until the paste is visible before the single Enter.
    """
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)
    # The draft is invisible for the first two snapshots, then commits.
    backend.push_snapshots(["❯ ", "❯ ", "❯ do the thing"])

    delivery.submit_once(
        draft_present=lambda pane: "do the thing" in pane,
        poll_interval_s=0.0,
        commit_timeout_s=0.5,
        settle_s=0.0,
    )

    assert backend.sent_keys.count(["Enter"]) == 1


async def test_submit_once_submits_blind_when_no_predicate() -> None:
    """With no predicate (unidentifiable draft), submit once without polling."""
    backend = await _launched_fake()
    delivery = TerminalDelivery(backend)

    delivery.submit_once(
        draft_present=None,
        poll_interval_s=0.0,
        commit_timeout_s=0.5,
        settle_s=0.0,
    )

    assert backend.sent_keys == [["Enter"]]


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


def test_tmux_type_literal_flag_order_before_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """``literal_flag_before_target`` swaps the literal argv to ``-t <target> -l``.

    Preserves cursor's ``/model`` picker order (``send-keys -t <target> -l``); the
    default keeps claude's ``-l -t <target>`` (pinned by the test above).
    """
    recorder = _TmuxRecorder()
    monkeypatch.setattr("subprocess.run", recorder)
    style = TmuxDeliveryStyle(literal_flag_before_target=True)
    delivery = TerminalDelivery(
        TmuxBackend(socket_path=_SOCKET, target=_TARGET, delivery_style=style)
    )

    delivery.type_literal("/model gpt-5.2")

    assert recorder.calls == [
        ["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "-l", "/model gpt-5.2"],
    ]


def test_tmux_send_keys_atomic_pins_single_call(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``send_keys_atomic`` emits ONE ``send-keys -t <target> k1 k2 …`` call.

    Byte-identical to goose's packed permission-dialog keystroke — a single
    atomic client command, not one call per key.
    """
    delivery, recorder = tmux_delivery
    delivery.send_keys_atomic(["Down", "Down", "Enter"])
    assert recorder.calls == [
        ["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "Down", "Down", "Enter"],
    ]


def test_tmux_send_keys_atomic_empty_emits_no_command(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """An empty key sequence emits no tmux command."""
    delivery, recorder = tmux_delivery
    delivery.send_keys_atomic([])
    assert recorder.calls == []


def test_tmux_kill_pins_kill_session(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``kill`` emits ``kill-session -t <target>`` (not kill-server, not send-keys)."""
    delivery, recorder = tmux_delivery
    delivery.kill()
    assert recorder.calls == [["tmux", "-S", _SOCKET, "kill-session", "-t", _TARGET]]


def test_tmux_is_alive_pins_has_session_argv(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``is_alive`` emits ``has-session -t <target>`` and reads True on exit 0."""
    delivery, recorder = tmux_delivery
    assert delivery.is_alive() is True
    assert recorder.calls == [["tmux", "-S", _SOCKET, "has-session", "-t", _TARGET]]


def test_tmux_is_alive_false_on_nonzero_never_raises(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """A non-zero ``has-session`` (endpoint gone) reads as False, never raises."""
    delivery, recorder = tmux_delivery
    recorder.fail_with("no server running")
    assert delivery.is_alive() is False


def test_tmux_send_keys_repeated_pins_n_argv(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """``send_keys_repeated`` emits one ``send-keys -t <target> -N <count> <key>``.

    Byte-identical to cursor's composer-clear burst: a single native repeat, not
    N separate presses.
    """
    delivery, recorder = tmux_delivery
    delivery.send_keys_repeated("BSpace", 200)
    assert recorder.calls == [
        ["tmux", "-S", _SOCKET, "send-keys", "-t", _TARGET, "-N", "200", "BSpace"],
    ]


def test_tmux_send_keys_repeated_zero_emits_no_command(
    tmux_delivery: tuple[TerminalDelivery, _TmuxRecorder],
) -> None:
    """A non-positive repeat count emits no tmux command at all."""
    delivery, recorder = tmux_delivery
    delivery.send_keys_repeated("BSpace", 0)
    assert recorder.calls == []


def test_tmux_delivery_style_paste_buffer_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom ``TmuxDeliveryStyle`` paste buffer name reaches the paste argv.

    Proves a migrated bridge keeps its own per-harness buffer name
    (``omnigent-cursor-paste`` etc.) rather than being normalized onto claude's.
    """
    recorder = _TmuxRecorder()
    monkeypatch.setattr("subprocess.run", recorder)
    style = TmuxDeliveryStyle(paste_buffer="omnigent-cursor-paste")
    delivery = TerminalDelivery(
        TmuxBackend(socket_path=_SOCKET, target=_TARGET, delivery_style=style)
    )

    delivery.paste_without_submit("hi")

    assert recorder.calls[0][:6] == [
        "tmux",
        "-S",
        _SOCKET,
        "load-buffer",
        "-b",
        "omnigent-cursor-paste",
    ]
    assert recorder.calls[1] == [
        "tmux",
        "-S",
        _SOCKET,
        "paste-buffer",
        "-p",
        "-d",
        "-b",
        "omnigent-cursor-paste",
        "-t",
        _TARGET,
    ]


def test_tmux_delivery_style_capture_flag_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """``capture_flag_before_target`` swaps the snapshot argv to ``-p -t <target>``.

    Preserves cursor/goose/kimi's ``capture-pane -p -t <target>`` order (claude's
    default is ``-t <target> -p``).
    """
    seen: list[list[str]] = []

    def recording_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout="pane", stderr="")

    monkeypatch.setattr("subprocess.run", recording_run)
    style = TmuxDeliveryStyle(capture_flag_before_target=True)
    delivery = TerminalDelivery(
        TmuxBackend(socket_path=_SOCKET, target=_TARGET, delivery_style=style)
    )

    assert delivery.snapshot() == "pane"
    assert seen == [["tmux", "-S", _SOCKET, "capture-pane", "-p", "-t", _TARGET]]


def test_tmux_delivery_style_command_timeout_threads_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The style's ``command_timeout_s`` is the per-command subprocess timeout.

    cursor/goose deliver with a 10.0s timeout vs claude's 5.0s; the value must
    reach ``subprocess.run(timeout=...)`` on every delivery command.
    """
    timeouts: list[float] = []

    def recording_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        timeouts.append(kwargs.get("timeout"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", recording_run)
    style = TmuxDeliveryStyle(command_timeout_s=10.0)
    delivery = TerminalDelivery(
        TmuxBackend(socket_path=_SOCKET, target=_TARGET, delivery_style=style)
    )

    delivery.send_keys(["Enter"])
    assert timeouts == [10.0]


def test_build_prompt_delivery_threads_tmux_style() -> None:
    """The factory forwards ``tmux_delivery_style`` to the tmux backend."""
    style = TmuxDeliveryStyle(
        paste_buffer="omnigent-goose-paste",
        capture_flag_before_target=True,
        literal_flag_before_target=True,
        command_timeout_s=10.0,
    )
    delivery = build_prompt_delivery(
        socket_path=_SOCKET, target=_TARGET, tmux_delivery_style=style
    )
    assert isinstance(delivery.backend, TmuxBackend)
    assert delivery.backend._delivery_style is style


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
