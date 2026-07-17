"""I/O-operation tests for :class:`~omnigent.inner.terminal.HerdrBackend` (#12).

Complements the lifecycle tests (:mod:`tests.inner.test_terminal_herdr_lifecycle`)
and the full conformance opt-in
(:mod:`tests.inner.test_terminal_backend_conformance_herdr`) with the quirk
regressions and native-signal logic #12 adds: the min-line-count snapshot
workaround, the plain/ANSI capture surface, local scrollback tailing, the full
key-translation table, the busy-state truth table (including the lying-idle
corroboration and the no-signal→unknown degradation), and composer
input-readiness.

Drives the real ``HerdrBackend`` against the deterministic, stdlib-only fake
herdr in :mod:`tests.inner._fake_herdr` via :envvar:`OMNIGENT_HERDR_BIN`. No real
herdr binary or session is ever touched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from omnigent.inner.terminal import HerdrBackend, TerminalLaunchRequest
from tests.inner import _fake_herdr
from tests.inner._fake_herdr import ANSI_MARKER, key_marker, read_log

_FAKE_PATH = Path(_fake_herdr.__file__).resolve()


@dataclass
class _HerdrEnv:
    """Handle over an installed fake herdr: a backend factory plus log access."""

    tmp_path: Path
    log_path: Path
    monkeypatch: pytest.MonkeyPatch

    def make_backend(self, target: str = "main", *, name: str = "herdr.sock") -> HerdrBackend:
        """Return a fresh, unlaunched backend bound to a private endpoint."""
        return HerdrBackend(socket_path=self.tmp_path / name, target=target)

    def read_invocations(self) -> list[list[str]]:
        """Return every logged herdr ``pane read`` invocation's argv."""
        return [argv for argv in read_log(self.log_path) if argv[2:4] == ["pane", "read"]]


@pytest.fixture
def herdr_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _HerdrEnv:
    """Install the fake herdr CLI and point the backend's env at it."""
    log_path = tmp_path / "herdr-invocations.log"
    monkeypatch.setenv(HerdrBackend.BIN_ENV_VAR, json.dumps([sys.executable, str(_FAKE_PATH)]))
    monkeypatch.setenv(_fake_herdr.STATE_DIR_ENV_VAR, str(tmp_path / "herdr-state"))
    monkeypatch.setenv(_fake_herdr.LOG_ENV_VAR, str(log_path))
    monkeypatch.setenv(_fake_herdr.PROTOCOL_ENV_VAR, _fake_herdr.DEFAULT_PROTOCOL)
    monkeypatch.setenv(_fake_herdr.VERSION_ENV_VAR, _fake_herdr.DEFAULT_VERSION)
    return _HerdrEnv(tmp_path=tmp_path, log_path=log_path, monkeypatch=monkeypatch)


def _request(command: list[str]) -> TerminalLaunchRequest:
    """Build a launch request against the fake."""
    return TerminalLaunchRequest(command=command, cwd=".", env={})


async def _seed_screen(backend: HerdrBackend, lines: list[str]) -> None:
    """Fill the pane screen with *lines* (joined by newlines) via a paste."""
    await backend.send_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Capture: min-line-count workaround, scrollback tailing, ANSI passthrough
# ---------------------------------------------------------------------------


async def test_scrollback_capture_over_fetches_then_tails_locally(herdr_env: _HerdrEnv) -> None:
    """A small scrollback request over-fetches (min-line workaround), then tails.

    The fake models the historic small-N empty-read quirk (a ``--lines`` below a
    threshold returns EMPTY). Because the backend always fetches at least
    :data:`HerdrBackend._SNAPSHOT_MIN_FETCH_LINES`, the capture never comes back
    empty; it is then narrowed to the requested size locally.
    """
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await _seed_screen(backend, [f"line{i}" for i in range(20)])

        snapshot = await backend.capture(scrollback=5)

        # Non-empty (the workaround defeated the small-N quirk) and tailed to 5.
        assert snapshot == "line15\nline16\nline17\nline18\nline19"
        # The wire read asked for the large floor, not the small requested size.
        read_argv = herdr_env.read_invocations()[-1]
        assert "--lines" in read_argv
        fetched = int(read_argv[read_argv.index("--lines") + 1])
        assert fetched == HerdrBackend._SNAPSHOT_MIN_FETCH_LINES
        assert fetched >= _fake_herdr.SMALL_N_EMPTY_THRESHOLD
    finally:
        await backend.close()


async def test_scrollback_capture_uses_recent_unwrapped_source(herdr_env: _HerdrEnv) -> None:
    """A scrollback>0 read requests logical (recent-unwrapped) lines."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.capture(scrollback=10)
        read_argv = herdr_env.read_invocations()[-1]
        assert read_argv[read_argv.index("--source") + 1] == "recent-unwrapped"
    finally:
        await backend.close()


async def test_visible_capture_uses_visible_source_and_ignores_lines(herdr_env: _HerdrEnv) -> None:
    """A scrollback=0 read is the whole visible viewport (no --lines tail)."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await _seed_screen(backend, [f"row{i}" for i in range(8)])
        snapshot = await backend.capture()  # scrollback=0
        # The entire viewport comes back, not a tail.
        assert snapshot == "\n".join(f"row{i}" for i in range(8))
        read_argv = herdr_env.read_invocations()[-1]
        assert read_argv[read_argv.index("--source") + 1] == "visible"
        assert "--lines" not in read_argv
    finally:
        await backend.close()


async def test_ansi_capture_preserves_escapes_plain_does_not(herdr_env: _HerdrEnv) -> None:
    """capture(ansi=True) uses --format ansi and passes SGR escapes through."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("colored output")

        ansi = await backend.capture(ansi=True)
        assert ANSI_MARKER in ansi
        assert "colored output" in ansi
        # The wire read selected the ANSI format.
        read_argv = herdr_env.read_invocations()[-1]
        assert read_argv[read_argv.index("--format") + 1] == "ansi"

        plain = await backend.capture(ansi=False)
        assert "\x1b[" not in plain
        assert "colored output" in plain
    finally:
        await backend.close()


async def test_capture_still_strips_crlf_on_ansi(herdr_env: _HerdrEnv) -> None:
    """CRLF normalization holds on the ANSI path too (no bare CR survives)."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("alpha\nbeta")
        assert "\r" not in await backend.capture(ansi=True)
    finally:
        await backend.close()


def test_fake_models_small_n_empty_read_quirk() -> None:
    """The fake returns EMPTY below the small-N threshold, content at/above it.

    Documents the quirk the backend's over-fetch defends against, so the
    workaround test above is meaningful rather than vacuous.
    """
    pane = {"screen": "\n".join(f"l{i}" for i in range(300))}
    below = _fake_herdr._rendered_read(
        pane, ["pane", "read", "p1", "--source", "recent-unwrapped", "--lines", "10"]
    )
    at_floor = _fake_herdr._rendered_read(
        pane,
        ["pane", "read", "p1", "--source", "recent-unwrapped", "--lines", "500"],
    )
    assert below == ""  # small-N empty-read quirk
    assert at_floor.splitlines()[-1] == "l299"  # over-fetch returns real content


# ---------------------------------------------------------------------------
# Key translation: the full table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("neutral", "expected"),
    [
        ("C-c", "ctrl+c"),
        ("C-a", "ctrl+a"),
        ("M-x", "alt+x"),
        ("M-b", "alt+b"),
        ("S-x", "shift+x"),
        ("BSpace", "Backspace"),
        ("BTab", "shift+tab"),
        ("Enter", "Enter"),
        ("Escape", "Escape"),
        ("Tab", "Tab"),
        ("Space", "Space"),
        ("Up", "Up"),
        ("Down", "Down"),
        ("Left", "Left"),
        ("Right", "Right"),
        ("F1", "F1"),
    ],
)
def test_translate_key_supported(neutral: str, expected: str) -> None:
    """Every supported neutral key maps to its herdr token."""
    assert HerdrBackend._translate_key(neutral) == expected


@pytest.mark.parametrize(
    "neutral",
    ["Home", "End", "PageUp", "PageDown", "Delete", "Insert", "PPage", "NPage", "DC", "IC"],
)
def test_translate_key_unsupported_is_skipped(neutral: str) -> None:
    """Every key herdr rejects is skipped (None), incl. tmux aliases."""
    assert HerdrBackend._translate_key(neutral) is None


async def test_send_keys_delivers_full_range_translated(herdr_env: _HerdrEnv) -> None:
    """A mixed key batch reaches the pane in herdr's syntax, unsupported dropped."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_keys(["BTab", "C-a", "Backspace", "Home", "Escape"])
        snapshot = await backend.capture()
        assert key_marker("shift+tab") in snapshot  # BTab
        assert key_marker("ctrl+a") in snapshot  # C-a
        assert key_marker("Backspace") in snapshot  # passthrough
        assert key_marker("Escape") in snapshot
        assert key_marker("Home") not in snapshot  # unsupported, skipped
    finally:
        await backend.close()


async def test_send_keys_all_unsupported_is_noop(herdr_env: _HerdrEnv) -> None:
    """A batch that translates to nothing sends no send-keys command."""
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_keys(["Home", "End", "Delete"])
        snapshot = await backend.capture()
        for key in ("Home", "End", "Delete"):
            assert key_marker(key) not in snapshot
        # No send-keys was logged at all.
        assert not [a for a in read_log(herdr_env.log_path) if a[2:4] == ["pane", "send-keys"]]
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Busy state: the corroboration truth table
# ---------------------------------------------------------------------------


async def test_busy_state_native_working_is_busy(herdr_env: _HerdrEnv) -> None:
    """A native 'working' status is authoritatively busy (no diff needed)."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "working")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.busy_state() is True
    finally:
        await backend.close()


async def test_busy_state_native_blocked_is_busy(herdr_env: _HerdrEnv) -> None:
    """A native 'blocked' status (paused mid-turn) is busy."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "blocked")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.busy_state() is True
    finally:
        await backend.close()


async def test_busy_state_native_idle_stable_screen_is_idle(herdr_env: _HerdrEnv) -> None:
    """Native idle with an unchanging screen reads idle."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "idle")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("prompt> ")
        assert await backend.busy_state() is False  # establishes the prior snapshot
        assert await backend.busy_state() is False  # unchanged screen → still idle
    finally:
        await backend.close()


async def test_busy_state_lying_idle_is_corrected_by_output_diff(herdr_env: _HerdrEnv) -> None:
    """Native idle but a still-changing screen → busy (the lying-idle case)."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "idle")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("thinking...")
        assert await backend.busy_state() is False  # prime the diff baseline
        await backend.send_text(" still working")  # screen changes under native idle
        assert await backend.busy_state() is True
    finally:
        await backend.close()


async def test_busy_state_no_signal_no_prior_is_unknown(herdr_env: _HerdrEnv) -> None:
    """No native signal and no prior snapshot → None (acceptance criterion 3)."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "unknown")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.busy_state() is None
    finally:
        await backend.close()


async def test_busy_state_gone_pane_degrades_without_raising(herdr_env: _HerdrEnv) -> None:
    """A dead pane makes busy_state degrade to None, not raise (gone-pane hardening).

    ``busy_state`` corroborates the native status with a screen capture, which
    raises once herdr has destroyed the pane. The codex-path caller polls
    busy_state on a possibly-just-exited terminal, so a gone pane must yield a
    graceful ``None`` (no usable signal) rather than propagating the capture
    error.
    """
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    await backend.close()  # the pane (and workspace) are now gone
    # Must not raise; no output to diff and no live native status → None.
    assert await backend.busy_state() is None


async def test_busy_state_no_native_falls_back_to_output_diff(herdr_env: _HerdrEnv) -> None:
    """With no native signal, output-diff alone drives busy/idle once primed."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "unknown")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        await backend.send_text("output")
        assert await backend.busy_state() is None  # first call: no prior → unknown
        assert await backend.busy_state() is False  # primed, unchanged → idle
        await backend.send_text(" more")
        assert await backend.busy_state() is True  # changed → busy
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Input readiness
# ---------------------------------------------------------------------------


async def test_input_ready_idle_is_ready(herdr_env: _HerdrEnv) -> None:
    """An alive pane with a native idle agent is ready for a new prompt."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "idle")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.input_ready() is True
    finally:
        await backend.close()


async def test_input_ready_working_is_not_ready(herdr_env: _HerdrEnv) -> None:
    """A pane whose agent is mid-turn is not ready."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "working")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.input_ready() is False
    finally:
        await backend.close()


async def test_input_ready_unknown_agent_is_none(herdr_env: _HerdrEnv) -> None:
    """No native agent signal on a live pane → unknown readiness."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "unknown")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    try:
        assert await backend.input_ready() is None
    finally:
        await backend.close()


async def test_input_ready_gone_endpoint_is_not_ready(herdr_env: _HerdrEnv) -> None:
    """A gone endpoint is never ready (False), never unknown."""
    herdr_env.monkeypatch.setenv(_fake_herdr.AGENT_STATUS_ENV_VAR, "idle")
    backend = herdr_env.make_backend()
    await backend.launch(_request(["sleep", "1000000"]))
    await backend.close()
    assert await backend.input_ready() is False


# ---------------------------------------------------------------------------
# Default seam behavior: other backends give no native signal
# ---------------------------------------------------------------------------


async def test_busy_and_ready_default_to_none_on_base_backend() -> None:
    """The ABC defaults return None so tmux/fake are behaviorally untouched."""
    from tests.inner.fake_terminal_backend import FakeBackend

    backend = FakeBackend(socket_path=Path("unused.sock"), target="main")
    assert await backend.busy_state() is None
    assert await backend.input_ready() is None
