"""Terminal environment: managed tmux sessions with optional OS environments.

Each terminal instance runs a command in its own tmux server (isolated socket)
with optional filesystem isolation (fork) and sandboxing (bwrap/seccomp).
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

from omnigent._platform import IS_WINDOWS
from omnigent.runner.identity import strip_runner_auth_secrets

from . import _proc
from .datamodel import OSEnvSandboxSpec, OSEnvSpec, TerminalEnvSpec
from .egress import EgressProxyHandle, apply_egress_env, start_egress_proxy
from .os_env import (
    OSEnvironment,
    _copy_tree,
    create_os_environment,
)
from .sandbox import (
    SandboxPolicy,
    cleanup_private_tmpdir,
    create_exec_launcher,
    create_private_tmpdir,
    resolve_sandbox,
    with_additional_write_roots,
    with_denied_unix_sockets,
)

# Heterogeneous JSON-shaped result returned by :meth:`TerminalInstance.send`
# and :meth:`TerminalInstance.read`. In practice the dicts carry a mix of
# ``{"status": str}``, ``{"error": str}``, and ``{"terminal": str, "screen":
# str, "scrollback_lines": int}`` — a TypedDict union would spread across
# every caller (session.py's ``_terminal_send`` / ``_terminal_read``) so we
# keep the boundary open and let those callers pass it through as a
# ``ToolResult``.
TerminalResult: TypeAlias = dict[str, Any]  # type: ignore[explicit-any]

logger = logging.getLogger(__name__)

_TMUX_CONFIG_PATH = os.devnull
_TMUX_CONVERSATION_LINK_OPTION = "@omnigent-conversation-link"

# Web-terminal attach transports. ``pty`` forks a full ``tmux attach`` client
# and streams the rendered screen (see terminals/ws_bridge.py); ``control``
# attaches a ``tmux -C`` control-mode client and streams per-pane ``%output``
# so the browser xterm owns scrollback + selection (see
# terminals/control_bridge.py). Both speak the identical browser wire protocol
# so they are interchangeable per attach.
TERMINAL_TRANSPORT_PTY = "pty"
TERMINAL_TRANSPORT_CONTROL = "control"
TERMINAL_TRANSPORT_SNAPSHOT = "snapshot"
TERMINAL_TRANSPORT_STREAM = "stream"
_VALID_TERMINAL_TRANSPORTS = frozenset(
    {
        TERMINAL_TRANSPORT_PTY,
        TERMINAL_TRANSPORT_CONTROL,
        TERMINAL_TRANSPORT_SNAPSHOT,
        TERMINAL_TRANSPORT_STREAM,
    }
)
# Values that select the PTY path in the config file, beyond the canonical
# ``pty`` name — the common falsy spellings so ``transport: false`` / ``: off``
# reads as PTY. Any other value (including ``control`` and truthy spellings)
# falls through to the control default.
_TRANSPORT_PTY_ALIASES = frozenset({TERMINAL_TRANSPORT_PTY, "0", "false", "no", "off"})
# Config-file location for the global default (``~/.omnigent/config.yaml``,
# honoring ``OMNIGENT_CONFIG_HOME`` for test isolation — same resolution the
# runner and CLI use). The transport lives under the ``terminal:`` table as
# ``terminal.transport``.
_CONFIG_HOME_ENV_VAR = "OMNIGENT_CONFIG_HOME"
_TERMINAL_CONFIG_TABLE = "terminal"
_TERMINAL_TRANSPORT_CONFIG_KEY = "transport"
# Config-file key and env-var override for the multiplexer backend selection.
# The backend lives under the same ``terminal:`` table as
# ``terminal.backend``; the env var takes precedence over the config file (but
# both yield to an explicit per-terminal spec). See
# :func:`resolve_terminal_backend_name`.
_TERMINAL_BACKEND_CONFIG_KEY = "backend"
_TERMINAL_BACKEND_ENV_VAR = "OMNIGENT_TERMINAL_BACKEND"


def _global_config_path() -> Path:
    """Return the global Omnigent config path visible to this process.

    Mirrors :func:`omnigent.runner._entry._runner_config_path` (kept local to
    avoid an inner→runner import): honors :envvar:`OMNIGENT_CONFIG_HOME` for
    test isolation and subprocess consistency, else ``~/.omnigent/config.yaml``.

    :returns: Config path, e.g. ``Path("~/.omnigent/config.yaml")``.
    """
    config_home = os.environ.get(_CONFIG_HOME_ENV_VAR)
    if config_home:
        return Path(config_home).expanduser() / "config.yaml"
    return Path.home() / ".omnigent" / "config.yaml"


def _global_terminal_transport_default() -> str:
    """Resolve the process-wide default web-terminal transport from config.

    Reads ``terminal.transport`` from ``~/.omnigent/config.yaml`` at call time
    (not import time) so a config edit takes effect on the next attach without
    a restart, and tests can point :envvar:`OMNIGENT_CONFIG_HOME` at a scratch
    config. Control mode is the default; set ``terminal.transport`` to a PTY
    alias to opt out. Recognized values (case-insensitive):

    - Missing / ``control`` / ``1`` / ``true`` / ``yes`` / ``on`` → ``control``.
    - ``pty`` / ``0`` / ``false`` / ``no`` / ``off`` → ``pty``.
    - Anything else → ``control`` (the default), so a typo can't strand an
      operator on the legacy path.

    A missing file, unreadable file, malformed YAML, or missing key all fall
    back to the control default — reading the transport must never crash an
    attach.

    :returns: ``"control"`` or ``"pty"``.
    """
    raw = _read_terminal_transport_config()
    if raw is not None and raw.strip().lower() in _TRANSPORT_PTY_ALIASES:
        return TERMINAL_TRANSPORT_PTY
    return TERMINAL_TRANSPORT_CONTROL


def _read_terminal_transport_config() -> str | None:
    """Read ``terminal.transport`` from the global config, or ``None``.

    Best-effort: any failure (missing/unreadable file, non-mapping YAML,
    absent table/key, non-string/bool value) returns ``None`` so the caller
    uses the control default. Never raises.

    An unquoted YAML ``true``/``false`` parses as a real bool rather than a
    string, so a bool value is normalized to its lowercase string spelling
    before returning — ``terminal.transport: false`` still selects the PTY
    alias in :data:`_TRANSPORT_PTY_ALIASES`.

    :returns: The raw configured transport string, or ``None`` when unset.
    """
    import yaml

    path = _global_config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict):
        return None
    table = raw.get(_TERMINAL_CONFIG_TABLE)
    if not isinstance(table, dict):
        return None
    value = table.get(_TERMINAL_TRANSPORT_CONFIG_KEY)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else None


def resolve_terminal_transport(
    *,
    override: str | None = None,
    spec_transport: str | None = None,
    backend_capabilities: TerminalBackendCapabilities | None = None,
) -> str:
    """Pick the web-terminal attach transport for one attach.

    Resolution order (first match wins):

    1. ``override`` — a per-attach ``?transport=control|pty`` query, letting a
       dev A/B two open terminals side by side right now.
    2. ``spec_transport`` — the per-terminal / per-harness
       :attr:`TerminalEnvSpec.terminal_transport`, the gradual-rollout dial.
    3. The global default from :func:`_global_terminal_transport_default`
       — ``control`` unless ``terminal.transport`` in ``~/.omnigent/config.yaml``
       opts out to ``pty``.

    Unrecognized values at any level are ignored (fall through) so a stray
    query string can never break an attach.

    :param override: Per-attach transport request, e.g. ``"control"``.
    :param spec_transport: The terminal spec's declared transport, or ``None``.
    :returns: ``"control"`` or ``"pty"``.
    """
    advertised = (
        backend_capabilities.attach_transports
        if backend_capabilities is not None
        else frozenset({TERMINAL_TRANSPORT_CONTROL, TERMINAL_TRANSPORT_PTY})
    )
    for candidate in (override, spec_transport):
        if candidate is not None:
            normalized = candidate.strip().lower()
            if normalized in _VALID_TERMINAL_TRANSPORTS and normalized in advertised:
                return normalized
    configured = _global_terminal_transport_default()
    if configured in advertised:
        return configured
    for fallback in (
        TERMINAL_TRANSPORT_CONTROL,
        TERMINAL_TRANSPORT_PTY,
        TERMINAL_TRANSPORT_SNAPSHOT,
        TERMINAL_TRANSPORT_STREAM,
    ):
        if fallback in advertised:
            return fallback
    raise RuntimeError("terminal backend advertises no web attach transports")


_TMUX_START_ON_ATTACH_CHANNEL = "omnigent-start-on-attach"
# Each terminal instance lives in a private tmpdir with this prefix
# (see ``create_terminal_instance``). The owner-pid marker inside it
# records the process that launched the instance so a later startup
# can reap tmux servers whose owner died without graceful shutdown
# (``reap_orphaned_terminals``).
_TERMINAL_DIR_PREFIX = "omnigent-terminal-"
_OWNER_PID_FILENAME = "owner.pid"
# Bound for each ``tmux kill-server`` in the orphan sweep; a wedged
# tmux must not stall runner startup.
_REAP_KILL_TIMEOUT_S = 10.0
# Literal tmux empty option value. Passing this as an argv value clears
# status segments and window formats; it is not an application sentinel.
_TMUX_EMPTY_OPTION_VALUE = ""


def _tmux_command_sequence(commands: list[list[str]]) -> list[str]:
    """
    Flatten tmux commands into one client command sequence.

    Tmux accepts multiple commands in one invocation when separated by
    a literal ``;`` argv. This lets Omnigent configure a fresh
    private server and create the session without writing a tmux conf
    file.

    :param commands: Tmux commands without the leading ``tmux`` argv,
        e.g. ``[["set-option", "-g", "mouse", "on"], ["new-session"]]``.
    :returns: Flattened argv suffix with command separators.
    """
    sequence: list[str] = []
    for command in commands:
        if sequence:
            sequence.append(";")
        sequence.extend(command)
    return sequence


def _tmux_managed_option_commands(
    scrollback: int,
    *,
    allow_passthrough: bool = False,
    keep_alive_after_exit: bool = False,
) -> list[list[str]]:
    """
    Build tmux commands for Omnigent-managed global options.

    :param scrollback: Tmux history limit, e.g. ``10000``.
    :param allow_passthrough: Whether to allow pane programs to send
        passthrough escape sequences to the real attached terminal.
    :param keep_alive_after_exit: When ``True``, keep the private tmux server
        alive after the pane's process exits (see
        :func:`_tmux_session_persistence_commands`). Opt-in because it changes
        the ``has-session``-means-alive contract that liveness probes rely on;
        callers that enable it must use pane-dead-aware liveness checks.
    :returns: List of tmux commands to run before ``new-session``.
    """
    commands = [
        *_tmux_input_option_commands(scrollback),
        *_tmux_lockdown_commands(),
        *_tmux_status_option_commands(),
    ]
    if keep_alive_after_exit:
        commands.extend(_tmux_session_persistence_commands())
    if allow_passthrough:
        commands.append(["set-option", "-g", "allow-passthrough", "on"])
    return commands


def _tmux_session_persistence_commands() -> list[list[str]]:
    """Keep the private tmux server alive when the pane's process exits.

    Each managed terminal runs exactly ONE inner CLI (claude / codex / cursor /
    pi / a shell) in a private, single-pane tmux server. Under tmux's defaults
    (``exit-empty on`` + ``remain-on-exit off``) the instant that CLI exits —
    a crash, ``/exit``, or an environment-specific early exit (issue #540: a
    claude-native sub-agent on WSL2 that renders its prompt then exits) — the
    pane closes, the lone session is destroyed, and the server exits on its
    private socket. Every later control command (send-keys, model / effort
    change, interrupt, stop) then fails with ``no server running`` and the CLI's
    final output is gone, so a single child-process exit becomes an
    unrecoverable, undiagnosable cascade and delegated messages are silently
    lost.

    ``remain-on-exit on`` keeps the dead pane — and therefore the session and
    server — present after the inner process exits, so the socket stays usable
    and the pane's last output stays capturable for diagnostics. The idle
    watcher then reports the exit deterministically by detecting the dead pane
    (see :meth:`TerminalInstance._pane_is_dead`) instead of racing the server's
    disappearance. ``exit-empty off`` is belt-and-suspenders for the case where
    the session is removed without the server being explicitly killed. Both use
    ``-q`` so a tmux too old to know the option does not fail launch;
    :meth:`TerminalInstance.close` still tears the server down unconditionally
    via ``kill-server``, so nothing leaks.

    :returns: Tmux option commands that keep the server alive past inner-CLI
        exit.
    """
    return [
        ["set-option", "-gq", "remain-on-exit", "on"],
        ["set-option", "-sq", "exit-empty", "off"],
    ]


def _tmux_input_option_commands(scrollback: int) -> list[list[str]]:
    """
    Build tmux options for scrollback and pane input behavior.

    ``history-limit`` is generated per terminal because it comes from
    ``TerminalEnvSpec.scrollback``. ``mouse on`` makes the attached web
    terminal scrollable. ``focus-events on`` lets interactive programs
    observe pane focus changes. ``extended-keys`` with CSI-u formatting
    lets programs inside tmux receive Kitty Keyboard Protocol keys such
    as Shift+Enter when the attached terminal supports them. Terminals
    without that protocol ignore tmux's request, and the quiet tmux
    options keep older tmux versions from failing launch. ``escape-time
    0`` prevents pasted ANSI escape bytes from accumulating tmux's
    default delay.

    :param scrollback: Tmux history limit, e.g. ``10000``.
    :returns: Tmux commands configuring pane input and scrollback.
    """
    return [
        ["set-option", "-g", "history-limit", str(scrollback)],
        ["set-option", "-sq", "extended-keys", "on"],
        ["set-option", "-sq", "extended-keys-format", "csi-u"],
        ["set-option", "-g", "mouse", "on"],
        ["set-option", "-g", "focus-events", "on"],
        ["set-option", "-g", "escape-time", "0"],
    ]


def _tmux_lockdown_commands() -> list[list[str]]:
    """
    Build tmux commands that remove user-facing pane/window creation controls.

    Managed terminals must stay inside Omnigent' terminal registry.
    Disabling the prefix table and right-click context menus prevents an
    attached user from creating extra panes, windows, or sessions through
    tmux UI controls. The root-table unbinds are quiet so missing default
    mouse bindings on a tmux version do not fail terminal launch.

    :returns: Tmux commands that disable prefix and creation menus.
    """
    return [
        ["set-option", "-g", "prefix", "None"],
        ["set-option", "-g", "prefix2", "None"],
        ["unbind-key", "-a", "-T", "prefix"],
        ["unbind-key", "-q", "-T", "root", "MouseDown3Pane"],
        ["unbind-key", "-q", "-T", "root", "M-MouseDown3Pane"],
        ["unbind-key", "-q", "-T", "root", "MouseDown3Status"],
        ["unbind-key", "-q", "-T", "root", "M-MouseDown3Status"],
        ["unbind-key", "-q", "-T", "root", "MouseDown3StatusLeft"],
        ["unbind-key", "-q", "-T", "root", "M-MouseDown3StatusLeft"],
    ]


def _tmux_status_option_commands() -> list[list[str]]:
    """
    Build tmux status-line options for managed terminals.

    The status line carries the conversation link while hiding tmux's
    window list so users do not see irrelevant tmux chrome for the
    private single-window server.

    :returns: Tmux commands configuring the managed status line.
    """
    return [
        ["set-option", "-g", "status", "on"],
        ["set-option", "-g", "status-style", "fg=default,bg=default"],
        [
            "set-option",
            "-g",
            "status-left",
            f"Omnigent: #{{{_TMUX_CONVERSATION_LINK_OPTION}}}",
        ],
        ["set-option", "-g", "status-left-style", "fg=default,bg=default"],
        ["set-option", "-g", "status-left-length", "200"],
        ["set-option", "-g", "status-right", _TMUX_EMPTY_OPTION_VALUE],
        ["set-option", "-g", "status-right-style", "fg=default,bg=default"],
        ["set-option", "-g", "status-right-length", "0"],
        ["set-option", "-g", "window-status-separator", _TMUX_EMPTY_OPTION_VALUE],
        ["set-window-option", "-g", "window-status-format", _TMUX_EMPTY_OPTION_VALUE],
        [
            "set-window-option",
            "-g",
            "window-status-current-format",
            _TMUX_EMPTY_OPTION_VALUE,
        ],
    ]


# How long the tmux pane must show no changes to be considered idle, and how
# often we poll capture-pane to check. Exposed as module-level constants so
# tests can lower them instead of waiting the full threshold per assertion.
_IDLE_THRESHOLD_SECONDS = 10.0
_IDLE_POLL_INTERVAL_SECONDS = 1.0

# When a web client interacts with the terminal (attach/detach, focus
# in/out, mouse, keystroke, resize — all stamped via
# ``TerminalInstance.note_client_interaction``), the TUI repaints in
# response. Those repaints are client-driven, not agent work, so the idle
# watcher discounts any pane change that lands within this window of the
# last interaction. The window must comfortably exceed the poll interval so
# a repaint that trails its triggering event by a tick (or a browser's
# burst of resizes on attach) is still absorbed. It only suppresses the
# *activity* edge — idle detection is unaffected — so the cost is at most a
# slightly-late ``running`` if the agent starts working within the window
# of an interaction.
_CLIENT_INTERACTION_WINDOW_SECONDS = 0.75

# Substrings that indicate the terminal is waiting for a human response even
# while other cells on the pane keep changing (e.g. Codex's blinking spinner
# glyph during a permission prompt). When any marker has been continuously
# visible in the ANSI-stripped pane capture for _IDLE_MARKER_THRESHOLD_SECONDS,
# the watcher treats the pane as idle. This is an alternative trigger to the
# diff-based one above; both tracks share a single ``idle_notified`` gate so
# at most one ``on_idle`` call fires per idle episode.
#
# Tests monkey-patch this list at the module level by rebinding (not by
# ``.append``/``.clear``), so production code must only read — never mutate —
# this list at runtime. Keep marker substrings short (well under 80 chars)
# so tmux's pane-width line wrapping (``-x 80`` at creation in ``launch``;
# wider once a client attaches) doesn't split them across a newline and
# defeat the substring match.
_IDLE_MARKER_SUBSTRINGS: list[str] = [
    "Press enter to confirm or esc to cancel",
    "1. Yes",
]
# Defaults to the same threshold as the diff path for simplicity. Kept as a
# separate name so tests (and future callers) can tune it independently; any
# future runtime change to _IDLE_THRESHOLD_SECONDS does NOT propagate here.
_IDLE_MARKER_THRESHOLD_SECONDS: float = _IDLE_THRESHOLD_SECONDS

# Bounded join window when stopping a threaded idle watcher. Long enough
# to let a tick that's currently inside ``subprocess.run`` finish, short
# enough that ``close()`` doesn't block the event loop visibly. The
# tmux capture-pane subprocess is the only operation in the loop body
# that can outlast a single Python frame; it normally returns in <50ms.
_IDLE_WATCHER_JOIN_TIMEOUT_S = 1.0

# tmux's client→server protocol rejects any single command larger than
# its 16KB imsg cap — the client exits non-zero with "command too long".
# Literal text typed via ``send-keys -l`` is therefore chunked so each
# invocation stays far under the cap even at 4 UTF-8 bytes per character
# (1024 chars ≤ 4KB packed). tmux writes each invocation's bytes to the
# pane in submission order, so the program sees one contiguous stream.
_SEND_KEYS_LITERAL_CHARS_PER_CALL = 1024

# --- Shared prompt-delivery surface (the "delivery dance") ------------------
#
# The seven TUI-typing native bridges (claude, cursor, goose, hermes, kimi,
# kiro, antigravity) each duplicated a private tmux helper and the full
# delivery dance — clear the composer draft, paste a multi-line prompt WITHOUT
# submitting it, verify via a screen snapshot that the draft landed, then submit
# and verify it left the box. :class:`TerminalDelivery` consolidates that dance
# onto the backend seam so it is written once and every backend (tmux, herdr,
# the in-process fake) hosts it through the same synchronous protocol. The
# bridges are stateless callers running in a worker thread (``asyncio.to_thread``
# with no event loop), so the surface is synchronous throughout — the same
# execution model the private helpers used, which keeps the migrated tmux
# command stream byte-identical.

# How long a single delivery subprocess (a ``tmux`` client command) may run
# before it is treated as failed. Matches the value the per-bridge private
# helpers used, so migrated delivery keeps the same timeout behavior.
_DELIVERY_SEND_TIMEOUT_S = 5.0

# Named tmux paste buffer the bracketed-paste delivery loads into. A fixed name
# (rather than the anonymous top buffer) lets ``paste-buffer -d`` drop exactly
# this buffer after use so no stale copies accumulate server-side.
_TMUX_PASTE_BUFFER = "omnigent-paste"


def _tmux_paste_payload_bytes(text: str) -> bytes:
    r"""Encode *text* as the byte payload for a tmux bracketed paste.

    Returns only the content bytes — ``paste-buffer -p`` wraps them in the
    ``ESC [ 2 0 0 ~`` / ``ESC [ 2 0 1 ~`` markers itself when delivering the
    buffer to the pane. Bytes are mapped so a TUI keeps the paste as editable
    data rather than submitting on each line:

    - ``\r\n`` / lone ``\r`` / ``\n`` all collapse to a single carriage return
      ``0x0d`` — the byte a real paste carries between lines inside the markers,
      so interior newlines stay data instead of becoming per-line submits.
    - ``\t`` becomes ``0x09``.
    - Any other control byte below ``0x20`` is dropped: a stray ``ESC`` (or BEL)
      would otherwise prematurely close the bracketed-paste sequence.
    - Everything else passes through as its UTF-8 bytes.

    This is the consolidated form of the per-bridge private ``_paste_payload_bytes``
    helpers; the tmux delivery backend uses it, and callers append their own
    trailing newline (which absorbs a trailing backslash so it cannot escape the
    submit ``Enter``).

    :param text: Raw text to paste, possibly multi-line, e.g. ``"a\r\nb"``.
    :returns: The normalized content bytes, e.g. ``b"a\rb"``.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    body = bytearray()
    for ch in normalized:
        if ch == "\n":
            body.append(0x0D)
            continue
        if ch == "\t":
            body.append(0x09)
            continue
        if ord(ch) < 0x20:
            continue
        body.extend(ch.encode("utf-8"))
    return bytes(body)


class _IdleDetector:
    """
    Pure state machine for the pane-idle decision.

    One instance per watcher invocation. Drive it by passing a fresh
    pane snapshot to :meth:`tick` once per poll interval; the return
    value indicates whether ``on_idle`` should fire this tick.

    Two parallel tracks share a single ``idle_notified`` gate so at
    most one notification fires per idle episode:

    1. **Marker track:** any substring in :data:`_IDLE_MARKER_SUBSTRINGS`
       that has been continuously visible for
       :data:`_IDLE_MARKER_THRESHOLD_SECONDS` triggers idleness even
       while other cells on the pane keep changing (e.g. a blinking
       spinner under a permission prompt).
    2. **Diff track:** the snapshot bytes have been unchanged for
       :data:`_IDLE_THRESHOLD_SECONDS`.

    Extracted from :meth:`TerminalInstance._idle_watch_loop` so the
    asyncio watcher (legacy inner Session path) and the threading
    watcher (AP ``sys_terminal_launch`` path) share one source of
    truth for the detection logic — refactoring the watcher to add a
    tracker now updates both paths automatically.
    """

    def __init__(self, *, idle_threshold_s: float | None = None) -> None:
        """Initialize per-watcher state.

        Each watcher invocation creates a fresh detector. The diff-track
        idle threshold defaults to the module constant (which tests
        rebind), but a caller can override it per-watcher — the
        claude-native status watcher uses a short threshold (~1s) so the
        session flips to ``idle`` promptly after Claude stops redrawing,
        while the generic terminal-activity watcher keeps the longer
        default.

        :param idle_threshold_s: Per-watcher diff-track idle threshold in
            seconds, e.g. ``1.0``. ``None`` falls back to the module
            constant :data:`_IDLE_THRESHOLD_SECONDS` at each tick (so
            tests that rebind the module constant still take effect).
            Does not affect the marker track, which always uses
            :data:`_IDLE_MARKER_THRESHOLD_SECONDS`.
        """
        self._last_snapshot: str | None = None
        self._last_change_at: float = time.monotonic()
        self._idle_notified: bool = False
        self._marker_first_seen_at: dict[str, float] = {}
        self._marker_notified: dict[str, bool] = {}
        # Per-watcher idle-threshold override; ``None`` means "read the
        # live module constant in ``tick``" so test rebinds still apply.
        self._idle_threshold_s: float | None = idle_threshold_s
        # Set by ``tick`` to whether the pane content changed *this* tick
        # (the diff track's edge). Read by the watcher loop to drive an
        # ``on_activity`` callback — the runner-determined "this terminal's
        # PTY produced output" signal that powers the web activity badge,
        # without any client PTY attach.
        self.changed_this_tick: bool = False

    def tick(self, snapshot: str, suppress_activity: bool = False) -> bool:
        """
        Feed a fresh pane snapshot and report whether idle fired.

        :param snapshot: The pane bytes from ``tmux capture-pane -p
            -e``, e.g. the raw ANSI-laden output of one capture call.
            Marker matching strips ANSI internally; the diff track
            compares the raw bytes verbatim.
        :param suppress_activity: When ``True``, a content change this
            tick is treated as a client-driven repaint (attach/detach
            reflow, focus, mouse, keystroke) rather than agent output: the
            snapshot is re-baselined but does NOT register as activity and
            does NOT reset the idle timer. The caller sets this when a web
            client interacted with the terminal within the recent window
            (see :data:`_CLIENT_INTERACTION_WINDOW_SECONDS`).
        :returns: ``True`` if this tick crosses an idle edge and the
            caller should invoke ``on_idle`` once. ``False`` on every
            subsequent tick of the same idle episode (re-arm requires
            new output that mutates the snapshot).
        """
        now = time.monotonic()
        # Reset the per-tick activity edge; set True below only when the
        # diff track sees the pane content actually change this tick.
        self.changed_this_tick = False
        stripped = _strip_ansi(snapshot) if _IDLE_MARKER_SUBSTRINGS else ""

        # Marker pass 1: update per-marker timers and cleanup absent
        # markers. Cleanup runs for ALL markers before the fire pass
        # so we don't leave stale per-marker state behind when we
        # break out of the fire pass below.
        for marker in _IDLE_MARKER_SUBSTRINGS:
            if marker in stripped:
                self._marker_first_seen_at.setdefault(marker, now)
            else:
                self._marker_first_seen_at.pop(marker, None)
                self._marker_notified.pop(marker, None)

        # Marker pass 2: pick the first eligible marker and fire once.
        # When we fire, mark EVERY currently-present marker as notified
        # so that if the diff track later clears ``idle_notified``
        # (because pane bytes keep changing under a persistent spinner),
        # another currently-visible marker cannot sneak through and
        # fire a second time within the same idle episode.
        if not self._idle_notified:
            for marker in _IDLE_MARKER_SUBSTRINGS:
                if (
                    marker in stripped
                    and not self._marker_notified.get(marker, False)
                    and now - self._marker_first_seen_at[marker] >= _IDLE_MARKER_THRESHOLD_SECONDS
                ):
                    self._idle_notified = True
                    for other in _IDLE_MARKER_SUBSTRINGS:
                        if other in stripped:
                            self._marker_notified[other] = True
                    return True

        # Diff track: shares ``idle_notified`` with the marker track
        # so we never double-fire.
        if self._last_snapshot is None:
            self._last_snapshot = snapshot
            self._last_change_at = now
            return False

        if snapshot != self._last_snapshot:
            self._last_snapshot = snapshot
            if suppress_activity:
                # A web client interacted within the recent window, so this
                # change is a client-driven repaint (attach/detach reflow,
                # focus, mouse, keystroke), not agent output. Re-baseline to
                # the new snapshot, but leave the change timer and idle
                # state untouched so it neither reads as ``running`` nor
                # re-arms an idle edge.
                return False
            self._last_change_at = now
            self._idle_notified = False
            self.changed_this_tick = True
            return False

        if self._idle_notified:
            return False

        visible_notified_marker = any(
            marker in stripped and self._marker_notified.get(marker, False)
            for marker in _IDLE_MARKER_SUBSTRINGS
        )
        idle_threshold_s = (
            self._idle_threshold_s
            if self._idle_threshold_s is not None
            else _IDLE_THRESHOLD_SECONDS
        )
        if now - self._last_change_at >= idle_threshold_s:
            self._idle_notified = True
            # If the diff track fires while an idle marker is visible, treat
            # that marker as having delivered this idle episode too. Otherwise
            # a shell can emit the marker, quiesce long enough for the diff
            # track to fire, repaint the prompt, and then let the still-visible
            # marker fire a duplicate notification before it disappears.
            for marker in _IDLE_MARKER_SUBSTRINGS:
                if marker in stripped:
                    self._marker_notified[marker] = True
            if visible_notified_marker:
                return False
            return True

        return False


def _clone_sandbox_spec(sandbox: OSEnvSandboxSpec | None) -> OSEnvSandboxSpec | None:
    """Deep-copy an :class:`OSEnvSandboxSpec` for a terminal launch.

    Uses :func:`dataclasses.replace` so every scalar/bool field is
    carried through automatically — the previous hand-written
    field-by-field constructor silently dropped fields added after
    it was written (``egress_rules``, ``egress_allow_private_destinations``,
    ``cwd_allow_hidden``, ``env_passthrough``,
    ``cwd_hidden_scan_max_entries``, ``cwd_hidden_scan_overflow``),
    which downgraded terminal sandboxes to "no MITM proxy, default
    env, only ``.venv`` allowed through" even when the YAML defined
    a strict policy. List fields are explicitly cloned to preserve
    the "doesn't mutate the original" invariant covered by
    :func:`test_build_terminal_os_env_spec_does_not_mutate_original_spec`.
    """
    if sandbox is None:
        return None
    return replace(
        sandbox,
        read_paths=list(sandbox.read_paths) if sandbox.read_paths is not None else None,
        write_paths=list(sandbox.write_paths) if sandbox.write_paths is not None else None,
        write_files=list(sandbox.write_files) if sandbox.write_files is not None else None,
        cwd_allow_hidden=(
            list(sandbox.cwd_allow_hidden) if sandbox.cwd_allow_hidden is not None else None
        ),
        env_passthrough=(
            list(sandbox.env_passthrough) if sandbox.env_passthrough is not None else None
        ),
        egress_rules=list(sandbox.egress_rules) if sandbox.egress_rules is not None else None,
    )


def _clone_os_env_spec(spec: OSEnvSpec) -> OSEnvSpec:
    """Deep-copy an :class:`OSEnvSpec` for a terminal launch.

    Uses :func:`dataclasses.replace` (with an explicitly-cloned
    ``sandbox`` via :func:`_clone_sandbox_spec`) so every
    :class:`OSEnvSpec` field — including ``start_in_scratch`` —
    is carried through. The previous hand-written constructor
    omitted ``start_in_scratch``, silently resetting it to
    ``False`` whenever a terminal inherited its parent's os_env.
    """
    return replace(spec, sandbox=_clone_sandbox_spec(spec.sandbox))


# Regex to strip ANSI escape codes from terminal output.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?(?:\x07|\x1b\\)|\x1b[()][AB012]|\x1b\[[\?]?[0-9;]*[hlm]"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from terminal output."""
    return _ANSI_RE.sub("", text)


def _is_utf8_locale_value(value: str | None) -> bool:
    """Whether a locale string names a UTF-8 codeset.

    A POSIX locale looks like ``language[_TERRITORY][.codeset][@modifier]``;
    the codeset after the dot is what selects the encoding (``en_US.UTF-8``,
    ``C.UTF-8``). Bare ``C`` / ``POSIX`` and empty values are not UTF-8.
    Matching is case- and separator-insensitive (``utf8`` == ``UTF-8``).

    :param value: A locale string such as ``"C.UTF-8"``, or ``None``.
    :returns: ``True`` when the codeset is UTF-8.
    """
    if not value:
        return False
    codeset = value.split("@", 1)[0]
    codeset = codeset.rsplit(".", 1)[-1] if "." in codeset else ""
    return codeset.replace("-", "").lower() == "utf8"


def _has_utf8_locale(env: dict[str, str]) -> bool:
    """Whether the env already carries a UTF-8 signal the TUI CLIs honor.

    The CLIs that mis-decode (opencode/pi/hermes) read ``LC_ALL`` / ``LANG``
    directly rather than calling ``setlocale``, so only those two vars count
    here; a UTF-8 ``LC_CTYPE`` alone does not help them. Per POSIX precedence
    a non-empty ``LC_ALL`` overrides ``LANG``.

    :param env: The prospective terminal spawn environment.
    :returns: ``True`` when the effective ``LC_ALL``/``LANG`` names UTF-8.
    """
    lc_all = env.get("LC_ALL")
    if lc_all:
        return _is_utf8_locale_value(lc_all)
    return _is_utf8_locale_value(env.get("LANG"))


def _apply_utf8_locale_default(env: dict[str, str]) -> None:
    """Force ``LANG=LC_ALL=C.UTF-8`` when the env lacks a UTF-8 locale signal.

    Mutates ``env`` in place. No-op on Windows (tmux terminals are POSIX-only)
    and when the operator already supplied a UTF-8 ``LC_ALL``/``LANG`` (that
    value is preserved). A pinned non-UTF-8 ``LC_ALL`` (e.g. ``C``) is
    corrected. ``C.UTF-8`` is chosen because it needs no locale archive and so
    is present on minimal container images where ``en_US.UTF-8`` is not.

    :param env: The terminal spawn environment to normalize.
    """
    if IS_WINDOWS:
        return
    if _has_utf8_locale(env):
        return
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"


def _tmux_available() -> bool:
    """Check if tmux is installed."""
    return shutil.which("tmux") is not None


def _process_alive(pid: int) -> bool:
    """
    Return whether a process with *pid* currently exists.

    Used by the orphan sweep as the owner-death check. The check is
    conservative in the dangerous direction: ``ProcessLookupError`` is a
    definitive "gone", while a reused pid (or one owned by another user,
    which raises ``PermissionError``) reads as alive and merely defers
    the reap to a later sweep — it can never kill a live owner's
    terminal.

    :param pid: Process id recorded at instance creation,
        e.g. ``48213``.
    :returns: ``True`` when a process with that pid exists.
    """
    # POSIX uses ``os.kill(pid, 0)`` so a killed-but-not-yet-reaped zombie
    # still counts as present (matches ``process_manager._pid_alive``). The
    # ``_proc.process_alive`` psutil probe treats a zombie as gone and can
    # transiently miss a live process, which raced the orphan sweep against a
    # just-exited owner. ``os.kill(pid, 0)`` can't be used on Windows (maps to
    # TerminateProcess and would kill the target), so fall back to psutil there.
    if IS_WINDOWS:
        return _proc.process_alive(pid)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminals_tmp_root() -> Path:
    """
    Return the directory scanned for terminal instance dirs.

    Indirection point so tests can retarget the orphan sweep at a
    scratch directory without monkeypatching the process-wide
    ``tempfile`` module (see omnigent-testing rule 14).

    :returns: The system temp directory, e.g. ``Path("/tmp")``.
    """
    return Path(tempfile.gettempdir())


def reap_orphaned_terminals() -> int:
    """
    Kill orphaned terminal multiplexer servers at runner startup.

    Thin module-level entry point (the runner imports this name) that
    delegates orphan enumeration and reaping to the tmux backend — the
    default backend today. See :meth:`TmuxBackend.reap_orphans` for the sweep
    semantics.

    :returns: The number of orphaned instance dirs reaped.
    """
    return TmuxBackend.reap_orphans()


def build_terminal_os_env_spec(
    spec: TerminalEnvSpec,
    *,
    parent_os_env_spec: OSEnvSpec | None = None,
    cwd_override: str | None = None,
    sandbox_override: str | None = None,
) -> OSEnvSpec:
    effective_os_env_spec: OSEnvSpec | None = None
    if spec.os_env == "inherit" or spec.os_env is None:
        effective_os_env_spec = (
            _clone_os_env_spec(parent_os_env_spec) if parent_os_env_spec is not None else None
        )
    elif isinstance(spec.os_env, OSEnvSpec):
        effective_os_env_spec = _clone_os_env_spec(spec.os_env)

    if effective_os_env_spec is None:
        effective_os_env_spec = OSEnvSpec(
            type="caller_process",
            cwd=os.getcwd(),
            sandbox=OSEnvSandboxSpec(type="none"),
        )

    if cwd_override is not None:
        if not spec.allow_cwd_override:
            raise ValueError("This terminal does not allow cwd overrides")
        # Containment check: the LLM-supplied cwd must resolve to the
        # spec's cwd or a subdirectory of it. Without this guard, an
        # LLM with ``allow_cwd_override: true`` could repoint the
        # terminal anchor anywhere — e.g. ``/``, ``~/.ssh``, ``/etc`` —
        # which would:
        #
        # - On bwrap: bind-mount that location as the workspace root
        #   (escapes the project sandbox).
        # - On seatbelt: anchor the dotfile/credential masker at the
        #   wrong root so e.g. ``~/.ssh/id_rsa`` is no longer a
        #   "hidden" path under the new cwd.
        # - Resolve ``write_paths: ["."]`` to the new root, granting
        #   writes anywhere the LLM picks.
        #
        # Relative overrides are interpreted against the spec's cwd
        # (not the supervisor's ``os.getcwd()``) so the LLM can say
        # ``cd .worktrees/foo`` without depending on where the
        # supervisor was launched from. Absolute overrides are
        # checked literally; they must still be under the spec cwd.
        if effective_os_env_spec.cwd:
            allowed_root = Path(effective_os_env_spec.cwd).expanduser().resolve(strict=False)
        else:
            allowed_root = Path(os.getcwd()).resolve(strict=False)
        override_path = Path(cwd_override).expanduser()
        if override_path.is_absolute():
            resolved_override = override_path.resolve(strict=False)
        else:
            resolved_override = (allowed_root / override_path).resolve(strict=False)
        try:
            resolved_override.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError(
                f"cwd override {cwd_override!r} resolves to {resolved_override} "
                f"which is outside the allowed root {allowed_root}. A terminal "
                "cwd override must point at the spec's cwd or a subdirectory "
                "of it; pointing elsewhere would escape the sandbox's "
                "filesystem and dotfile-masking anchors."
            ) from exc
        effective_os_env_spec.cwd = str(resolved_override)

    if sandbox_override is not None:
        if not spec.allow_sandbox_override:
            raise ValueError("This terminal does not allow sandbox overrides")
        sandbox = effective_os_env_spec.sandbox or OSEnvSandboxSpec(type="none")
        # Defense in depth on top of the parse-time check
        # (omnigent/inner/loader.py rejects allow_sandbox_override:
        # true paired with egress_rules at agent-load time). This
        # branch also fires for specs built programmatically without
        # going through the loader and catches any future code path
        # that synthesizes an override before launch. An override to
        # ``"none"`` can't hard-enforce network isolation; letting the
        # override drop ``sandbox.type`` to it while ``egress_rules``
        # stay on the policy would silently bypass the network
        # allow-list.
        if sandbox.egress_rules:
            raise ValueError(
                "sandbox_override is not allowed on a terminal whose "
                "effective sandbox declares egress_rules: overriding "
                "to 'none' would drop hard network "
                "enforcement while egress_rules remain as inert "
                "decoration on the policy."
            )
        sandbox.type = sandbox_override
        effective_os_env_spec.sandbox = sandbox

    return effective_os_env_spec


class Liveness(enum.Enum):
    """
    Uniform liveness verdict for a hosted terminal session.

    Answers the two questions the terminal machinery needs about a managed
    terminal — does the *host endpoint* (the multiplexer's session/pane) still
    exist, and is the *inner agent process* inside it still running — as a
    single verdict every backend can produce. Modeling it as a verdict rather
    than a raw ``pane_dead`` flag lets callers degrade safely without
    per-backend branching, and lets backends that cannot observe an inner
    exit collapse it into :attr:`ENDPOINT_GONE`.

    Deliberately carries no exit code: herdr drops the pane (and its final
    screen) the instant the inner process exits, so no backend may promise
    one. :attr:`UNKNOWN` is the safe-degradation verdict when a probe cannot
    run — callers treat it as not-alive without asserting the endpoint is
    gone.
    """

    ALIVE = "alive"
    """The host endpoint exists and the inner process is still running."""

    INNER_EXITED = "inner_exited"
    """The inner process exited but the host kept the endpoint and its final
    screen (tmux ``remain-on-exit``). Backends that cannot preserve a dead
    endpoint report :attr:`ENDPOINT_GONE` on exit instead."""

    ENDPOINT_GONE = "endpoint_gone"
    """The host endpoint (session / pane / server) no longer exists."""

    UNKNOWN = "unknown"
    """The probe could not determine liveness (it failed to run). Callers must
    degrade safely — treat as not-alive without concluding the endpoint is
    gone."""


@dataclass(frozen=True)
class TerminalBackendCapabilities:
    """
    Static capability declaration for a :class:`TerminalBackend`.

    Read at backend-selection time (the terminal factory) and by machinery
    above the seam. A feature a backend lacks degrades to a documented no-op:
    callers read the flag and skip the feature rather than branching on the
    backend's identity — a lesson imported from firstmate's backend contract.
    """

    native_popup: bool = False
    """The backend can host a native popup over the pane (tmux's cost popup).
    Where ``False`` the web approval card is the elicitation surface."""

    start_on_attach: bool = False
    """The backend can delay inner-command startup until the first client
    attaches. Where ``False`` a backend watcher loop emulates it."""

    native_busy_state: bool = False
    """The backend exposes a native agent busy/idle signal. Where ``False``
    the capture-diff idle watcher is the only truth."""

    push_events: bool = False
    """The backend can push output/state deltas over a control channel. Where
    ``False`` polling is the truth."""

    control_mode_attach: bool = False
    """The backend offers a control-mode attach transport in addition to a
    PTY attach (tmux ``-C``)."""

    attach_transports: frozenset[str] = field(
        default_factory=lambda: frozenset({TERMINAL_TRANSPORT_PTY})
    )
    """Web attach transports the backend can host."""

    status_line: bool = False
    """The backend has a host status line that can carry the cosmetic
    conversation link (tmux's status-left). Where ``False``,
    :meth:`TerminalBackend.set_status_link` is a documented no-op — the link
    is droppable because the web UI is the primary surface."""


@dataclass(frozen=True)
class TerminalLaunchRequest:
    """
    Backend-neutral request to launch one hosted terminal session.

    Speaks omnigent domain terms — a command argv, a working directory, an
    environment, a viewport — plus cross-backend behavioral options. The
    caller has already resolved everything process-shaped (sandbox/egress
    wrapping of :attr:`command`, environment merging and leak-stripping of
    :attr:`env`); the backend only hosts it. Options a backend cannot honor
    degrade per its :class:`TerminalBackendCapabilities`.
    """

    command: list[str]
    """The already-resolved argv to run inside the hosted pane. Hosted
    verbatim."""

    cwd: str
    """Working directory for the inner process."""

    env: dict[str, str]
    """Full environment for the inner process (already merged and stripped)."""

    size: tuple[int, int] = (80, 24)
    """Initial viewport as ``(cols, rows)``. Deliberately small so the first
    client attach grows it losslessly."""

    scrollback: int = 10000
    """Lines of scrollback history the host should retain."""

    keep_alive_after_exit: bool = False
    """Keep the host endpoint alive after the inner process exits so its final
    screen stays capturable and liveness reports :attr:`Liveness.INNER_EXITED`
    rather than racing endpoint teardown. Backends that cannot preserve a dead
    endpoint ignore this and report :attr:`Liveness.ENDPOINT_GONE` on exit."""

    allow_passthrough: bool = False
    """Allow the inner program to drive the host terminal via passthrough
    escapes. Ignored by backends without a passthrough concept."""

    start_on_attach: bool = False
    """Delay inner-command startup until the first client attaches. Needs the
    :attr:`TerminalBackendCapabilities.start_on_attach` capability; otherwise
    the backend emulates or ignores it."""

    status_link: str | None = None
    """Cosmetic conversation link to seed the host's status line, or ``None``.
    Purely decorative; backends without a status line ignore it."""


class TerminalBackend(ABC):
    """
    Backend interface for a terminal multiplexer.

    One instance backs one :class:`TerminalInstance` and owns how that
    terminal's hosting endpoint is created on a specific multiplexer (tmux
    today, herdr next), driven (input, capture), probed for liveness, and torn
    down. The machinery above the seam — the registry, the ``sys_terminal``
    tools, the native bridges, the web attach routes — talks only to the
    :class:`TerminalInstance` public surface and never to a multiplexer
    directly. The factory is the single point that selects a backend.

    **Scope.** The backend owns session lifecycle (launch, liveness, close,
    the class-level orphan sweep), input delivery (non-submitting text
    injection and named-key send), screen capture (plain or ANSI), and the
    cosmetic status-line link. The idle/activity *watcher loops* themselves —
    the diff logic, timers, and edge callbacks — stay on
    :class:`TerminalInstance`; only their multiplexer touchpoints (capture,
    liveness, detach-on-exit) route through this seam. The per-harness native
    bridges and web-attach transports are migrated in later changes.

    **Vocabulary.** Methods speak Omnigent domain terms, not one multiplexer's
    CLI flags: "capture the screen as text (optionally with ANSI)", "type this
    literal text", "press these named keys". Each backend translates that
    vocabulary into its own CLI itself, and normalizes captured output (line
    endings, etc.) so callers never see backend-specific quirks.
    """

    name: str
    """Stable backend identifier, e.g. ``"tmux"``. Selection fails loudly on
    an unknown name."""

    capabilities: TerminalBackendCapabilities
    """Static capability declaration, checked at selection time."""

    platforms: frozenset[str]
    """Platform tags the backend supports (``"posix"`` / ``"windows"``)."""

    @classmethod
    def ensure_available(cls) -> None:
        """Verify the backend's multiplexer binary is present and usable.

        Called at selection time (the terminal factory) after the platform
        check passes, so a chosen backend whose binary is missing or too old
        fails loudly with an actionable, install-hint error instead of
        crashing mysteriously at launch. The default is a no-op; a backend
        driven by an external binary (tmux, herdr) overrides this to probe
        that binary — and, where the binary's protocol churns (herdr is
        pre-1.0), to gate its version.
        """
        del cls  # base no-op; binary-backed backends override this

    @classmethod
    def construct_for_instance(cls, *, socket_path: Path, target: str) -> TerminalBackend | None:
        """Optional construction hook used by :func:`_construct_terminal_backend`.

        Each backend takes different constructor arguments, so a
        :class:`TerminalInstance` cannot build one with a uniform call. Rather
        than grow a per-backend ``if`` ladder in
        :func:`_construct_terminal_backend`, a backend that an instance must be
        able to build declares *how* to build itself here: it returns a fresh
        instance bound to this terminal's private endpoint (*socket_path*) and
        *target*.

        The default returns ``None``, meaning "this backend provides no
        construction hook" — :func:`_construct_terminal_backend` then fails
        loudly with :class:`NotImplementedError` rather than silently falling
        back to another backend. tmux keeps its own dedicated branch in the
        dispatcher (so its construction stays byte-identical); the in-process
        test :class:`FakeBackend` and the future ``HerdrBackend`` (#11) override
        this and become constructible with no edit to the dispatcher.

        :param socket_path: Private multiplexer socket path for this instance.
        :param target: Session/pane target name, e.g. ``"main"``.
        :returns: A fresh backend bound to this terminal, or ``None`` when the
            backend has no construction hook.
        """
        del socket_path, target  # base: no hook; overriding backends build here
        return None

    def advertised_target(self, *, fallback: str) -> str:
        """Return the address persisted for later delivery reconstruction."""
        return fallback

    @abstractmethod
    async def launch(self, request: TerminalLaunchRequest) -> None:
        """Create the hosted session/pane and start the inner command.

        :param request: The backend-neutral launch request.
        :raises RuntimeError: If the multiplexer rejects the launch.
        """
        raise NotImplementedError

    @abstractmethod
    async def liveness(self) -> Liveness:
        """Probe the hosted session's liveness as a :class:`Liveness` verdict.

        Pure: never mutates caller state. Returns :attr:`Liveness.UNKNOWN`
        when the probe itself cannot run.
        """
        raise NotImplementedError

    @abstractmethod
    def liveness_sync(self) -> Liveness:
        """Synchronous sibling of :meth:`liveness` for callers without an
        event loop (the threaded idle watcher). Same verdict semantics."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Tear down the hosted session/server. Idempotent; must not raise."""
        raise NotImplementedError

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Type literal text into the hosted pane WITHOUT submitting it.

        The text is delivered verbatim and non-submitting (bracketed-paste
        safe): no trailing newline is implied, so a multi-line prompt or a
        pasted code block arrives intact for the caller to submit separately
        via :meth:`send_keys`. The backend chunks internally as needed to stay
        under any wire-protocol size cap; the inner program sees one
        contiguous stream in submission order.

        :param text: The literal characters to type.
        :raises RuntimeError: If the multiplexer rejects the input (e.g. the
            host endpoint is gone).
        """
        raise NotImplementedError

    @abstractmethod
    async def send_keys(self, keys: Sequence[str]) -> None:
        """Press named keys in the hosted pane, in order.

        Keys use Omnigent's backend-neutral key vocabulary — ``"Enter"``,
        ``"Escape"``, ``"Tab"``, ``"Up"``, and ``"C-a"``-style modifier
        notation for control/meta chords (the form Omnigent already uses
        internally). Each backend translates these names into its own key
        syntax itself: herdr, for example, uses ``ctrl+a`` plus-notation and
        lacks Home/End/PageUp/Delete. A backend that cannot express a given
        key should skip it rather than send a wrong key.

        :param keys: Ordered key names, e.g. ``["Enter"]`` or ``["C-c"]``.
        :raises RuntimeError: If the multiplexer rejects the input (e.g. the
            host endpoint is gone).
        """
        raise NotImplementedError

    @abstractmethod
    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Snapshot the hosted pane's rendered screen as text.

        :param ansi: When ``True``, include ANSI styling escapes (for human
            display and the idle watcher's diff); when ``False``, plain text
            only (the ``read`` path).
        :param scrollback: Extra lines of scrollback history to include above
            the visible viewport. ``0`` captures only the visible screen.
        :returns: The screen text with line endings normalized to ``\\n`` —
            the backend strips any native line-ending quirk (e.g. CRLF) so
            callers never branch on it.
        :raises RuntimeError: If the snapshot command fails (typically because
            the host endpoint has gone away); the caller reads this as "host
            went away" and stops.
        """
        raise NotImplementedError

    @abstractmethod
    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Synchronous sibling of :meth:`capture` for callers without an event
        loop (the threaded idle watcher). Same params, verdict, and line-ending
        normalization."""
        raise NotImplementedError

    async def set_status_link(self, link: str | None) -> None:
        """Update the cosmetic conversation link on the host's status line.

        Capability-gated by :attr:`TerminalBackendCapabilities.status_line`:
        the link is purely decorative (the web UI is the primary surface), so
        a backend without a status line implements this as a documented no-op.
        The default implementation here is that no-op; backends that declare
        ``status_line=True`` override it.

        :param link: Conversation URL to show, e.g. ``"/c/conv_abc123"``, or
            ``None`` to clear it.
        """
        del link  # base no-op; backends with a status line override this

    async def detach_display_clients(self) -> None:  # noqa: B027 — optional override hook; default is a no-op
        """Detach any attached human display clients from the hosted session.

        Called after the inner process exits while the host keeps the endpoint
        alive (see :attr:`Liveness.INNER_EXITED`), so attached display clients
        — a ``tmux attach`` subprocess, the server-side bridge PTY — exit
        cleanly instead of hanging on the frozen final frame. The default is a
        no-op: backends that drop the endpoint on exit, or that have no
        attachable display client, have nothing to detach.
        """

    def detach_display_clients_sync(self) -> None:  # noqa: B027 — optional override hook; default is a no-op
        """Synchronous sibling of :meth:`detach_display_clients` for the
        threaded idle watcher. Default no-op."""

    async def busy_state(self) -> bool | None:
        """Report whether the inner agent is mid-turn (busy), if the backend can.

        Optional signal for machinery above the seam (a native complement to the
        capture-diff idle watcher). The default returns ``None`` — "no native
        busy signal" — for backends without
        :attr:`TerminalBackendCapabilities.native_busy_state` (tmux, the
        in-process fake); a backend that exposes a native agent state overrides
        this. ``True`` = busy, ``False`` = idle, ``None`` = no usable signal (the
        caller then relies on its own capture-diff watcher).
        """
        return None

    async def input_ready(self) -> bool | None:
        """Report whether the pane's composer is ready for a new prompt, if known.

        Optional signal for the input-delivery dance (paste then submit). The
        default returns ``None`` — "unknown, deliver optimistically" — for
        backends that cannot observe composer readiness; a backend with a native
        agent state overrides this. ``True`` = ready, ``False`` = not ready
        (mid-turn or gone), ``None`` = unknown.
        """
        return None

    # --------------------------------------------------- delivery surface (sync)
    #
    # The synchronous half of the protocol, driving :class:`TerminalDelivery`
    # (the consolidated prompt-delivery dance). The native bridges deliver from a
    # worker thread with no event loop, so these siblings exist alongside the
    # async ``send_text`` / ``send_keys`` / ``capture`` for exactly the same
    # reason ``capture_sync`` / ``liveness_sync`` do. ``paste_without_submit_sync``
    # is a NEW primitive (bracketed non-submitting multi-line paste) distinct
    # from literal ``send_text`` typing; ``kill_session_sync`` is a hard stop of
    # this one session (which a multiplexer may distinguish from tearing down the
    # whole host — see :meth:`close`).

    @abstractmethod
    def send_text_sync(self, text: str) -> None:
        """Synchronous sibling of :meth:`send_text` (literal, non-submitting).

        Same verbatim, non-submitting, backend-chunked semantics as
        :meth:`send_text`; provided for the thread-based delivery callers that
        have no event loop.
        """
        raise NotImplementedError

    @abstractmethod
    def send_keys_sync(self, keys: Sequence[str]) -> None:
        """Synchronous sibling of :meth:`send_keys` (named keys, in order)."""
        raise NotImplementedError

    @abstractmethod
    def paste_without_submit_sync(self, text: str) -> None:
        """Paste multi-line *text* into the composer WITHOUT submitting it.

        The delivery-dance primitive: *text* lands as a single editable draft in
        the pane's input box (bracketed-paste safe), so interior newlines stay
        data rather than submitting per line, and the caller submits separately
        via :meth:`send_keys_sync`. Distinct from :meth:`send_text_sync`, which
        types literally: a backend may use a bulk paste channel here (tmux loads
        a buffer and pastes it with bracketed-paste markers) that differs from
        literal keystroke injection. Callers append their own trailing newline
        when they need one (it absorbs a trailing backslash so it cannot escape
        the submit key).

        :param text: The draft to paste, possibly multi-line.
        :raises RuntimeError: If the multiplexer rejects the paste (e.g. the host
            endpoint is gone).
        """
        raise NotImplementedError

    @abstractmethod
    def kill_session_sync(self) -> None:
        """Hard-stop THIS hosted session, terminating its inner process.

        The "Stop session" affordance behind the web UI. Distinct from
        :meth:`close` where a backend separates one session from its host server:
        this kills the session/pane (and the inner CLI in it) specifically.
        Raises on failure so the caller can surface it (a wedged host must not
        read as a successful stop).

        :raises RuntimeError: If the multiplexer rejects the kill.
        """
        raise NotImplementedError

    def delivery_snapshot_sync(self) -> str:
        """Snapshot the pane for the delivery dance's readiness/verify polls.

        Unlike :meth:`capture_sync`, this NEVER raises: a transient capture
        failure during boot or a mid-turn repaint is "not ready yet" to the
        polling caller, not an error, so it degrades to ``""``. The default wraps
        :meth:`capture_sync`; a backend whose delivery capture must match a
        pre-existing byte-exact command stream overrides it.

        :returns: The plain (no-ANSI) screen text, or ``""`` when the snapshot
            could not be taken.
        """
        try:
            return self.capture_sync(ansi=False)
        except RuntimeError:
            return ""

    def delivery_liveness_sync(self) -> bool:
        """Report whether this session's host endpoint currently exists.

        The delivery dance's fast-fail probe (:meth:`TerminalDelivery.is_alive`):
        a bridge checks this before injecting a web message so a prompt into an
        already-exited TUI raises a clear "restart the session" error instead of
        being silently typed into a dead pane. Distinct from :meth:`liveness_sync`
        (which grades ALIVE / INNER_EXITED / ENDPOINT_GONE / UNKNOWN): the
        delivery caller only needs a boolean "endpoint present", so this collapses
        the verdict. NEVER raises — a probe that cannot run reads as not-alive.

        The default derives the boolean from :meth:`liveness_sync` (so the herdr
        backend routes through its existing pane-liveness machinery, and the
        in-process fake through its scriptable verdict). A backend whose delivery
        liveness must match a pre-existing byte-exact command stream — tmux's
        ``has-session`` — overrides this.

        :returns: ``True`` when the endpoint exists (ALIVE or INNER_EXITED),
            ``False`` otherwise (ENDPOINT_GONE / UNKNOWN).
        """
        return self.liveness_sync() in (Liveness.ALIVE, Liveness.INNER_EXITED)

    def send_keys_repeated_sync(self, key: str, count: int) -> None:
        """Press *key* *count* times (the composer-clear burst primitive).

        Backs :meth:`TerminalDelivery.send_keys_repeated`, the primitive cursor's
        backspace-flood composer clear is built on. The default expresses the
        repeat as *count* presses through the existing send path — the semantics a
        backend without a native repeat has (herdr sends N single keys). tmux
        overrides this with its native ``send-keys -N`` repeat so the migrated
        cursor command stream stays byte-identical. A non-positive *count* is a
        no-op.

        :param key: A single named key in the neutral vocabulary, e.g. ``"BSpace"``.
        :param count: How many times to press it; ``<= 0`` sends nothing.
        """
        if count <= 0:
            return
        self.send_keys_sync([key] * count)

    def send_keys_atomic_sync(self, keys: Sequence[str]) -> None:
        """Press all *keys* in ONE multiplexer client command (atomic multi-key).

        Backs :meth:`TerminalDelivery.send_keys_atomic`, for a source stream that
        packed several named keys into a single injection — e.g. goose's
        permission-dialog ``Down Down Enter``. A single client command is atomic
        relative to any other attached client, so the sequence cannot be
        interleaved by a concurrent client and mis-answer the dialog. Distinct
        from :meth:`send_keys_sync`, which the tmux backend sends as one command
        PER key.

        The default delegates to :meth:`send_keys_sync`, which already packs every
        key into a single command on backends whose send path is inherently atomic
        (herdr's one ``pane send-keys``; the in-process fake records one call);
        tmux overrides this to emit a single ``send-keys``.

        :param keys: The ordered keys to deliver as one command, e.g.
            ``["Down", "Down", "Enter"]``.
        """
        self.send_keys_sync(keys)

    def native_popup_launch(
        self,
        *,
        config_file: Path,
        session_id: str,
        elicitation_id: str,
        message: str,
        policy_name: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        """Overlay a native approval popup on the pane, if the backend can.

        Capability-gated by :attr:`TerminalBackendCapabilities.native_popup`
        (checked by :meth:`TerminalDelivery.launch_native_popup`). The default is
        a documented no-op — where a backend cannot host a pane popup the web
        approval card remains the elicitation surface, exactly the degradation
        the spec calls for. tmux overrides this.

        :param config_file: AP-routing config the popup reads (base URL + auth
            headers).
        :param session_id: Omnigent session id that owns the elicitation.
        :param elicitation_id: Outstanding elicitation correlation id.
        :param message: Approval reason rendered in the popup.
        :param policy_name: Deciding policy name (modal header), or ``None``.
        :param python_executable: Interpreter to run the popup module, or
            ``None`` for the caller's default.
        """
        del (
            config_file,
            session_id,
            elicitation_id,
            message,
            policy_name,
            python_executable,
        )


@dataclass(frozen=True)
class TmuxDeliveryStyle:
    """Per-bridge tmux delivery argv dialect for :class:`TmuxBackend`.

    The native bridges migrated in #9 (cursor, goose, kimi) each drove tmux with
    their OWN paste-buffer name, ``capture-pane`` flag order, and per-command
    timeout, and their per-harness argv tests pin those exact bytes. Consolidating
    them onto the shared :class:`TerminalDelivery` must therefore PRESERVE each
    bridge's stream, not normalize it onto claude's. This carries the three knobs
    that differ between bridges; every default reproduces claude's audited #8
    command stream byte-for-byte, so a :class:`TmuxBackend` built WITHOUT a style
    — every #8 caller, and every :class:`TerminalInstance` — is byte-unchanged.

    :param paste_buffer: Named tmux buffer the bracketed paste loads into
        (``load-buffer -b`` / ``paste-buffer -b``). Claude/default:
        ``"omnigent-paste"``; cursor/goose/kimi pass their own.
    :param capture_flag_before_target: ``capture-pane`` flag order for the
        delivery snapshot. ``False`` (default) → ``capture-pane -t <target> -p``
        (claude's order, the #8 surface); ``True`` → ``capture-pane -p -t
        <target>`` (cursor/goose/kimi's order).
    :param literal_flag_before_target: ``send-keys -l`` flag order for literal
        typing (:meth:`TmuxBackend.send_text_sync`). ``False`` (default) →
        ``send-keys -l -t <target> <text>`` (claude's order, the #8 surface);
        ``True`` → ``send-keys -t <target> -l <text>`` (cursor's ``/model``
        picker order). Only cursor differs here (goose/kimi type no literals).
    :param command_timeout_s: Per-command subprocess timeout for every delivery
        client command (send/paste/kill/snapshot/liveness). Default matches
        claude's :data:`_DELIVERY_SEND_TIMEOUT_S`; cursor/goose pass ``10.0``,
        kimi passes the default.
    """

    paste_buffer: str = _TMUX_PASTE_BUFFER
    capture_flag_before_target: bool = False
    literal_flag_before_target: bool = False
    command_timeout_s: float = _DELIVERY_SEND_TIMEOUT_S


# The claude-shaped default: byte-identical to the #8 surface, used by every
# construction that does not pass its own style.
_DEFAULT_TMUX_DELIVERY_STYLE = TmuxDeliveryStyle()


class TmuxBackend(TerminalBackend):
    """
    tmux multiplexer backend.

    Hosts each managed terminal in its own private tmux server (isolated
    socket, no user ``~/.tmux.conf``). Owns launch, liveness, close, the
    class-level orphan sweep, input delivery (``send-keys``), screen capture
    (``capture-pane``), and the status-line link (``set-option``), translating
    the backend-neutral protocol into tmux argv. All tmux vocabulary — the
    socket path, ``-t`` targets, ``-F`` formats, ``#{pane_dead}``, ``-l``
    literal keys, ``-p``/``-e`` capture flags — is confined to this class.
    """

    name = "tmux"
    capabilities = TerminalBackendCapabilities(
        native_popup=True,
        start_on_attach=True,
        native_busy_state=False,
        push_events=False,
        control_mode_attach=True,
        attach_transports=frozenset({TERMINAL_TRANSPORT_CONTROL, TERMINAL_TRANSPORT_PTY}),
        status_line=True,
    )
    # tmux is POSIX-only; native Windows uses a different backend, selected by
    # the terminal factory (which raises a clear availability error on Windows
    # until a Windows-native backend is registered).
    platforms = frozenset({"posix"})

    @classmethod
    def ensure_available(cls) -> None:
        """Fail loudly with an install hint when tmux is not on PATH.

        The tmux backend hosts each managed terminal in a private tmux server,
        so a missing ``tmux`` binary means no native harness can launch. Raise
        an actionable error naming the common install commands rather than
        letting the first ``tmux`` subprocess fail opaquely.

        :raises RuntimeError: When ``tmux`` is not installed or not on PATH.
        """
        if not _tmux_available():
            raise RuntimeError(
                "tmux is not installed or not on PATH. The tmux terminal "
                "backend hosts native harnesses in a private tmux server; "
                "install tmux (e.g. `apt install tmux`, `brew install tmux`, "
                "`dnf install tmux`) and ensure it is on PATH."
            )

    def __init__(
        self,
        *,
        socket_path: str | Path,
        target: str = "main",
        config_path: str = _TMUX_CONFIG_PATH,
        delivery_style: TmuxDeliveryStyle | None = None,
        paste_dir: str | Path | None = None,
    ) -> None:
        """
        :param socket_path: Private tmux socket path for this instance's
            server. Accepted as ``str`` or :class:`~pathlib.Path` — it is only
            ever stringified onto a tmux ``-S`` argv, so a bridge advertising a
            socket string (:func:`build_prompt_delivery`) reaches the same argv
            as a :class:`TerminalInstance` passing a ``Path`` without a
            platform-dependent round-trip.
        :param target: Session/pane target name, e.g. ``"main"``.
        :param config_path: tmux config file to load, ``os.devnull`` so a
            managed session never inherits the user's ``~/.tmux.conf``.
        :param delivery_style: Per-bridge tmux delivery argv dialect
            (:class:`TmuxDeliveryStyle` — paste-buffer name, capture flag order,
            per-command timeout). ``None`` selects the claude-shaped default, so a
            backend built without one drives the byte-identical #8 command stream;
            a migrated native bridge passes its own via
            :func:`build_prompt_delivery`.
        """
        self._socket_path = socket_path
        self._target = target
        self._config_path = config_path
        self._delivery_style = delivery_style or _DEFAULT_TMUX_DELIVERY_STYLE
        self._paste_dir = Path(paste_dir) if paste_dir is not None else Path(socket_path).parent

    def _base_cmd(self) -> list[str]:
        """Build the tmux argv prefix for this instance's private server."""
        return ["tmux", "-S", str(self._socket_path), "-f", self._config_path]

    async def _run(self, *args: str) -> None:
        """Run a tmux command against this server; raise on non-zero exit."""
        proc = await asyncio.create_subprocess_exec(
            *self._base_cmd(),
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"tmux command failed: {' '.join(args)}: {stderr.decode().strip()}")

    async def _run_output(self, *args: str) -> str:
        """Run a tmux command against this server and return its stdout."""
        proc = await asyncio.create_subprocess_exec(
            *self._base_cmd(),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"tmux command failed: {' '.join(args)}: {stderr.decode().strip()}")
        return stdout.decode()

    def _run_output_sync(self, *args: str) -> str:
        """Synchronous sibling of :meth:`_run_output` for the threaded watcher.

        Same error semantics: a non-zero exit raises :class:`RuntimeError`
        carrying the stderr (typically because the server has gone away).
        """
        proc = subprocess.run([*self._base_cmd(), *args], capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"tmux command failed: {' '.join(args)}: {proc.stderr.decode().strip()}"
            )
        return proc.stdout.decode()

    def _capture_args(self, *, ansi: bool, scrollback: int) -> list[str]:
        """Build the ``capture-pane`` argv for a screen snapshot.

        ``-p`` prints to stdout; ``-e`` preserves ANSI styling; ``-S -N``
        reaches ``N`` lines back into scrollback above the visible screen.

        :param ansi: Include ANSI escapes (``-e``).
        :param scrollback: Extra scrollback lines to include; ``0`` for the
            visible viewport only.
        :returns: The tmux argv, e.g. ``["capture-pane", "-t", "main", "-p"]``.
        """
        args = ["capture-pane", "-t", self._target, "-p"]
        if ansi:
            args.append("-e")
        if scrollback > 0:
            args.extend(["-S", f"-{scrollback}"])
        return args

    async def launch(self, request: TerminalLaunchRequest) -> None:
        """Create the private tmux server and single session for *request*."""
        inner_str = " ".join(_shell_quote(c) for c in request.command)
        if request.start_on_attach:
            inner_str = f"tmux wait-for {_TMUX_START_ON_ATTACH_CHANNEL}; exec {inner_str}"

        option_commands = [
            *_tmux_managed_option_commands(
                request.scrollback,
                allow_passthrough=request.allow_passthrough,
                keep_alive_after_exit=request.keep_alive_after_exit,
            ),
            [
                "set-option",
                "-g",
                _TMUX_CONVERSATION_LINK_OPTION,
                request.status_link or _TMUX_EMPTY_OPTION_VALUE,
            ],
        ]
        if request.start_on_attach:
            option_commands.append(
                [
                    "set-hook",
                    "-g",
                    "client-attached",
                    f"wait-for -S {_TMUX_START_ON_ATTACH_CHANNEL}",
                ]
            )
        # ``pane-died`` is a window-scope hook that fires when remain-on-exit
        # keeps the pane alive after the inner process exits. It is set AFTER
        # new-session (not before) because window scope requires an existing
        # window, and global scope (-g) does not fire for pane-died.
        pane_died_hook: list[list[str]] = (
            [["set-hook", "-w", "pane-died", "detach-client -a"]]
            if request.keep_alive_after_exit
            else []
        )
        cols, rows = request.size
        cmd = [
            *self._base_cmd(),
            *_tmux_command_sequence(
                [
                    *option_commands,
                    [
                        "new-session",
                        "-d",
                        "-s",
                        self._target,
                        # Deliberately small: first attach GROWS (lossless).
                        # The old 200x50 meant first attach SHRANK, and ink's
                        # cursor-up repaint (counted in unwrapped rows)
                        # stitched frames into rewrapped debris — garbled text.
                        "-x",
                        str(cols),
                        "-y",
                        str(rows),
                        "-c",
                        request.cwd,
                        inner_str,
                    ],
                    *pane_died_hook,
                ]
            ),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=request.env,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"tmux launch failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

    async def liveness(self) -> Liveness:
        """Probe ``#{pane_dead}`` and map it to a :class:`Liveness` verdict.

        ``list-panes`` errors on an unknown target (unlike ``display-message``,
        which silently falls back to another pane), so a non-zero exit is a
        reliable "session/server gone" signal.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._base_cmd(),
                "list-panes",
                "-t",
                self._target,
                "-F",
                "#{pane_dead}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
        except OSError:
            return Liveness.UNKNOWN
        return self._verdict(proc.returncode, stdout.decode())

    def liveness_sync(self) -> Liveness:
        """Synchronous :meth:`liveness` for the daemon idle-watcher thread."""
        try:
            proc = subprocess.run(
                [*self._base_cmd(), "list-panes", "-t", self._target, "-F", "#{pane_dead}"],
                capture_output=True,
                check=False,
            )
        except OSError:
            return Liveness.UNKNOWN
        return self._verdict(proc.returncode, proc.stdout.decode())

    @staticmethod
    def _verdict(returncode: int | None, stdout: str) -> Liveness:
        """Map a ``list-panes -F #{pane_dead}`` result to a verdict.

        rc != 0 or no panes → the session/server is gone; a ``1`` line → the
        pane process exited but ``remain-on-exit`` kept the session; a ``0``
        line → the pane is live.
        """
        panes = stdout.split()
        if returncode != 0 or not panes:
            return Liveness.ENDPOINT_GONE
        if "1" in panes:
            return Liveness.INNER_EXITED
        return Liveness.ALIVE

    async def close(self) -> None:
        """Kill the private tmux server. Suppresses the already-gone case."""
        with contextlib.suppress(RuntimeError):
            await self._run("kill-server")

    async def send_text(self, text: str) -> None:
        """Type literal text via ``send-keys -l``, chunked under tmux's cap.

        tmux's client->server protocol rejects any single command over its
        16KB imsg cap, so the literal is split into
        :data:`_SEND_KEYS_LITERAL_CHARS_PER_CALL`-char invocations; tmux writes
        each in submission order so the pane sees one contiguous stream.
        """
        for start in range(0, len(text), _SEND_KEYS_LITERAL_CHARS_PER_CALL):
            await self._run(
                "send-keys",
                "-l",
                "-t",
                self._target,
                text[start : start + _SEND_KEYS_LITERAL_CHARS_PER_CALL],
            )

    async def send_keys(self, keys: Sequence[str]) -> None:
        """Press each named key via ``send-keys``.

        tmux's key names are Omnigent's neutral vocabulary verbatim
        (``Enter``, ``Escape``, ``C-c``, ...), so no translation is needed.
        """
        for key in keys:
            await self._run("send-keys", "-t", self._target, key)

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Capture the pane via ``capture-pane``.

        tmux ``capture-pane -p`` already emits logical lines joined by ``\\n``
        (no CR), so the :meth:`TerminalBackend.capture` ``\\n``-normalization
        contract holds with no extra transform.
        """
        return await self._run_output(*self._capture_args(ansi=ansi, scrollback=scrollback))

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Synchronous :meth:`capture` for the threaded idle watcher."""
        return self._run_output_sync(*self._capture_args(ansi=ansi, scrollback=scrollback))

    async def set_status_link(self, link: str | None) -> None:
        """Set the status-left conversation link option on the server."""
        await self._run(
            "set-option",
            "-g",
            _TMUX_CONVERSATION_LINK_OPTION,
            link or _TMUX_EMPTY_OPTION_VALUE,
        )

    async def detach_display_clients(self) -> None:
        """Detach all clients from the session via ``detach-client -s``."""
        await self._run_output("detach-client", "-s", self._target)

    def detach_display_clients_sync(self) -> None:
        """Synchronous :meth:`detach_display_clients` for the threaded watcher."""
        self._run_output_sync("detach-client", "-s", self._target)

    # --------------------------------------------------- delivery surface (sync)
    #
    # These reproduce, byte-for-byte, the command stream the seven native bridges'
    # private ``_run_tmux`` / ``_capture_pane`` helpers produced, so migrating a
    # bridge onto the shared :class:`TerminalDelivery` leaves its POSIX behavior
    # (and the tests that pin its exact tmux argv) unchanged. Deliberately built
    # WITHOUT the ``-f`` config flag ``_base_cmd`` carries: these are pure client
    # commands against the already-running server (the runner created it with
    # ``-f``), so ``-f`` — consulted only when a server starts — is inert here,
    # and its absence is what keeps the migrated command stream identical to the
    # private helpers' (which never passed it).

    def _delivery_cmd(self, *args: str) -> list[str]:
        """Build a delivery client-command argv: ``tmux -S <sock> <args...>``.

        No ``-f``: see the delivery-surface note above.
        """
        return ["tmux", "-S", str(self._socket_path), *args]

    def _delivery_run_sync(self, *args: str) -> None:
        """Run one delivery client command; raise on non-zero exit or timeout.

        Byte-identical error semantics to the bridges' private ``_run_tmux``: a
        non-zero exit raises :class:`RuntimeError` carrying the stderr (typically
        "no server running" once the pane is gone), so the delivery caller can
        surface a transport failure rather than a silent success.
        """
        timeout_s = self._delivery_style.command_timeout_s
        try:
            proc = subprocess.run(
                self._delivery_cmd(*args),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"tmux command timed out after {timeout_s}s") from exc
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "<no output>"
            raise RuntimeError(f"tmux command failed (rc={proc.returncode}): {detail}")

    def send_text_sync(self, text: str) -> None:
        """Type literal *text* via ``send-keys -l``, chunked under tmux's cap.

        Synchronous, byte-identical sibling of :meth:`send_text`. The ``-l`` / ``-t``
        flag order follows the bound bridge's dialect
        (:attr:`TmuxDeliveryStyle.literal_flag_before_target`) so cursor's
        ``send-keys -t <target> -l`` picker order is preserved; the default keeps
        claude's ``-l -t <target>`` order.
        """
        for start in range(0, len(text), _SEND_KEYS_LITERAL_CHARS_PER_CALL):
            chunk = text[start : start + _SEND_KEYS_LITERAL_CHARS_PER_CALL]
            if self._delivery_style.literal_flag_before_target:
                self._delivery_run_sync("send-keys", "-t", self._target, "-l", chunk)
            else:
                self._delivery_run_sync("send-keys", "-l", "-t", self._target, chunk)

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        """Press each named key via ``send-keys`` (sync sibling of :meth:`send_keys`)."""
        for key in keys:
            self._delivery_run_sync("send-keys", "-t", self._target, key)

    def paste_without_submit_sync(self, text: str) -> None:
        """Paste *text* as one bracketed paste via a loaded tmux buffer.

        Delivered through a file-backed buffer (``load-buffer`` then
        ``paste-buffer -p``), NOT ``send-keys -l`` argv: tmux caps a single
        client->server command at ~16KB, so a large payload (a PR diff in a
        sub-agent dispatch) failed with "command too long"; ``load-buffer``
        streams the file without that cap. ``-p`` wraps it in bracketed-paste
        markers so interior newlines (encoded to CR by
        :func:`_tmux_paste_payload_bytes`) stay data instead of per-line submits
        (anthropics/claude-code#52126); ``-d`` drops the buffer after pasting so
        no stale copies accumulate server-side.
        """
        paste_buffer = self._delivery_style.paste_buffer
        with tempfile.NamedTemporaryFile(
            dir=self._paste_dir, prefix="paste_", suffix=".bin", delete=False
        ) as paste_file:
            paste_file.write(_tmux_paste_payload_bytes(text))
            paste_path = paste_file.name
        try:
            self._delivery_run_sync("load-buffer", "-b", paste_buffer, paste_path)
            self._delivery_run_sync(
                "paste-buffer",
                "-p",  # bracketed-paste markers — the TUI keeps newlines as data
                "-d",  # drop the buffer after pasting (no stale copies server-side)
                "-b",
                paste_buffer,
                "-t",
                self._target,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(paste_path)

    def kill_session_sync(self) -> None:
        """Hard-stop this session via ``kill-session -t <target>``.

        Kills the session and its pane (terminating the inner CLI). Distinct from
        :meth:`close` (``kill-server``): a private single-session server run with
        ``exit-empty off`` (claude-native's keep-alive) would otherwise outlive a
        session kill, so this targets the session specifically — the exact command
        the bridge's private ``kill_session`` issued.
        """
        self._delivery_run_sync("kill-session", "-t", self._target)

    def delivery_liveness_sync(self) -> bool:
        """Report endpoint existence via ``has-session -t <target>`` (rc==0).

        Byte-identical to the migrated bridges' private ``_session_alive``: one
        ``tmux -S <sock> has-session -t <target>`` client command (no ``-f``, same
        as the other delivery ops), ``True`` on exit 0, and NEVER raises — a
        timeout or spawn error reads as not-alive (``False``), the fast-fail the
        bridge keyed off before injecting into a possibly-dead pane.
        """
        try:
            proc = subprocess.run(
                self._delivery_cmd("has-session", "-t", self._target),
                check=False,
                capture_output=True,
                text=True,
                timeout=self._delivery_style.command_timeout_s,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return proc.returncode == 0

    def send_keys_repeated_sync(self, key: str, count: int) -> None:
        """Press *key* *count* times via one ``send-keys -N <count>`` call.

        Byte-identical to cursor's composer-clear burst
        (``send-keys -t <target> -N <count> <key>``): tmux's native repeat sends
        the key *count* times in a single client command, which the default
        (N separate presses) would not reproduce. A non-positive *count* is a
        no-op, so an empty burst emits no command.
        """
        if count <= 0:
            return
        self._delivery_run_sync("send-keys", "-t", self._target, "-N", str(count), key)

    def send_keys_atomic_sync(self, keys: Sequence[str]) -> None:
        """Press all *keys* in one ``send-keys -t <target> k1 k2 …`` call.

        Byte-identical to goose's packed permission-dialog keystroke
        (``send-keys -t <target> Down Down Enter``): one client command, so the
        sequence is atomic against any other attached tmux client. An empty
        sequence emits no command.
        """
        keys = list(keys)
        if not keys:
            return
        self._delivery_run_sync("send-keys", "-t", self._target, *keys)

    def delivery_snapshot_sync(self) -> str:
        """Snapshot the pane via ``capture-pane -p`` for the delivery dance.

        Byte-identical to the bridges' private ``_capture_pane``: never raises —
        a transient capture failure returns ``""`` (the "not ready yet" signal
        the readiness/verify polls key off) rather than the "host went away"
        exception :meth:`capture_sync` raises. The ``-p`` / ``-t`` flag order
        follows the bound bridge's dialect (:attr:`TmuxDeliveryStyle.
        capture_flag_before_target`) so each migrated bridge's pinned capture argv
        is preserved; the default keeps claude's ``-t <target> -p`` order.
        """
        if self._delivery_style.capture_flag_before_target:
            capture_args = ("capture-pane", "-p", "-t", self._target)
        else:
            capture_args = ("capture-pane", "-t", self._target, "-p")
        try:
            proc = subprocess.run(
                self._delivery_cmd(*capture_args),
                check=False,
                capture_output=True,
                text=True,
                timeout=self._delivery_style.command_timeout_s,
            )
        except (subprocess.SubprocessError, OSError):
            return ""
        return proc.stdout if proc.returncode == 0 else ""

    def native_popup_launch(
        self,
        *,
        config_file: Path,
        session_id: str,
        elicitation_id: str,
        message: str,
        policy_name: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        """Overlay the cost-approval popup on this pane via ``tmux display-popup``.

        Delegates to :func:`omnigent.native_cost_popup.launch_cost_popup` with
        this backend's own socket + target, so the popup renders on the pane the
        delivery surface is bound to. Fire-and-forget (the launcher spawns a
        detached ``Popen`` and skips silently when no client is attached).
        """
        from omnigent.native_cost_popup import launch_cost_popup

        launch_cost_popup(
            str(self._socket_path),
            self._target,
            config_file,
            session_id=session_id,
            elicitation_id=elicitation_id,
            message=message,
            policy_name=policy_name,
            python_executable=python_executable,
        )

    @classmethod
    def reap_orphans(cls) -> int:
        """
        Kill terminal tmux servers whose owning process is gone.

        Terminal tmux servers are deliberately detached so they survive
        transient client disconnects; graceful shutdown closes them
        (``TerminalRegistry.shutdown``), but a SIGKILL'd runner — or one whose
        whole process group is torn down by a test harness — leaks them
        forever, one per session now that runner-bound SDK sessions
        auto-create the embedded REPL terminal. Each instance dir records its
        owner pid at creation; this sweep (run at runner startup) kills the
        tmux server of every instance whose owner no longer exists and removes
        the instance dir. Dirs without an owner-pid marker are left untouched
        — they are either from an older version or not ours.

        :returns: The number of orphaned instance dirs reaped.
        """
        if not _tmux_available():
            return 0
        reaped = 0
        for entry in _terminals_tmp_root().glob(f"{_TERMINAL_DIR_PREFIX}*"):
            try:
                pid = int((entry / _OWNER_PID_FILENAME).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            if _process_alive(pid):
                continue
            socket_path = entry / "tmux.sock"
            if socket_path.exists():
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    subprocess.run(
                        ["tmux", "-S", str(socket_path), "kill-server"],
                        # kill-server on an already-dead server exits non-zero;
                        # that is the common case for half-torn-down orphans.
                        check=False,
                        capture_output=True,
                        timeout=_REAP_KILL_TIMEOUT_S,
                    )
            shutil.rmtree(entry, ignore_errors=True)
            reaped += 1
        return reaped


class HerdrBackend(TerminalBackend):
    """
    herdr multiplexer backend (native Windows).

    Hosts each managed terminal in an omnigent-scoped herdr *session* (a
    server-per-named-session), one durable-labeled *workspace* per terminal, one
    *tab*/*pane* running the inner command. Every operation is driven purely
    through the herdr **CLI as a subprocess** — no protocol library is linked or
    vendored (a deliberate licensing posture) — and every invocation carries an
    explicit ``--session`` (see :meth:`_base_argv`). All herdr vocabulary — the
    ``--session`` target, ``workspace``/``agent``/``pane`` subcommands, plus-
    notation keys (``ctrl+c``), the default ``{"id": .., "result": ..}`` JSON
    envelope, CRLF captures — is confined to this class.

    **Session lifecycle (#11):** launch (with husk adopt/replace), liveness,
    close/kill, orphan reaping, geometry pinning, Windows path translation, and
    CRLF stripping. **Server bring-up (#13):** a named session's server does not
    auto-start, so :meth:`launch` starts it headless before the first socket verb
    (:meth:`_ensure_server_running`); stopping the idle server and crash-safe
    server reaping remain #15.

    **I/O operations (#12):** non-submitting multi-line paste (``pane
    send-text``), atomic submit / named-key delivery with the full
    key-translation table (:meth:`_translate_key`), plain + ANSI screen
    snapshots (``pane read --format text|ansi``) with the min-line-count
    workaround (:meth:`_read_argv` / :meth:`_tail_lines`), native busy-state
    corroborated with output-diff (:meth:`busy_state`), and composer
    input-readiness (:meth:`input_ready`). The protocol probe reads ``herdr api
    schema`` (:meth:`ensure_available`), and every pane subcommand addresses the
    pane positionally (:meth:`_pane_argv`).

    **Real-herdr reconciliation (#13).** The #11-designed launch verbs did not
    match real herdr and are reconciled in :meth:`launch`: ``workspace create``
    and ``tab create`` take no command/geometry and the socket-API commands emit
    their JSON envelope by default (no ``--format`` — an unknown flag is
    rejected), so the inner command is now spawned with ``agent start <name> --
    <argv>`` (the spike-verified verb) and its pane isolated by a ``pane list``
    diff. The ``pane get`` envelope is reconciled to the real shape: success is
    ``{"id": "cli:pane:get", "result": {"pane": {"agent_status": ...}, "type":
    "pane_info"}}`` (``type`` a semantic constant, not the verb), a DEAD pane an
    *error* envelope ``{"error": {"code": "pane_not_found", ...}}`` with process
    exit code 1 — so ``pane_not_found`` + exit-1 is the ENDPOINT_GONE signal
    (:meth:`_interpret_pane_get`), and the native status is read at
    ``.result.pane.agent_status`` (:meth:`_agent_status`). Write verbs print
    nothing on success and only an error envelope on failure — already matched
    (:meth:`_run` ignores stdout and gates on exit code). **Live-run-only
    (verified via ``--help`` syntax, not exercised — the tech lead's live run):**
    the exact ``.result`` id-paths of ``workspace create`` / ``agent start`` /
    ``pane list`` (the adapter avoids depending on them by re-listing by label and
    diffing pane ids), whether ``agent start`` reuses the workspace root pane or
    adds one, ~52-col headless geometry, and ``pane send-text`` reading a very
    large paste (the OS argv length cap — see :meth:`send_text`). Explicit
    headless ``server`` management and threading the full inner-process
    environment through that server are #15 integration concerns; see
    :meth:`launch`.

    **No remain-on-exit.** herdr auto-destroys a pane when its process exits and
    the ``pane_exited`` event carries no exit code, so a dead endpoint cannot be
    preserved: an inner exit maps to :attr:`Liveness.ENDPOINT_GONE` (there is no
    reachable :attr:`Liveness.INNER_EXITED` for herdr) and
    :attr:`TerminalLaunchRequest.keep_alive_after_exit` is ignored.
    """

    name = "herdr"
    capabilities = TerminalBackendCapabilities(
        # No native popup surface: the web approval card is the elicitation
        # surface on Windows.
        native_popup=False,
        # start-on-attach is not modeled in #11 (no attach transport migrated
        # yet); a watcher loop emulates delayed start where needed.
        start_on_attach=False,
        # herdr exposes a native agent busy/idle signal (``agent_status`` —
        # claude auto-detected in the spike). The flag is declared now so
        # machinery above the seam can prefer it; the actual wiring is #12.
        native_busy_state=True,
        # ``pane_output_changed`` events are reachable only via the session
        # socket, not the CLI, so the CLI-only adapter cannot push deltas —
        # polling is the truth here.
        push_events=False,
        # No control-mode attach transport is offered by the CLI adapter.
        control_mode_attach=False,
        attach_transports=frozenset({TERMINAL_TRANSPORT_SNAPSHOT}),
        # No host status line to carry the cosmetic conversation link, so
        # :meth:`set_status_link` stays the inherited no-op (the link is
        # droppable — the web UI is the primary surface).
        status_line=False,
    )
    # The spike validated herdr on native Windows only; POSIX keeps tmux.
    platforms = frozenset({"windows"})

    # Env-var override for the herdr binary (see :meth:`_command_prefix`).
    BIN_ENV_VAR = "OMNIGENT_HERDR_BIN"
    DEFAULT_BIN = "herdr"
    # Gate on the wire PROTOCOL number, not the version string: herdr's version
    # moved (0.7.1 → 0.7.4-preview) without the protocol moving off 16, so the
    # protocol is the stable compatibility axis.
    MIN_PROTOCOL = 16
    # omnigent-scoped prefixes so a derived session/label can NEVER collide with
    # herdr's ``default`` session (the user's live panes).
    _SESSION_PREFIX = "omnigent-"
    _LABEL_PREFIX = "omnigent-ws-"
    # Prefix for the omnigent-scoped herdr *agent name* passed to ``agent start``
    # (the spike-verified spawn verb). Scoped so it can never collide with a
    # user's own detected/named agents.
    _AGENT_PREFIX = "omnigent-agent-"
    # A throwaway omnigent-scoped session for the version/protocol probe, so even
    # the session-independent probe carries an explicit ``--session`` (uniform
    # "never bare" posture — nothing the adapter emits can target ``default``).
    _PROBE_SESSION = "omnigent-probe"
    _CLI_TIMEOUT_S = 15.0
    # A named herdr session's server does NOT auto-start: the first socket-API
    # verb against a session with no server fails with an OS NotFound. :meth:`launch`
    # starts it headless and waits up to this long for the socket to answer,
    # polling at this interval (a headless server binds its socket in ~1 s).
    _SERVER_READY_TIMEOUT_S = 10.0
    _SERVER_POLL_INTERVAL_S = 0.2
    # Neutral key names herdr has no equivalent for; skipped rather than sent as
    # a wrong key. Covers both the plain spellings and tmux's aliases for the
    # same keys (``PPage``/``NPage`` = PageUp/PageDown, ``DC``/``IC`` =
    # Delete/Insert) because Omnigent's neutral vocabulary is tmux's key names.
    # The spike confirmed herdr rejects every one of these as ``invalid_key``.
    _UNSUPPORTED_KEYS = frozenset(
        {
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "Delete",
            "Insert",
            "PPage",
            "NPage",
            "DC",
            "IC",
        }
    )
    # Explicit neutral-name → herdr-name renames for named keys whose herdr
    # spelling differs from Omnigent's tmux-derived vocabulary. ``BSpace`` (tmux)
    # is ``Backspace`` in herdr (herdr rejects ``BSpace``); ``BTab`` (tmux
    # back-tab) is the ``shift+tab`` chord. Everything else herdr supports
    # (``Enter``/``Escape``/``Tab``/``Space``/arrows/``F1``..) shares Omnigent's
    # spelling and passes through unchanged.
    _KEY_RENAMES = {"BSpace": "Backspace", "BTab": "shift+tab"}  # noqa: RUF012 (read-only)
    # Native ``agent_status`` values that mean the agent is mid-turn (busy):
    # ``working`` (running) and ``blocked`` (paused on input/approval, still in a
    # turn). ``idle`` (at the composer) and ``done`` (turn finished) are not
    # busy; anything else (incl. ``unknown``/missing) is treated as *no native
    # signal* and falls back to the output-diff heuristic. See :meth:`busy_state`.
    _NATIVE_BUSY_STATES = frozenset({"working", "blocked"})
    _NATIVE_IDLE_STATES = frozenset({"idle", "done"})
    # Defensive floor for ``pane read --lines``. The historical small-N empty-read
    # quirk (a ``--lines`` below some threshold returning an empty capture) did
    # NOT reproduce on herdr 0.7.4, but the ticket mandates the workaround as
    # defense-in-depth: never ask herdr for fewer than this many logical lines,
    # then tail locally to the size the caller actually requested (see
    # :meth:`capture`). Large enough to clear any plausible small-N threshold yet
    # cheap at ~31 ms/read.
    _SNAPSHOT_MIN_FETCH_LINES = 500

    @classmethod
    def ensure_available(cls) -> None:
        """Verify the herdr binary is present AND speaks a supported protocol.

        Two distinct, loud, actionable failures: a missing/unspawnable binary
        (names the binary and the :envvar:`OMNIGENT_HERDR_BIN` override), and an
        unsupported wire protocol (names the binary, the protocol found, and the
        minimum required).

        The protocol is discovered by running ``herdr api schema`` — a static,
        session-independent schema dump (``herdr status`` exposes the same number
        less conveniently, and ``herdr --version`` prints only the version
        *string*, never the protocol). The **real binary's default output is
        human-readable text** with a ``protocol: <N>`` line (``herdr api schema
        --json`` prints the full schema as JSON) — verified directly, correcting
        the spike's shorthand that implied JSON — so the protocol is extracted
        with a small regex tolerant of both the text ``protocol: 16`` and a JSON
        ``"protocol": 16``. ``api schema`` cannot touch any session's panes, but
        the probe still leads with an explicit ``--session``
        (:data:`_PROBE_SESSION`) as a defensive global so nothing the adapter
        emits is ever a bare subcommand — confirmed against the real binary,
        which accepts the leading ``--session`` and still returns the schema.

        :raises RuntimeError: When herdr is not installed/spawnable, when the
            ``api schema`` probe fails or has no parseable protocol, or when the
            reported protocol is below :data:`MIN_PROTOCOL`.
        """
        argv = [
            *cls._command_prefix(),
            "--session",
            cls._PROBE_SESSION,
            "api",
            "schema",
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, check=False, timeout=cls._CLI_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"herdr is not installed or not spawnable ({cls._command_prefix()[0]!r}: "
                f"{exc}). The herdr terminal backend hosts native Windows harnesses "
                f"by driving the herdr CLI; install herdr and ensure it is on PATH, "
                f"or point {cls.BIN_ENV_VAR} at the herdr binary."
            ) from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"herdr 'api schema' probe failed (rc={proc.returncode}): "
                f"{proc.stderr.decode(errors='replace').strip()}. Ensure "
                f"{cls._command_prefix()[0]!r} is a working herdr binary "
                f"(override with {cls.BIN_ENV_VAR})."
            )
        text = cls._normalize_newlines(proc.stdout.decode(errors="replace"))
        # Tolerant of the real text form (``protocol: 16``) and a JSON form
        # (``"protocol": 16``): match the label, an optional closing quote, the
        # colon, then the number.
        match = re.search(r'protocol"?\s*:\s*(\d+)', text)
        if match is None:
            raise RuntimeError(
                f"could not determine herdr protocol from its 'api schema' report "
                f"({text!r}). Ensure {cls._command_prefix()[0]!r} is a working herdr "
                f"binary (override with {cls.BIN_ENV_VAR})."
            )
        protocol = int(match.group(1))
        if protocol < cls.MIN_PROTOCOL:
            raise RuntimeError(
                f"herdr protocol {protocol} is unsupported: the herdr backend "
                f"requires protocol >= {cls.MIN_PROTOCOL} (binary "
                f"{cls._command_prefix()[0]!r}). Upgrade herdr."
            )

    @classmethod
    def construct_for_instance(cls, *, socket_path: Path, target: str) -> TerminalBackend:
        """Build a herdr backend bound to a terminal instance's endpoint.

        The construction hook :func:`_construct_terminal_backend` prefers, so
        herdr becomes constructible with no dispatcher edit (the seam the
        in-process ``FakeBackend`` already uses).
        """
        backend = cls(socket_path=socket_path)
        if target != "main":
            backend._pane_id = target
        return backend

    def __init__(self, *, socket_path: Path, target: str = "main") -> None:
        """
        :param socket_path: Private per-instance endpoint path. The herdr
            ``--session`` name and workspace label are derived deterministically
            from it (see :meth:`_session_name` / :meth:`_workspace_label`) so
            they are unique per terminal and omnigent-scoped.
        :param target: Session/pane target name, e.g. ``"main"``; folded into the
            durable workspace label.
        """
        self._socket_path = socket_path
        self._target = target
        self._session = self._session_name(socket_path)
        self._label = self._workspace_label(socket_path, target)
        self._agent = self._agent_name(socket_path)
        # Populated by :meth:`launch`; used by liveness/capture/input.
        self._workspace_id: str | None = None
        self._tab_id: str | None = None
        self._pane_id: str | None = None
        self._closed = False
        # Last plain snapshot seen by :meth:`busy_state`, for the output-diff
        # corroboration heuristic (native ``agent_status`` reads idle during long
        # foreground tool calls, so a changing screen overrides a native idle).
        self._last_activity_snapshot: str | None = None

    def advertised_target(self, *, fallback: str) -> str:
        """Return the live herdr pane id used by delivery commands."""
        return self._pane_id or fallback

    # ------------------------------------------------------------- derivation

    @classmethod
    def _session_name(cls, socket_path: Path) -> str:
        """Derive this instance's omnigent-scoped herdr ``--session`` name.

        A stable short hash of the private socket path: unique per terminal
        instance and, by the :data:`_SESSION_PREFIX`, guaranteed distinct from
        herdr's ``default`` session (the user's live panes).
        """
        import hashlib

        digest = hashlib.sha1(str(socket_path).encode("utf-8")).hexdigest()[:12]
        return f"{cls._SESSION_PREFIX}{digest}"

    @classmethod
    def _workspace_label(cls, socket_path: Path, target: str) -> str:
        """Derive this terminal's durable workspace label.

        Deterministic in ``(socket_path, target)`` so a restart that reuses the
        same private endpoint sees the same label and can adopt/replace the
        leftover as a husk (see :meth:`launch`). With per-launch socket paths the
        label is effectively unique and husk adoption is a safe no-op.
        """
        import hashlib

        slug = re.sub(r"[^A-Za-z0-9]+", "-", target).strip("-").lower() or "main"
        digest = hashlib.sha1(str(socket_path).encode("utf-8")).hexdigest()[:12]
        return f"{cls._LABEL_PREFIX}{digest}-{slug}"

    @classmethod
    def workspace_label_for(cls, socket_path: Path, target: str = "main") -> str:
        """Public deriver for a terminal's durable herdr workspace label.

        The label (``omnigent-ws-<hash>-<slug>``) is a pure function of the
        terminal's private endpoint and target, so a client that only has the
        terminal resource's socket path + target (e.g. the ``omnigent codex`` CLI
        pointing a Windows user at the herdr GUI, where the equivalent of ``tmux
        attach`` is opening the herdr app and finding this workspace by its label)
        can recompute the exact label without reaching into the running backend.
        Delegates to :meth:`_workspace_label`.
        """
        return cls._workspace_label(socket_path, target)

    @classmethod
    def _agent_name(cls, socket_path: Path) -> str:
        """Derive this terminal's omnigent-scoped herdr agent name.

        The inner command is launched with ``agent start <name> -- <argv>`` (the
        spike-verified spawn verb — real herdr's ``tab create`` cannot run a
        command; see :meth:`launch`). The name is a stable short hash of the
        private endpoint, :data:`_AGENT_PREFIX`-scoped so it is unique per
        terminal and never collides with a user's own agents.
        """
        import hashlib

        digest = hashlib.sha1(str(socket_path).encode("utf-8")).hexdigest()[:12]
        return f"{cls._AGENT_PREFIX}{digest}"

    @staticmethod
    def _result_payload(data: dict[str, Any]) -> dict[str, Any]:  # type: ignore[explicit-any]
        """Unwrap real herdr's ``{"id": .., "result": {..}}`` success envelope.

        Real herdr wraps every socket-API success as ``{"id": "cli:<group>:<verb>",
        "result": {<payload>, "type": "<semantic_const>"}}`` (spike-verified). The
        adapter reads the inner ``result`` payload. Tolerant of a bare (already
        unwrapped) mapping so a probe result that is not enveloped still parses —
        which keeps the mapping robust if a future herdr flattens a verb.
        """
        result = data.get("result")
        return result if isinstance(result, dict) else data

    @classmethod
    def _workspaces_of(cls, data: dict[str, Any]) -> list[dict[str, Any]]:  # type: ignore[explicit-any]
        """Return the ``workspaces`` list from a ``workspace list`` envelope.

        Each entry's id field is ``workspace_id`` (verified against real herdr
        0.7.4 — NOT a bare ``id``; the entry also carries ``label``, ``number``,
        ``pane_count``, ``active_tab_id``, etc., which callers ignore).
        """
        workspaces = cls._result_payload(data).get("workspaces")
        if not isinstance(workspaces, list):
            return []
        return [ws for ws in workspaces if isinstance(ws, dict)]

    @classmethod
    def _pane_ids_of(cls, data: dict[str, Any]) -> list[str]:  # type: ignore[explicit-any]
        """Return the pane ids from a ``pane list`` envelope, in listed order.

        The id field is ``pane_id`` (verified against real herdr 0.7.4 — a
        namespaced ``wN:pN``, NOT a bare ``id``; entries also carry ``tab_id``,
        ``workspace_id``, ``terminal_id``, ``cwd``, etc., which callers ignore).
        """
        panes = cls._result_payload(data).get("panes")
        if not isinstance(panes, list):
            return []
        return [
            p["pane_id"]
            for p in panes
            if isinstance(p, dict) and isinstance(p.get("pane_id"), str)
        ]

    @staticmethod
    def _to_windows_path(cwd: str) -> str:
        """Return *cwd* as a Windows-native (backslash, drive-letter) path.

        herdr runs on native Windows and expects Windows-native paths; a
        forward-slash spelling (from config or a POSIX default) is converted
        deterministically via :class:`~pathlib.PureWindowsPath` — host-
        independent, so it is unit-testable on any platform.
        """
        from pathlib import PureWindowsPath

        return str(PureWindowsPath(cwd))

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """Normalize CRLF/CR line endings to ``\\n``.

        herdr captures on Windows carry CRLF; the backend strips that quirk so
        callers above the seam never branch on line endings (the
        :meth:`TerminalBackend.capture` contract). Applied to every parsed CLI
        output, not just screen captures.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @classmethod
    def _translate_key(cls, key: str) -> str | None:
        """Translate an Omnigent neutral key name into herdr's key syntax.

        Omnigent's neutral vocabulary is tmux's key names (the seam contract);
        herdr's is different, so every key is translated. The full table,
        grounded in the spike's verified ``send-keys`` surface:

        - **Chords** use plus-notation, not tmux's dash form: ``C-x`` →
          ``ctrl+x``, ``M-x`` → ``alt+x``, ``S-x`` → ``shift+x`` (herdr rejects
          the dash form for everything except ``C-c``, but translating uniformly
          means the accepted ``ctrl+c`` is always what we emit).
        - **Renamed named keys** (:data:`_KEY_RENAMES`): ``BSpace`` →
          ``Backspace`` (herdr rejects ``BSpace``), ``BTab`` → ``shift+tab``.
        - **Pass-through named keys** herdr shares with Omnigent:
          ``Enter``/``Escape``/``Tab``/``Space``/``Up``/``Down``/``Left``/
          ``Right``/``F1``.. — returned unchanged.
        - **Unsupported keys** (:data:`_UNSUPPORTED_KEYS`:
          Home/End/PageUp/PageDown/Delete/Insert and their tmux aliases) are
          skipped (returns ``None``) rather than sent as a wrong key.

        :param key: A neutral key name, e.g. ``"Enter"``, ``"C-c"``, ``"BTab"``.
        :returns: The herdr key token, or ``None`` to skip an unsupported key.
        """
        if key in cls._UNSUPPORTED_KEYS:
            return None
        if key in cls._KEY_RENAMES:
            return cls._KEY_RENAMES[key]
        if key.startswith("C-") and len(key) > 2:
            return "ctrl+" + key[2:]
        if key.startswith("M-") and len(key) > 2:
            return "alt+" + key[2:]
        if key.startswith("S-") and len(key) > 2:
            return "shift+" + key[2:]
        return key

    # ------------------------------------------------------------- CLI plumbing

    @classmethod
    def _command_prefix(cls) -> list[str]:
        """Return the argv prefix that invokes the herdr CLI.

        Read from :envvar:`OMNIGENT_HERDR_BIN` at call time (default
        ``"herdr"``). Normally a single binary path; tests point it at a scripted
        fake by encoding a multi-token launcher as a JSON array (e.g.
        ``'["/usr/bin/python", "/path/_fake_herdr.py"]'``) so an
        interpreter+script can stand in for the binary without a ``.cmd``/PATHEXT
        shim — unreliable under Windows ``CreateProcess``. A non-array value, a
        JSON parse failure, or a non-string element falls back to treating the
        whole value as one binary path.
        """
        import json

        raw = os.environ.get(cls.BIN_ENV_VAR, cls.DEFAULT_BIN).strip() or cls.DEFAULT_BIN
        if raw.startswith("["):
            try:
                tokens = json.loads(raw)
            except json.JSONDecodeError:
                return [raw]
            if isinstance(tokens, list) and tokens and all(isinstance(t, str) for t in tokens):
                return tokens
        return [raw]

    def _base_argv(self) -> list[str]:
        """Return the herdr argv prefix carrying this instance's ``--session``.

        Every invocation leads with ``--session <name>`` before the subcommand.
        A bare herdr subcommand targets the ``default`` session — the user's LIVE
        panes — and ambient targeting has destroyed live sessions in prior art;
        the derived session name is omnigent-scoped so it can never be
        ``default``.
        """
        return [*self._command_prefix(), "--session", self._session]

    async def _run(self, *args: str, stdin_data: bytes | None = None) -> None:
        """Run a herdr command against this session; raise on non-zero exit."""
        proc = await asyncio.create_subprocess_exec(
            *self._base_argv(),
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await self._communicate(proc, args, stdin_data)
        if proc.returncode != 0:
            raise RuntimeError(
                f"herdr command failed: {' '.join(args)}: "
                f"{self._normalize_newlines(stderr.decode(errors='replace')).strip()}"
            )

    async def _run_output(self, *args: str) -> str:
        """Run a herdr command against this session and return its stdout."""
        proc = await asyncio.create_subprocess_exec(
            *self._base_argv(),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await self._communicate(proc, args)
        if proc.returncode != 0:
            raise RuntimeError(
                f"herdr command failed: {' '.join(args)}: "
                f"{self._normalize_newlines(stderr.decode(errors='replace')).strip()}"
            )
        return stdout.decode(errors="replace")

    async def _communicate(
        self,
        proc: asyncio.subprocess.Process,
        args: Sequence[str],
        stdin_data: bytes | None = None,
    ) -> tuple[bytes, bytes]:
        """Communicate with a herdr child within the CLI timeout."""
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data), timeout=self._CLI_TIMEOUT_S
            )
            return stdout or b"", stderr or b""
        except asyncio.TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._CLI_TIMEOUT_S)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            raise RuntimeError(
                f"herdr command timed out after {self._CLI_TIMEOUT_S:.0f}s: {' '.join(args)}"
            ) from exc

    async def _run_json(self, *args: str) -> dict[str, Any]:  # type: ignore[explicit-any]
        """Run a herdr socket-API command and parse its JSON envelope.

        Real herdr's ``workspace``/``tab``/``pane``/``agent`` socket-API commands
        emit a ``{"id": "cli:<group>:<verb>", "result": {..}}`` JSON envelope by
        default — there is no ``--format json`` flag (an unknown flag is
        rejected). Callers unwrap the ``result`` payload via
        :meth:`_result_payload` / :meth:`_workspaces_of` / :meth:`_pane_ids_of`.
        """
        import json

        raw = await self._run_output(*args)
        return json.loads(self._normalize_newlines(raw))

    def _run_output_sync(self, *args: str) -> str:
        """Synchronous sibling of :meth:`_run_output` for the threaded watcher."""
        proc = subprocess.run(
            [*self._base_argv(), *args],
            capture_output=True,
            check=False,
            timeout=self._CLI_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"herdr command failed: {' '.join(args)}: "
                f"{self._normalize_newlines(proc.stderr.decode(errors='replace')).strip()}"
            )
        return proc.stdout.decode(errors="replace")

    # ------------------------------------------------------------- server mgmt

    async def _server_responsive(self) -> bool:
        """Return whether this session's herdr server answers a socket verb.

        A cheap read-only ``workspace list`` against the session: exit 0 means the
        server is up and reachable; a non-zero exit (the OS NotFound when no
        server is bound) or a spawn failure means not yet.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._base_argv(),
                "workspace",
                "list",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            returncode = await asyncio.wait_for(proc.wait(), timeout=self._CLI_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            return False
        return returncode == 0

    async def _ensure_server_running(self) -> None:
        """Start this session's headless herdr server and wait until it answers.

        A named herdr session's server does not auto-start — the first socket-API
        verb against a serverless session fails with an OS NotFound (``Os { code:
        2, kind: NotFound }``), the failure #13's first real-herdr contact hit. So
        ``launch`` starts the server explicitly with ``herdr --session <s> server``
        (headless, long-lived) BEFORE any socket verb.

        **Idempotent** and cheap on the hot path: if the server is already up (a
        husk restart on the same endpoint) the fast-path check returns immediately
        and no second server is spawned; a redundant real ``server`` start would in
        any case exit with "already running" (verified), which the readiness poll
        simply rides through. The server is spawned in its own process group
        (:func:`_proc.spawn_kwargs`) so it is not tied to this call, and its output
        is discarded (herdr keeps its own log file).

        **Lifetime / ownership.** The session is omnigent-scoped and per-endpoint
        (:meth:`_session_name`), so its server is exclusively this terminal's. It
        is intentionally NOT stopped in :meth:`close`: closing the workspace makes
        liveness read the clean ``workspace_not_found`` ENDPOINT_GONE signal (a
        *stopped server* would instead give an ambiguous NotFound), and a later
        restart on the same endpoint reuses the still-running server. Stopping the
        idle server on close and crash-safe server reaping (the herdr analog of
        :func:`reap_orphaned_terminals`) are the residual #15 "explicit headless
        server management" items — the leaked server is an empty idle process, and
        the inner pane's process is already reaped by the workspace close.

        :raises RuntimeError: If the server does not answer within
            :data:`_SERVER_READY_TIMEOUT_S`.
        """
        if await self._server_responsive():
            return
        # Fire-and-forget headless server: a plain (non-awaited) Popen because it
        # is long-lived — its own process group, output discarded.
        with contextlib.suppress(OSError):
            subprocess.Popen(
                [*self._command_prefix(), "--session", self._session, "server"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_proc.spawn_kwargs(),
            )
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._SERVER_READY_TIMEOUT_S
        while loop.time() < deadline:
            if await self._server_responsive():
                return
            await asyncio.sleep(self._SERVER_POLL_INTERVAL_S)
        raise RuntimeError(
            f"herdr session server for {self._session!r} did not become ready within "
            f"{self._SERVER_READY_TIMEOUT_S:.0f}s (a headless 'herdr --session <s> server' "
            f"could not be started or did not bind its socket)."
        )

    # ---------------------------------------------------------------- protocol

    async def launch(self, request: TerminalLaunchRequest) -> None:
        """Create the workspace + agent pane for *request* (adopt-or-replace husks).

        Reconciled against real herdr 0.7.4 in #13. The #11-designed
        ``tab create --cols --rows --command -- <argv>`` launch path does **not
        exist** in real herdr — ``tab create`` takes no command or geometry, and
        the socket-API commands reject an unknown ``--format`` flag (they emit
        their ``{"id": .., "result": ..}`` JSON envelope by default). The real,
        spike-verified spawn sequence the adapter now uses:

        1. ``workspace create --label <L> --cwd <winpath> --no-focus`` — creates a
           labeled workspace (herdr auto-spawns a root shell pane in it). The new
           workspace is then re-found by label in ``workspace list`` rather than
           parsed out of the create envelope, so the adapter does not depend on
           the (live-only) shape of the create result.
        2. ``agent start <name> --workspace <id> --cwd <winpath> --no-focus --
           <argv>`` — runs the inner command as a herdr *agent* (the spike
           launched real Claude Code this way; it also arms native agent-status
           detection). The command's pane is identified by diffing ``pane list``
           for the workspace across the ``agent start`` call, so no assumption is
           made about the ``agent start`` result envelope.

        Ordering stays **create-before-close**: the fresh workspace/pane come up
        FIRST, then any same-label husks (restart leftovers) are closed, so a
        crash mid-launch never strands this terminal with zero workspaces.

        Geometry pinning is **dropped**: real ``pane resize`` is *relative*
        (``--direction``/``--amount``), so an absolute ``--cols``/``--rows`` at
        spawn is not expressible; headless panes come up ~52 columns and the
        capture path relies on ``--source recent-unwrapped`` for logical lines
        (see :meth:`_read_argv`). :attr:`~TerminalLaunchRequest.keep_alive_after_exit`
        is ignored (herdr has no remain-on-exit). The inner process environment
        (:attr:`TerminalLaunchRequest.env`, already harness-filtered — codex needs
        only ~3 vars: CODEX_HOME + the optional Databricks pair) is threaded onto
        the pane as repeated ``agent start ... --env KEY=VALUE`` flags (before the
        ``--`` marker); the OS argv cap for a pathologically large env is a
        documented limitation, and threading a full merged env through a
        per-session server stays a #15 integration concern.

        :param request: The backend-neutral launch request.
        :raises RuntimeError: If herdr rejects the workspace/agent creation, or if
            the session's headless server cannot be started (see
            :meth:`_ensure_server_running`).
        """
        self._closed = False
        # A named session's server does not auto-start; bring it up before the
        # first socket-API verb (else ``workspace list`` fails with an OS NotFound).
        await self._ensure_server_running()
        listing = await self._run_json("workspace", "list")
        husks = [
            ws["workspace_id"]
            for ws in self._workspaces_of(listing)
            if ws.get("label") == self._label and ws.get("workspace_id") is not None
        ]

        win_cwd = self._to_windows_path(request.cwd)
        # ``workspace create`` emits its JSON envelope by default; no ``--format``
        # (real herdr rejects the unknown flag). Re-find the new workspace by
        # label so we never depend on the create result's id path.
        await self._run(
            "workspace", "create", "--label", self._label, "--cwd", win_cwd, "--no-focus"
        )
        after_create = self._workspaces_of(await self._run_json("workspace", "list"))
        mine = [
            ws["workspace_id"]
            for ws in after_create
            if ws.get("label") == self._label
            and ws.get("workspace_id") not in husks
            and ws.get("workspace_id")
        ]
        if not mine:
            raise RuntimeError(
                f"herdr 'workspace create' did not yield a workspace labeled {self._label!r}"
            )
        self._workspace_id = mine[-1]

        # Panes already in the workspace (herdr auto-spawned a root shell pane);
        # ``agent start`` adds the inner-command pane, which we isolate by diff.
        before = set(
            self._pane_ids_of(
                await self._run_json("pane", "list", "--workspace", self._workspace_id)
            )
        )
        # Thread the (harness-filtered) environment onto the pane as repeated
        # ``--env KEY=VALUE`` flags — the spike-verified per-pane env vehicle.
        # These MUST precede the ``--`` marker (everything after ``--`` is the
        # inner argv, so no herdr flag may follow it). ``request.env`` is already
        # the small env the harness needs (codex threads ~3 vars — CODEX_HOME +
        # the optional Databricks pair), so per-pane ``--env`` fits; a
        # pathologically large ``request.env`` is bounded by the OS argv cap
        # (documented limitation), and the full-merged-env / per-session-server
        # vehicle stays a #15 integration concern.
        env_flags: list[str] = []
        for key, value in request.env.items():
            env_flags += ["--env", f"{key}={value}"]
        await self._run(
            "agent",
            "start",
            self._agent,
            "--workspace",
            self._workspace_id,
            "--cwd",
            win_cwd,
            "--no-focus",
            *env_flags,
            "--",
            *request.command,
        )
        after = self._pane_ids_of(
            await self._run_json("pane", "list", "--workspace", self._workspace_id)
        )
        new_panes = [pid for pid in after if pid not in before]
        # The agent's pane is the one that appeared; fall back to the last pane in
        # the workspace if the spawn reused the root pane rather than adding one.
        self._pane_id = new_panes[-1] if new_panes else (after[-1] if after else None)
        if self._pane_id is None:
            raise RuntimeError("herdr 'agent start' did not create a pane for the inner command")

        # Create-before-close: only now retire the husks.
        for husk_id in husks:
            with contextlib.suppress(RuntimeError):
                await self._run("workspace", "close", husk_id)

    def _pane_argv(self, action: str, *args: str) -> list[str]:
        """Build a ``pane <action> <pane-id> [args...]`` suffix (no session prefix).

        The pane id is a **positional** argument immediately after the action —
        the spike-verified form (``pane read w1:p1 ...``, ``pane get <id>``,
        ``pane send-text <id> <text>``, ``pane send-keys <id> <key...>``), not a
        ``--pane <id>`` flag. Centralized here so every pane subcommand addresses
        the pane identically. Returns only the subcommand suffix; callers that
        run through :meth:`_run` / :meth:`_run_output` get the leading
        ``--session`` prefix prepended for them, while direct-``subprocess``
        callers wrap it with :meth:`_base_argv` themselves.
        """
        return ["pane", action, self._pane_id or "", *args]

    def _pane_get_argv(self) -> list[str]:
        """Build the full ``pane get`` argv used by both liveness probes.

        Real herdr's ``pane get <id>`` emits its ``{"id": .., "result": ..}`` JSON
        envelope by default — no ``--format`` (the unknown flag would be
        rejected). The pane id is positional. This one carries the ``--session``
        prefix because it is handed straight to ``subprocess`` (not through
        :meth:`_run`).
        """
        return [*self._base_argv(), *self._pane_argv("get")]

    @classmethod
    def _parse_envelope(cls, stdout: bytes, stderr: bytes = b"") -> dict[str, Any] | None:  # type: ignore[explicit-any]
        """Parse a herdr CLI JSON envelope from stdout, falling back to stderr.

        herdr prints a success envelope on stdout and — for a failed command —
        an ``{"error": {"code": ..}}`` envelope (which may ride on stderr, with a
        non-zero exit). Try stdout first, then stderr; return the first mapping
        that parses, else ``None``.
        """
        import json

        for raw in (stdout, stderr):
            if not raw:
                continue
            try:
                data = json.loads(cls._normalize_newlines(raw.decode(errors="replace")))
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @classmethod
    def _interpret_pane_get(
        cls, returncode: int | None, stdout: bytes, stderr: bytes = b""
    ) -> Liveness:
        """Map a real-herdr ``pane get`` result to a :class:`Liveness` verdict.

        Reconciled to real herdr's envelope in #13. A **live** pane is a success
        envelope ``{"id": "cli:pane:get", "result": {"pane": {..}, "type":
        "pane_info"}}`` with exit 0 → :attr:`Liveness.ALIVE`. A **dead** pane (the
        inner process exited and herdr destroyed the pane) or a gone workspace is
        an *error* envelope ``{"error": {"code": "pane_not_found" |
        "workspace_not_found", ..}}`` **with process exit code 1** →
        :attr:`Liveness.ENDPOINT_GONE`. #11 mapped any non-zero exit to UNKNOWN,
        which would misread a real dead pane; the error envelope is now parsed on
        a non-zero exit (from stdout or stderr) so the ENDPOINT_GONE signal is not
        lost. Anything that leaves no clean verdict — an unparseable output, an
        unrecognized error code — degrades to :attr:`Liveness.UNKNOWN` (never a
        false ENDPOINT_GONE).
        """
        data = cls._parse_envelope(stdout, stderr)
        if data is None:
            return Liveness.UNKNOWN
        error = data.get("error")
        if isinstance(error, dict):
            if error.get("code") in ("pane_not_found", "workspace_not_found"):
                return Liveness.ENDPOINT_GONE
            return Liveness.UNKNOWN
        result = cls._result_payload(data)
        if returncode == 0 and isinstance(result.get("pane"), dict):
            return Liveness.ALIVE
        return Liveness.UNKNOWN

    async def liveness(self) -> Liveness:
        """Probe the pane and map it to a verdict (see :meth:`_interpret_pane_get`).

        INNER_EXITED is unreachable for herdr: with no remain-on-exit an exited
        pane is destroyed, so its verdict is ENDPOINT_GONE, not INNER_EXITED.
        stderr is captured (not discarded) because a dead pane's error envelope
        can ride there.
        """
        if self._closed:
            return Liveness.ENDPOINT_GONE
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._pane_get_argv(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._CLI_TIMEOUT_S
            )
        except (OSError, asyncio.TimeoutError):
            return Liveness.UNKNOWN
        return self._interpret_pane_get(proc.returncode, stdout, stderr)

    def liveness_sync(self) -> Liveness:
        """Synchronous :meth:`liveness` for the daemon idle-watcher thread."""
        if self._closed:
            return Liveness.ENDPOINT_GONE
        try:
            proc = subprocess.run(
                self._pane_get_argv(),
                capture_output=True,
                check=False,
                timeout=self._CLI_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired):
            return Liveness.UNKNOWN
        return self._interpret_pane_get(proc.returncode, proc.stdout, proc.stderr)

    async def close(self) -> None:
        """Close this terminal's workspaces, then stop its dedicated server.

        Idempotent and never raises: a workspace already gone (orphan-reaped, or
        a crashed server) closes quietly. The reap sweep retires any leftover
        same-label workspaces of THIS session (scope: same session + same label —
        never another session, never ``default``). The server remains live until
        workspace cleanup completes so missing workspaces retain their precise
        liveness signal; it is then stopped to avoid accumulating idle servers.
        """
        if self._workspace_id is not None:
            with contextlib.suppress(RuntimeError):
                await self._run("workspace", "close", self._workspace_id)
        await self._reap_labeled_workspaces()
        with contextlib.suppress(RuntimeError):
            await self._run("server", "stop")
        self._closed = True

    async def _reap_labeled_workspaces(self, *, exclude: str | None = None) -> None:
        """Close every same-label workspace of this session (orphan reaping).

        :param exclude: A workspace id to leave alone (the freshly-created one).
        """
        try:
            listing = await self._run_json("workspace", "list")
        except RuntimeError:
            return
        for ws in self._workspaces_of(listing):
            wsid = ws.get("workspace_id")
            if ws.get("label") == self._label and wsid is not None and wsid != exclude:
                with contextlib.suppress(RuntimeError):
                    await self._run("workspace", "close", wsid)

    async def send_text(self, text: str) -> None:
        """Type literal *text* into the pane WITHOUT submitting it.

        Delivered as the positional ``<text>`` argument of ``pane send-text``
        (the spike-verified form, which landed a multi-line payload as a
        non-submitting draft with full Unicode fidelity). No trailing newline is
        implied, so a multi-line paste arrives intact for the caller to submit
        separately via :meth:`send_keys`.

        .. note::
            The whole paste rides in one argv element (no shell — the argv is
            handed straight to the OS spawn), so quoting/newlines are never an
            issue, but a *very* large paste is bounded by the OS command-line
            length limit. #11 delivered the text on stdin to sidestep that cap;
            the spike only verified the positional form, so we adopt it here and
            leave any stdin/chunking fallback for very large pastes to #13's
            real-herdr validation (``pane send-text`` reading stdin is
            unverified).
        """
        await self._run(*self._pane_argv("send-text", text))

    async def send_keys(self, keys: Sequence[str]) -> None:
        """Press named keys, translating each into herdr's key syntax.

        Keys are positional after the pane id (``pane send-keys <id> <key...>``).
        See :meth:`_translate_key`: chords become plus-notation, some named keys
        are renamed, and unsupported keys are skipped. A keystroke set that
        translates to nothing is a no-op.
        """
        wire = [translated for key in keys if (translated := self._translate_key(key)) is not None]
        if wire:
            await self._run(*self._pane_argv("send-keys", *wire))

    # --------------------------------------------------- delivery surface (sync)
    #
    # Synchronous siblings for the thread-based :class:`TerminalDelivery` callers.
    # These mirror the async pane commands through the sync runner (fake-tested
    # against ``_fake_herdr``; native Windows delivery lands in a later change).
    # herdr's ``pane send-text`` is already non-submitting, so paste and literal
    # typing are the same command here (unlike tmux, which needs a buffer paste).

    def send_text_sync(self, text: str) -> None:
        """Type literal *text* (sync sibling of :meth:`send_text`)."""
        self._run_output_sync(*self._pane_argv("send-text", text))

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        """Press named keys, translated (sync sibling of :meth:`send_keys`)."""
        wire = [translated for key in keys if (translated := self._translate_key(key)) is not None]
        if wire:
            self._run_output_sync(*self._pane_argv("send-keys", *wire))

    def paste_without_submit_sync(self, text: str) -> None:
        """Paste *text* without submitting — herdr ``send-text`` is non-submitting."""
        self.send_text_sync(text)

    def kill_session_sync(self) -> None:
        """Hard-stop this session by closing its workspace.

        Raises on a non-zero close so a wedged host surfaces (the delivery
        contract). A workspace already gone is nothing to stop, so it returns
        quietly.
        """
        if self._workspace_id is None:
            return
        self._run_output_sync("workspace", "close", self._workspace_id)

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Snapshot the pane via ``pane read``, normalizing CRLF → ``\\n``.

        ANSI passthrough is off by default (``--format text`` vs ``--format
        ansi`` — the spike-verified flag; ``--format ansi`` preserves 256-color
        SGR for the browser feed). See :meth:`_read_argv` for the source /
        min-line-count strategy; ``scrollback`` history is tailed locally to the
        requested size (:meth:`_tail_lines`).
        """
        raw = self._normalize_newlines(await self._run_output(*self._read_argv(ansi, scrollback)))
        return self._tail_lines(raw, scrollback)

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Synchronous :meth:`capture` for the threaded idle watcher."""
        raw = self._normalize_newlines(self._run_output_sync(*self._read_argv(ansi, scrollback)))
        return self._tail_lines(raw, scrollback)

    def _read_argv(self, ansi: bool, scrollback: int) -> list[str]:
        """Build the ``pane read`` argv for a snapshot (spike-verified surface).

        - ``--format text`` / ``--format ansi`` selects plain vs SGR-preserving
          output (never ``--ansi``; that was a #11 placeholder).
        - ``scrollback == 0`` reads ``--source visible`` — the current viewport,
          which herdr returns whole regardless of ``--lines`` — matching tmux's
          ``capture-pane`` default.
        - ``scrollback > 0`` reads ``--source recent-unwrapped`` (logical,
          non-wrapped lines) with ``--lines`` set to *at least*
          :data:`_SNAPSHOT_MIN_FETCH_LINES` (the min-line-count workaround:
          fetch large so a small-N read can never come back empty), then
          :meth:`capture` tails the result to ``scrollback`` lines locally.
        """
        extra = ["--format", "ansi" if ansi else "text"]
        if scrollback > 0:
            extra += ["--source", "recent-unwrapped"]
            extra += ["--lines", str(max(scrollback, self._SNAPSHOT_MIN_FETCH_LINES))]
        else:
            extra += ["--source", "visible"]
        return self._pane_argv("read", *extra)

    @staticmethod
    def _tail_lines(text: str, scrollback: int) -> str:
        """Keep only the last *scrollback* lines of *text* (local tail).

        The min-line-count workaround over-fetches (see :meth:`_read_argv`), so a
        ``scrollback > 0`` request is narrowed back to the caller's size here. A
        ``scrollback <= 0`` request (visible viewport) is returned unchanged — the
        viewport is already exactly what the caller asked for.
        """
        if scrollback <= 0:
            return text
        # Split off a single trailing newline so it is not counted as an empty
        # final line, then restore it, so a captured screen that ends in "\n"
        # tails to the same visible line count a caller expects.
        trailing = "\n" if text.endswith("\n") else ""
        body = text[: -len(trailing)] if trailing else text
        lines = body.split("\n")
        if len(lines) <= scrollback:
            return text
        return "\n".join(lines[-scrollback:]) + trailing

    # ------------------------------------------------------------- busy / ready

    async def _agent_status(self) -> str | None:
        """Return the pane's native ``agent_status`` string, or ``None``.

        Read from ``pane get`` (the spike found herdr's native agent detection
        populates a pane ``agent_status`` of
        ``idle``/``working``/``blocked``/``done``/``unknown``). ``None`` on any
        failure — a gone pane, an unspawnable CLI, unparseable output, or a
        missing field — so callers treat it as "no native signal" rather than
        crashing.

        Reconciled to real herdr's envelope in #13: the status is nested at
        ``.result.pane.agent_status`` (``{"id": "cli:pane:get", "result":
        {"pane": {"agent_status": ..}, "type": "pane_info"}}``), read via
        :meth:`_result_payload`.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._pane_get_argv(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._CLI_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError):
            return None
        if proc.returncode != 0:
            return None
        data = self._parse_envelope(stdout)
        if data is None:
            return None
        pane = self._result_payload(data).get("pane")
        status = pane.get("agent_status") if isinstance(pane, dict) else None
        return status if isinstance(status, str) else None

    async def busy_state(self) -> bool | None:
        """Report whether the inner agent is mid-turn (busy), corroborated.

        herdr's native ``agent_status`` is the primary signal, but the spike
        found it reads *idle* during long foreground tool calls, so it is
        corroborated with an output-diff of consecutive plain snapshots. The
        truth table (native status vs. whether the screen changed since the last
        call):

        ===================  ==================  ==============  =============
        native ``agent_status``  output changed?     no prior snap    verdict
        ===================  ==================  ==============  =============
        working / blocked    (ignored)           (ignored)       ``True``
        idle / done          yes                 —               ``True``
        idle / done          no                  —               ``False``
        idle / done          —                   yes             ``False``
        unknown / absent     yes                 —               ``True``
        unknown / absent     no                  —               ``False``
        unknown / absent     —                   yes             ``None``
        ===================  ==================  ==============  =============

        In words: a native *busy* status is authoritative. A native *idle*
        status is trusted only until the screen contradicts it (a changing
        screen under a native idle → the lying-idle case → busy). With no usable
        native signal we fall back to output-diff alone; and with neither a
        native signal nor a prior snapshot to diff against, there is no signal at
        all → ``None`` (unknown).

        A **gone pane** is handled gracefully rather than by raising: the
        corroborating :meth:`capture` raises ``RuntimeError`` once herdr has
        destroyed the pane, so it is guarded — a native *busy* status still wins,
        otherwise a gone pane yields ``None`` (no usable signal). This keeps the
        codex-path caller (which polls ``busy_state`` on a possibly-just-exited
        terminal) from seeing an exception instead of a verdict.

        :returns: ``True`` (busy), ``False`` (idle), or ``None`` (no signal).
        """
        native = await self._agent_status()
        try:
            current = await self.capture()
        except RuntimeError:
            # The pane is gone (capture raises once herdr destroys it). Trust a
            # native busy status if we somehow still have one; otherwise there is
            # no output to diff, so degrade to "no signal" instead of raising.
            self._last_activity_snapshot = None
            return True if native in self._NATIVE_BUSY_STATES else None
        prior = self._last_activity_snapshot
        self._last_activity_snapshot = current
        changed = prior is not None and current != prior

        if native in self._NATIVE_BUSY_STATES:
            return True
        if native in self._NATIVE_IDLE_STATES:
            return changed
        # No usable native signal: output-diff only, degrading to unknown when
        # there is not even a prior snapshot to diff against.
        if prior is None:
            return None
        return changed

    async def input_ready(self) -> bool | None:
        """Report whether the composer is ready to accept a new prompt.

        The "delivery dance" (paste then submit) needs the composer idle at its
        prompt, not mid-turn. Derived from the native ``agent_status`` alone
        (readiness is a composer-state question, not an output-activity one):
        ``idle``/``done`` → ready; ``working``/``blocked`` → not ready; no
        native signal → ``None`` (unknown). A gone endpoint reads as not-ready
        (``False``) rather than unknown, since input can never be delivered to
        it.

        :returns: ``True`` (ready), ``False`` (busy/gone), or ``None`` (unknown).
        """
        if await self.liveness() != Liveness.ALIVE:
            return False
        native = await self._agent_status()
        if native in self._NATIVE_IDLE_STATES:
            return True
        if native in self._NATIVE_BUSY_STATES:
            return False
        return None


# ---------------------------------------------------------------------------
# Backend registry + selection
# ---------------------------------------------------------------------------
#
# A name→backend-class registry (mirroring the ``SandboxBackend`` registry in
# ``sandbox.py``) plus the selection contract the firstmate abstraction proved
# out: an explicit per-terminal spec beats an env override beats user config
# beats the platform default, with unknown names and platform mismatches
# failing loudly rather than falling back silently.

# Registered terminal backends, keyed by :attr:`TerminalBackend.name`.
# Populated by :func:`register_terminal_backend`; ``TmuxBackend`` registers at
# import. A new backend (herdr, a later change) registers its class here and is
# then selectable by name with no change to the machinery above the seam.
_TERMINAL_BACKENDS: dict[str, type[TerminalBackend]] = {}


def register_terminal_backend(backend_cls: type[TerminalBackend]) -> None:
    """Register *backend_cls* under its :attr:`TerminalBackend.name`.

    :param backend_cls: A concrete :class:`TerminalBackend` subclass whose
        ``name`` / ``capabilities`` / ``platforms`` class attributes are set.
    """
    _TERMINAL_BACKENDS[backend_cls.name] = backend_cls


register_terminal_backend(TmuxBackend)
register_terminal_backend(HerdrBackend)
if IS_WINDOWS:
    from . import conpty_backend as _conpty_backend  # noqa: F401


# Platform → default backend name used when nothing is explicitly selected.
# POSIX defaults to tmux (the compatibility contract: an absent backend field
# means tmux, so every pre-existing persisted spec keeps working unchanged).
# Native Windows defaults to ConPTY, whose live handle remains runner-owned;
# advertised delivery reaches it through the authenticated control channel.
_PLATFORM_DEFAULT_BACKEND: dict[str, str] = {
    "posix": TmuxBackend.name,
    "windows": "conpty",
}


def _current_platform_tag() -> str:
    """Return this host's backend platform tag (``"posix"`` / ``"windows"``).

    Routed through the module-level :data:`IS_WINDOWS` so tests can simulate
    the other platform by monkeypatching it.
    """
    return "windows" if IS_WINDOWS else "posix"


def _env_terminal_backend() -> str | None:
    """Read the :data:`_TERMINAL_BACKEND_ENV_VAR` override, or ``None``.

    An unset or blank/whitespace-only value returns ``None`` so it does not
    shadow the config or platform-default tiers.
    """
    value = os.environ.get(_TERMINAL_BACKEND_ENV_VAR)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _read_terminal_backend_config() -> str | None:
    """Read ``terminal.backend`` from the global config, or ``None``.

    Best-effort, mirroring :func:`_read_terminal_transport_config`: any failure
    (missing/unreadable file, non-mapping YAML, absent table/key, non-string
    value) returns ``None`` so the caller falls through to the platform
    default. Never raises — reading the backend must not crash terminal
    construction.

    :returns: The configured backend name, or ``None`` when unset.
    """
    import yaml

    path = _global_config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict):
        return None
    table = raw.get(_TERMINAL_CONFIG_TABLE)
    if not isinstance(table, dict):
        return None
    value = table.get(_TERMINAL_BACKEND_CONFIG_KEY)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def resolve_terminal_backend_name(*, spec_backend: str | None = None) -> str:
    """Resolve the backend name for one terminal by selection precedence.

    Precedence (first non-empty wins):

    1. ``spec_backend`` — the per-terminal
       :attr:`~omnigent.inner.datamodel.TerminalEnvSpec.terminal_backend`, the
       most specific selection.
    2. The :data:`_TERMINAL_BACKEND_ENV_VAR` env override.
    3. ``terminal.backend`` in ``~/.omnigent/config.yaml`` (the user config).
    4. The platform default in :data:`_PLATFORM_DEFAULT_BACKEND` — tmux on
       POSIX; none on Windows (yet).

    An absent value at every tier resolves to the platform default, which is
    the compatibility contract that keeps pre-existing specs working as tmux.

    :param spec_backend: The per-terminal backend name, or ``None``.
    :returns: The resolved backend name (not yet validated against the
        registry or platform — see :func:`select_terminal_backend_class`).
    :raises RuntimeError: When no tier selects a backend and the current
        platform has no default (native Windows until herdr lands).
    """
    for candidate in (spec_backend, _env_terminal_backend(), _read_terminal_backend_config()):
        if candidate is not None and candidate.strip():
            return candidate.strip()
    platform_tag = _current_platform_tag()
    default = _PLATFORM_DEFAULT_BACKEND.get(platform_tag)
    if default is None:
        raise RuntimeError(
            f"No terminal multiplexer backend supports this platform "
            f"({platform_tag}); native harnesses require a platform-native "
            "backend. tmux is POSIX-only and no Windows-native backend "
            "is available. Run an SDK-based harness via "
            "`omnigent run <agent.yaml>` (e.g. the claude-sdk, cursor, "
            "copilot, or codex harness) or use the web UI."
        )
    return default


def select_terminal_backend_class(*, spec_backend: str | None = None) -> type[TerminalBackend]:
    """Select and validate the backend class for one terminal.

    Resolves the name via :func:`resolve_terminal_backend_name`, then checks it
    loudly at selection time — the single point the terminal factory calls:

    - an unknown name raises, listing the known backends;
    - a backend whose :attr:`~TerminalBackend.platforms` excludes the current
      platform raises. No silent fallback.

    Binary availability is checked separately by the caller via
    :meth:`TerminalBackend.ensure_available` (so the platform verdict and the
    install-hint error stay distinct).

    :param spec_backend: The per-terminal backend name, or ``None``.
    :returns: The selected, registered, platform-compatible backend class.
    :raises RuntimeError: On an unknown name, a platform mismatch, or no
        available backend for the platform (via the resolver).
    """
    name = resolve_terminal_backend_name(spec_backend=spec_backend)
    backend_cls = _TERMINAL_BACKENDS.get(name)
    if backend_cls is None:
        known = ", ".join(sorted(_TERMINAL_BACKENDS)) or "(none registered)"
        raise RuntimeError(f"Unknown terminal backend {name!r}. Known backends: {known}.")
    platform_tag = _current_platform_tag()
    if platform_tag not in backend_cls.platforms:
        supported = ", ".join(sorted(backend_cls.platforms)) or "(none)"
        raise RuntimeError(
            f"Terminal backend {name!r} does not support this platform "
            f"({platform_tag}). Supported platforms: {supported}."
        )
    return backend_cls


def _construct_terminal_backend(
    name: str,
    *,
    socket_path: str | Path,
    target: str,
    tmux_delivery_style: TmuxDeliveryStyle | None = None,
    paste_dir: str | Path | None = None,
) -> TerminalBackend:
    """Construct the registered backend *name* for a terminal instance.

    The construction seam for :meth:`TerminalInstance.__post_init__`. Each
    backend takes different constructor arguments, so this cannot call a uniform
    constructor. tmux keeps its own dedicated branch (its construction stays
    byte-identical); every other backend declares how to build itself via the
    :meth:`TerminalBackend.construct_for_instance` hook, which this dispatcher
    prefers when the backend overrides it. A backend that overrides neither path
    (the base hook returns ``None``) fails loudly rather than falling back to
    tmux — so herdr (#11) and the in-process ``FakeBackend`` become
    constructible by overriding the hook, with no edit to this function.

    :param name: A registered backend name, e.g. ``"tmux"``.
    :param socket_path: Private multiplexer socket path for this instance.
    :param target: Session/pane target name, e.g. ``"main"``.
    :param tmux_delivery_style: Per-bridge tmux delivery argv dialect, forwarded
        to :class:`TmuxBackend` only (inert for other backends, whose delivery
        stream is not tmux argv). ``None`` selects the claude-shaped default, so a
        :class:`TerminalInstance` — which never passes one — is byte-unchanged.
    :returns: A fresh backend instance bound to this terminal.
    :raises RuntimeError: When *name* is not registered.
    :raises NotImplementedError: When *name* is registered but provides no
        construction hook yet.
    """
    backend_cls = _TERMINAL_BACKENDS.get(name)
    if backend_cls is None:
        known = ", ".join(sorted(_TERMINAL_BACKENDS)) or "(none registered)"
        raise RuntimeError(f"Unknown terminal backend {name!r}. Known backends: {known}.")
    if backend_cls is TmuxBackend:
        return TmuxBackend(
            socket_path=socket_path,
            target=target,
            delivery_style=tmux_delivery_style,
            paste_dir=paste_dir,
        )
    # Non-tmux hooks take a ``Path``; normalize here (tmux keeps the value raw
    # so a bridge-advertised socket string reaches an identical ``-S`` argv).
    backend = backend_cls.construct_for_instance(socket_path=Path(socket_path), target=target)
    if backend is not None:
        return backend
    raise NotImplementedError(
        f"terminal backend {name!r} is registered but its constructor is not "
        "wired into _construct_terminal_backend yet."
    )


class TerminalDelivery:
    """Shared, synchronous prompt-delivery surface over a :class:`TerminalBackend`.

    The consolidated home of the "delivery dance" the seven TUI-typing native
    bridges each open-coded against a private tmux helper: clear the composer
    draft, paste a multi-line prompt WITHOUT submitting it, verify via a screen
    snapshot that the draft landed, then submit and verify it left the box —
    plus named-key send (Enter/Escape/Ctrl-C/…), a non-raising screen snapshot,
    a hard session kill, and the capability-gated native popup. Delivery bugs are
    now fixed once here instead of seven times.

    The surface is **synchronous** because its callers are the bridges, which
    deliver from a worker thread with no event loop (``asyncio.to_thread``);
    every method drives the backend's ``*_sync`` protocol. It holds no state of
    its own — a fresh instance binds to one terminal's advertised endpoint via
    :func:`build_prompt_delivery`. TUI-specific knowledge (what the composer's
    prompt glyph looks like, when a draft is "still in the box") stays with the
    caller and enters through the ``draft_present`` predicate, so this surface
    serves any TUI without embedding one vendor's screen grammar.
    """

    def __init__(self, backend: TerminalBackend) -> None:
        """:param backend: The backend bound to this terminal's endpoint."""
        self._backend = backend

    @property
    def backend(self) -> TerminalBackend:
        """The backend this surface drives (its capabilities, name, …)."""
        return self._backend

    def snapshot(self) -> str:
        """Return the pane's plain-text screen, or ``""`` when unavailable.

        Never raises: the delivery dance polls this to watch the composer, and a
        transient capture miss is "not ready yet", not a failure.
        """
        return self._backend.delivery_snapshot_sync()

    def is_alive(self) -> bool:
        """Return whether this session's host endpoint still exists.

        The delivery fast-fail: a bridge probes this before injecting so a web
        message into an exited TUI raises a clear "restart" error rather than
        being typed into a dead pane (or polling a dead pane for the full
        readiness timeout). Never raises — an unrunnable probe reads as not-alive.
        """
        return self._backend.delivery_liveness_sync()

    def send_keys(self, keys: Sequence[str]) -> None:
        """Press *keys* in order (Enter/Escape/Ctrl-C/… — the interrupt and
        submit primitive)."""
        self._backend.send_keys_sync(list(keys))

    def send_keys_repeated(self, key: str, count: int) -> None:
        """Press *key* *count* times (the composer-clear backspace burst).

        The primitive cursor's ``_clear_composer`` flood is built on: where a
        composer ignores readline kill keys and only ``Backspace`` deletes, a
        burst of *count* presses clears a line in one shot. The tmux backend emits
        a single native ``send-keys -N`` repeat; other backends fall back to
        *count* single presses.
        """
        self._backend.send_keys_repeated_sync(key, count)

    def send_keys_atomic(self, keys: Sequence[str]) -> None:
        """Press all *keys* in ONE multiplexer client command (atomic multi-key).

        Use when the source stream packed several named keys into a single
        injection — e.g. a permission dialog's ``Down Down Enter`` — so a
        concurrent client cannot interleave the sequence and mis-answer the
        dialog. The tmux backend emits one ``send-keys`` with every key; herdr's
        one ``pane send-keys`` is already atomic. Single-key callers should use
        :meth:`send_keys`.
        """
        self._backend.send_keys_atomic_sync(list(keys))

    def type_literal(self, text: str) -> None:
        """Type *text* literally into the composer without submitting (e.g. a
        slash command's characters)."""
        self._backend.send_text_sync(text)

    def paste_without_submit(self, text: str) -> None:
        """Paste multi-line *text* as one non-submitting draft into the composer."""
        self._backend.paste_without_submit_sync(text)

    def kill(self) -> None:
        """Hard-stop this session (the web UI "Stop session" affordance)."""
        self._backend.kill_session_sync()

    def launch_native_popup(
        self,
        *,
        config_file: Path,
        session_id: str,
        elicitation_id: str,
        message: str,
        policy_name: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        """Overlay a native approval popup on the pane where the backend can.

        Capability-gated: on a backend without
        :attr:`TerminalBackendCapabilities.native_popup` (herdr, the fake) this
        is a no-op and the web approval card remains the elicitation surface —
        the degradation the spec calls for. Arguments are forwarded to
        :meth:`TerminalBackend.native_popup_launch`.
        """
        if not self._backend.capabilities.native_popup:
            return
        self._backend.native_popup_launch(
            config_file=config_file,
            session_id=session_id,
            elicitation_id=elicitation_id,
            message=message,
            policy_name=policy_name,
            python_executable=python_executable,
        )

    def submit_and_verify(
        self,
        *,
        draft_present: Callable[[str], bool],
        error_message: str,
        poll_interval_s: float,
        commit_timeout_s: float,
        settle_s: float,
        verify_timeout_s: float,
        retry_interval_s: float,
    ) -> None:
        """Submit a pasted draft with commit-then-verify handshaking.

        The submit half of the delivery dance, reproduced from the bridges'
        private helper so its behavior is byte-for-byte unchanged:

        1. **Commit wait** — poll :meth:`snapshot` up to *commit_timeout_s* until
           *draft_present* sees the paste land in the composer. A TUI coalesces a
           rapid stdin burst into a paste, so a submit key that arrives mid-paste
           is folded into the draft as a newline instead of submitting; waiting
           for the visible draft makes the handoff deterministic. When the draft
           is never identifiable (e.g. whitespace-only content) the loop falls
           through and submits blind, matching the pre-consolidation behavior.
        2. **Settle** — a brief *settle_s* pause, then send ``Enter``.
        3. **Verify + retry** — if the draft was observed, poll up to
           *verify_timeout_s* that it left the box; re-send ``Enter`` no more
           often than *retry_interval_s* while it has not, since a swallowed
           submit must be retried but a slow-but-successful one must not be
           double-tapped. Raise *error_message* if the draft never clears.

        :param draft_present: Predicate over a snapshot: is the draft still in the
            composer? (TUI-specific; supplied by the caller.)
        :param error_message: RuntimeError text raised if the draft never
            submits (the caller owns the wording so it stays vendor-accurate).
        :param poll_interval_s: Seconds between polls.
        :param commit_timeout_s: Max seconds to wait for the draft to land.
        :param settle_s: Pause after the draft lands before the submit Enter.
        :param verify_timeout_s: Max seconds to confirm the draft left the box.
        :param retry_interval_s: Minimum spacing between retry Enters.
        :raises RuntimeError: With *error_message* if the draft never submits.
        """
        draft_seen = False
        deadline = time.monotonic() + commit_timeout_s
        while time.monotonic() < deadline:
            if draft_present(self.snapshot()):
                draft_seen = True
                break
            time.sleep(poll_interval_s)
        time.sleep(settle_s)
        self.send_keys(["Enter"])
        if not draft_seen:
            # The draft was never observed, so its absence proves nothing —
            # verification would trivially "pass". Submit blind as before.
            return
        deadline = time.monotonic() + verify_timeout_s
        last_enter = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(poll_interval_s)
            if not draft_present(self.snapshot()):
                return
            if time.monotonic() - last_enter >= retry_interval_s:
                self.send_keys(["Enter"])
                last_enter = time.monotonic()
        raise RuntimeError(error_message)

    def submit_once(
        self,
        *,
        draft_present: Callable[[str], bool] | None,
        poll_interval_s: float,
        commit_timeout_s: float,
        settle_s: float,
    ) -> None:
        """Submit a pasted draft with a single Enter and NO verify/retry.

        The submit half of the cursor/goose/kimi delivery dance, reproduced from
        their private helpers so the behavior is byte-for-byte unchanged — and
        deliberately distinct from :meth:`submit_and_verify`, which re-sends Enter
        until the draft leaves the box. These TUIs submit on ONE Enter (goose
        inserts a newline on Ctrl+J and submits on Enter, so a second Enter would
        submit twice), so this never re-sends and never raises:

        1. **Commit wait** — when *draft_present* is given, poll :meth:`snapshot`
           up to *commit_timeout_s* until it sees the paste land in the composer,
           so a submit key arriving mid-paste isn't folded into the draft as a
           newline. When *draft_present* is ``None`` (no usable needle — e.g.
           whitespace-only content), skip the wait and submit blind, matching the
           bridges' ``if needle:`` guard.
        2. **Settle** — a brief *settle_s* pause (always, even on the blind path).
        3. **Submit** — exactly one ``Enter``. No verification, no retry.

        :param draft_present: Predicate over a snapshot: has the paste landed in
            the composer? ``None`` skips the commit wait (blind submit).
        :param poll_interval_s: Seconds between commit-wait polls.
        :param commit_timeout_s: Max seconds to wait for the draft to land before
            falling through to the submit anyway.
        :param settle_s: Pause after the commit wait, before the submit Enter.
        """
        if draft_present is not None:
            deadline = time.monotonic() + commit_timeout_s
            while time.monotonic() < deadline:
                if draft_present(self.snapshot()):
                    break
                time.sleep(poll_interval_s)
        time.sleep(settle_s)
        self.send_keys(["Enter"])


class _RemoteConptyBackend(TerminalBackend):
    """Forward delivery primitives to the runner process that owns a ConPTY."""

    name = "conpty"
    capabilities = TerminalBackendCapabilities(native_popup=False)
    platforms = frozenset({"windows"})

    def __init__(self, *, control_url: str, control_token: str | None) -> None:
        self._control_url = control_url
        self._control_token = control_token

    def _call(self, operation: str, **payload: object) -> object:
        body = json.dumps({"operation": operation, **payload}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._control_token:
            headers["Authorization"] = f"Bearer {self._control_token}"
        request = urllib.request.Request(self._control_url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ConPTY control request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ConPTY control endpoint is unreachable: {exc.reason}") from exc
        return decoded.get("result")

    async def launch(self, _request: TerminalLaunchRequest) -> None:
        raise RuntimeError("A remote ConPTY proxy cannot launch terminals")

    async def liveness(self) -> Liveness:
        return await asyncio.to_thread(self.liveness_sync)

    def liveness_sync(self) -> Liveness:
        return Liveness(str(self._call("liveness")))

    async def close(self) -> None:
        await asyncio.to_thread(self.kill_session_sync)

    async def send_text(self, text: str) -> None:
        await asyncio.to_thread(self.send_text_sync, text)

    async def send_keys(self, keys: Sequence[str]) -> None:
        await asyncio.to_thread(self.send_keys_sync, keys)

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        return await asyncio.to_thread(self.capture_sync, ansi=ansi, scrollback=scrollback)

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        return str(self._call("snapshot", ansi=ansi, scrollback=scrollback))

    def send_text_sync(self, text: str) -> None:
        self._call("send_text", text=text)

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        self._call("send_keys", keys=list(keys))

    def paste_without_submit_sync(self, text: str) -> None:
        self._call("paste_without_submit", text=text)

    def kill_session_sync(self) -> None:
        self._call("kill")


def build_prompt_delivery(
    *,
    socket_path: str | Path,
    target: str,
    backend_name: str | None = None,
    tmux_delivery_style: TmuxDeliveryStyle | None = None,
    paste_dir: str | Path | None = None,
    control_url: str | None = None,
    control_token: str | None = None,
) -> TerminalDelivery:
    """Build a :class:`TerminalDelivery` bound to an advertised terminal endpoint.

    The bridges advertise their hosted terminal's private socket + pane target
    (the runner writes them after launch) and call this to obtain the shared
    delivery surface, instead of shelling out to a multiplexer themselves. The
    backend is constructed through the same seam a real
    :class:`TerminalInstance` uses (:func:`_construct_terminal_backend`), so the
    surface is backend-agnostic.

    :param socket_path: The terminal's private multiplexer socket path, as the
        ``str`` a bridge read from its advertisement or a :class:`~pathlib.Path`.
    :param target: The session/pane target the terminal was launched on, e.g.
        ``"main"`` or ``"claude:0.0"``.
    :param backend_name: Backend to construct. ``None`` selects tmux — the POSIX
        default that hosts every native harness today; a backend-aware
        advertisement can pass a name once non-tmux delivery is enabled.
    :param tmux_delivery_style: Per-bridge tmux delivery argv dialect
        (:class:`TmuxDeliveryStyle`). ``None`` selects the claude-shaped default,
        so the #8 command stream is byte-unchanged; a migrated native bridge
        whose stream differs (cursor/goose/kimi — own paste buffer, capture flag
        order, per-command timeout) passes its own. Inert for non-tmux backends.
    :returns: A fresh delivery surface bound to this endpoint.
    """
    if backend_name == "conpty":
        if control_url is None:
            raise RuntimeError("ConPTY prompt delivery requires a runner control URL")
        return TerminalDelivery(
            _RemoteConptyBackend(control_url=control_url, control_token=control_token)
        )
    backend = _construct_terminal_backend(
        backend_name or TmuxBackend.name,
        socket_path=socket_path,
        target=target,
        tmux_delivery_style=tmux_delivery_style,
        paste_dir=paste_dir,
    )
    return TerminalDelivery(backend)


@dataclass
class TerminalInstance:
    """
    One running tmux session for a terminal environment.

    :param name: Terminal name from the agent spec, e.g. ``"bash"``.
    :param session_key: Per-launch session key, e.g. ``"s1"``.
    :param socket_path: Private tmux socket path for this instance.
    :param private_dir: Private directory holding the tmux socket and
        any forked workspace state.
    :param os_env: Optional OS environment backing this terminal.
    :param command: Executable to run inside tmux, e.g. ``"bash"``.
    :param args: Command arguments.
    :param env: Extra environment variables for the terminal process.
    :param env_unset: Environment variables to strip from the
        terminal's environment before launching, e.g.
        ``["DATABRICKS_CONFIG_PROFILE"]``. Applied AFTER ``env``
        is merged, so a listed key is removed unconditionally —
        if the same key also appears in ``env``, the strip wins.
        Intentional: ``env_unset`` is a leak-prevention boundary,
        not a soft default.
    :param inherit_env: Whether to start from ``os.environ`` before applying
        ``env`` / ``env_unset``.
    :param sandbox_policy: Optional sandbox wrapper policy.
    :param conversation_link: Optional web UI link for the owning
        conversation, e.g. ``"/c/conv_abc123"``.
    :param scrollback: Tmux scrollback history limit.
    :param tmux_allow_passthrough: Whether pane applications may use
        tmux passthrough escapes to query/control the attached terminal.
    :param tmux_start_on_attach: Whether to delay command startup until
        the first tmux client attaches to the session.
    :param running: Whether the tmux server is currently expected to
        be alive.
    """

    name: str
    session_key: str
    socket_path: Path
    private_dir: Path
    os_env: OSEnvironment | None = None
    command: str = "bash"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_unset: list[str] = field(default_factory=list)
    inherit_env: bool = True
    sandbox_policy: SandboxPolicy | None = None
    conversation_link: str | None = None
    # Egress allow-list to enforce for this terminal. Populated
    # from the effective ``OSEnvSandboxSpec.egress_rules`` at
    # create-instance time. When non-empty AND the sandbox uses
    # a spawn-time backend (``linux_bwrap`` / ``darwin_seatbelt``),
    # :meth:`launch` starts a parent-side L7 MITM proxy and threads
    # ``HTTP_PROXY`` / ``HTTPS_PROXY`` / CA env vars through to the
    # tmux-spawned shell so its outbound HTTP(S) traffic is
    # allow-listed by the same engine that gates the helper. The
    # ``SandboxPolicy.egress_relay_port`` / ``egress_socket_path``
    # fields are populated from the proxy handle before encoding
    # the policy into the launcher script.
    egress_rules: list[str] | None = None
    egress_allow_private_destinations: bool = False
    scrollback: int = 10000
    tmux_allow_passthrough: bool = False
    tmux_start_on_attach: bool = False
    # Keep the private tmux server alive after the pane's inner process exits
    # (``remain-on-exit`` / ``exit-empty off``). Opt-in per terminal because it
    # changes the ``has-session``-means-alive contract: with it on, liveness is
    # decided by ``#{pane_dead}`` (see :meth:`is_alive`), not session existence.
    # Enabled for the claude-native agent terminal so a single inner-CLI exit no
    # longer reaps the server and cascades into ``no server running`` (#540).
    keep_alive_after_exit: bool = False
    # Preferred web-attach transport for this terminal (``"pty"`` /
    # ``"control"``), or ``None`` to defer to the global default. Read by the
    # attach routes via :func:`resolve_terminal_transport`; does not affect how
    # the tmux server itself is launched.
    terminal_transport: str | None = None
    # Multiplexer backend name for this instance (``"tmux"`` today). The
    # factory sets this from the selection precedence
    # (:func:`select_terminal_backend_class`); direct construction (test
    # paths) defaults to tmux, preserving today's behavior byte-for-byte.
    # ``__post_init__`` constructs the matching backend via
    # :func:`_construct_terminal_backend`.
    backend_name: str = TmuxBackend.name
    running: bool = False
    launch_cwd: str | None = None
    # Owned per-launch egress proxy. ``None`` when the sandbox
    # carries no ``egress_rules`` or the backend doesn't need a
    # spawn-time wrap (the ``none`` backend does nothing here). Cleaned
    # up in :meth:`close` so the asyncio thread and the bound
    # Unix socket don't outlive the terminal.
    _egress_handle: EgressProxyHandle | None = field(default=None, repr=False)
    _egress_tmpdir: Path | None = field(default=None, repr=False)
    _idle_task: asyncio.Task[None] | None = field(default=None, repr=False)
    # Threaded idle-watcher state. Mirrors :attr:`_idle_task` but for
    # callers that don't have a long-lived event loop (the Omnigent path:
    # ``SysTerminalLaunchTool`` runs ``asyncio.run`` per call, so an
    # asyncio task started inside it dies the moment ``launch``
    # returns). The thread polls tmux capture-pane synchronously
    # under ``_idle_stop_event``.
    _idle_thread: threading.Thread | None = field(default=None, repr=False)
    _idle_stop_event: threading.Event | None = field(default=None, repr=False)
    # Monotonic timestamp of the last client interaction observed on this
    # terminal's web attach (keystroke / focus / mouse / resize / connect /
    # disconnect — see :meth:`note_client_interaction`). The idle watcher
    # discounts pane changes that land within a short window of this stamp,
    # so a client attaching, detaching, focusing, clicking, or typing does
    # not read as agent activity. ``-inf`` until the first interaction.
    _last_client_interaction_at: float = field(default=float("-inf"), repr=False)
    _last_pane_snapshot: str | None = field(default=None, repr=False)
    # Multiplexer backend. Built in ``__post_init__`` from ``socket_path``
    # (tmux is the only backend today), so every existing construction path —
    # including tests that build ``TerminalInstance`` directly — gets one
    # without a signature change. All multiplexer touchpoints route through it:
    # lifecycle (launch, liveness, close, reaping), input (send text / keys),
    # capture (read + idle snapshots), the status-line link, and detach-on-exit.
    # The idle-watch loops themselves stay on this instance.
    _backend: TerminalBackend = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Construct this instance's multiplexer backend from ``backend_name``.

        Defaults to tmux (see :attr:`backend_name`), so every existing
        construction path — including tests that build ``TerminalInstance``
        directly — gets a tmux backend for its socket without any signature
        change. The factory overrides ``backend_name`` from the selection
        precedence before construction.
        """
        self._backend = _construct_terminal_backend(
            self.backend_name,
            socket_path=self.socket_path,
            target=self.tmux_target,
        )

    @property
    def tmux_target(self) -> str:
        """The tmux target for send-keys/capture-pane (always 'main')."""
        return "main"

    @property
    def backend_capabilities(self) -> TerminalBackendCapabilities:
        """The multiplexer backend's static capability declaration.

        Read by machinery above the seam to degrade a feature the backend cannot
        host to its documented fallback — e.g. the native cost popup checks
        :attr:`TerminalBackendCapabilities.native_popup` and, where ``False``
        (herdr on Windows), routes the ASK verdict to the web approval card
        instead of a pane overlay. Capability-driven, never backend-identity- or
        platform-driven.
        """
        return self._backend.capabilities

    @property
    def terminal_backend(self) -> TerminalBackend:
        """Return the backend for capability-driven attach machinery."""
        return self._backend

    @property
    def delivery_target(self) -> str:
        """Return the backend-native target for a persisted delivery handle."""
        return self._backend.advertised_target(fallback=self.tmux_target)

    def note_client_interaction(self) -> None:
        """Record that a web client just interacted with this terminal.

        Called from the WebSocket attach bridge on every client event —
        connect, disconnect, a forwarded keystroke/focus/mouse byte, or a
        resize message. The idle watcher reads
        :attr:`_last_client_interaction_at` and discounts pane changes
        within a short window of it, so client-driven repaints (attach /
        detach reflow, focus in/out, clicks, typing) don't register as
        agent activity.

        Thread-safety: this is written on the event loop (the attach
        bridge) and read on the watcher's daemon thread. It's a single
        ``float`` assignment, atomic under the GIL, so no lock is needed —
        a stale read is at worst a timestamp a few milliseconds old, which
        the window tolerates.

        :returns: None.
        """
        self._last_client_interaction_at = time.monotonic()

    def last_pane_text(self) -> str | None:
        """Return the last visible pane text captured for diagnostics.

        The value is updated opportunistically by reads and watcher polls.
        It is intentionally a snapshot, not a live tmux query, so callers can
        still retrieve useful context after tmux has already disappeared.
        """
        snapshot = self._last_pane_snapshot
        if snapshot is None:
            return None
        text = _strip_ansi(snapshot).strip()
        return text or None

    def _remember_pane_snapshot(self, snapshot: str) -> None:
        """Store a pane capture for later exit diagnostics."""
        self._last_pane_snapshot = snapshot

    def _tmux_base_cmd(self) -> list[str]:
        """
        Build the tmux argv prefix for this instance's private server.

        Managed terminal sessions must not inherit the user's
        ``~/.tmux.conf``. The terminal integration owns the server
        lifecycle and applies the supported options explicitly during
        launch, so user config would make identical agent specs behave
        differently across machines.

        :returns: Base argv for subprocess calls, e.g.
            ``["tmux", "-S", "/tmp/.../tmux.sock", "-f", "/dev/null"]``.
        """
        return ["tmux", "-S", str(self.socket_path), "-f", _TMUX_CONFIG_PATH]

    async def set_conversation_link(self, conversation_link: str | None) -> None:
        """
        Update the link shown in this terminal's status bar.

        Cosmetic: delegates to the backend's capability-gated
        :meth:`TerminalBackend.set_status_link`, which no-ops on backends
        without a status line. The in-memory :attr:`conversation_link` is
        always updated so a later launch seeds the link regardless of backend.

        :param conversation_link: Conversation URL to show, e.g.
            ``"/c/conv_abc123"``, or ``None`` to clear the status
            value.
        :returns: None.
        :raises RuntimeError: If a running backend with a status line rejects
            the update.
        """
        self.conversation_link = conversation_link
        if not self.running:
            return
        await self._backend.set_status_link(conversation_link)

    async def launch(self, *, cwd: Path | None = None) -> None:
        """Start the tmux session."""
        if self.running:
            return
        effective_cwd = str(cwd or self.private_dir)

        # Do NOT advertise the tmux control socket path to the
        # pane. The tmux server runs unsandboxed, so exposing its socket
        # let pane code run ``tmux -S <sock> run-shell '...'`` to execute
        # commands outside the sandbox. The host-side control plane
        # addresses the socket via ``self.socket_path`` directly and never
        # needs the env var; any inherited value is stripped below too.
        if self.inherit_env:
            env = os.environ.copy()
        else:
            env = {}
        env.pop("OMNIGENT_TMUX_SOCK", None)
        # Apply per-terminal env overrides (takes precedence over inherited env).
        env.update(self.env)
        # Strip vars the caller asked us not to leak into the terminal —
        # ambient values like ``DATABRICKS_CONFIG_PROFILE`` would otherwise
        # propagate to the terminal's children (including MCP servers),
        # whose own auth resolution then picks up the parent's profile
        # instead of the credentials they were explicitly configured with.
        # Applied AFTER ``env.update`` so the strip wins even if the
        # same key was set in ``self.env`` — ``env_unset`` is a
        # leak-prevention boundary, not a soft default.
        for key in self.env_unset:
            env.pop(key, None)
        # Strip the runner-auth secret: native agents run their shell in
        # this tmux pane, so the binding token must never reach it.
        # After ``env.update`` so ``self.env`` can't re-admit it.
        env = strip_runner_auth_secrets(env)
        # Force a UTF-8 locale into the pane env when the inherited env
        # carries no UTF-8 signal in the vars the native TUI CLIs actually
        # read (LC_ALL/LANG). Without it, CLIs that read LC_ALL/LANG directly
        # (opencode/pi/hermes) instead of calling setlocale fall back to an
        # ASCII/Latin-1 codeset and re-encode their UTF-8 output byte-by-byte,
        # rendering multibyte characters as mojibake in the pane (issue #2427).
        _apply_utf8_locale_default(env)

        # Build the command to run inside tmux. If a sandbox policy
        # is configured, wrap the command in the sandbox launcher so
        # the process tree runs under bwrap / seatbelt —
        # the launcher's ``run_launcher`` re-execs itself under the
        # spawn-time wrap for ``linux_bwrap`` / ``darwin_seatbelt``
        # before activating the in-process pieces (relay daemon,
        # seccomp filter).
        #
        # When the sandbox carries ``egress_rules``, we also start
        # a parent-side MITM proxy here and bake its socket path /
        # relay port / CA bundle into the policy + env BEFORE
        # encoding the launcher. The launcher (post-wrap) reads
        # them off the encoded policy and starts the in-namespace
        # relay daemon during ``activate_sandbox``; the shell
        # spawned beyond the launcher inherits HTTP_PROXY / CA
        # env vars so its outbound traffic is filtered.
        sandbox_for_launcher: SandboxPolicy | None = self.sandbox_policy
        if sandbox_for_launcher is not None and sandbox_for_launcher.active:
            if self.egress_rules:
                sandbox_for_launcher = self._bootstrap_egress_proxy(sandbox_for_launcher, env)
            cli_path = shutil.which(self.command) or self.command
            launcher_path = create_exec_launcher(cli_path, sandbox_for_launcher)
            inner_cmd = [launcher_path, *self.args]
        else:
            inner_cmd = [self.command, *self.args]
        # Hand the resolved command + environment to the multiplexer backend,
        # which owns session/pane creation and the tmux option translation.
        # The behavioral toggles (keep-alive-after-exit, start-on-attach,
        # passthrough, the cosmetic status link) travel as backend-neutral
        # request fields.
        await self._backend.launch(
            TerminalLaunchRequest(
                command=inner_cmd,
                cwd=effective_cwd,
                env=env,
                scrollback=self.scrollback,
                keep_alive_after_exit=self.keep_alive_after_exit,
                allow_passthrough=self.tmux_allow_passthrough,
                start_on_attach=self.tmux_start_on_attach,
                status_link=self.conversation_link,
            )
        )

        self.running = True
        self.launch_cwd = effective_cwd

    async def send(
        self,
        text: str | None = None,
        *,
        keys: str = "Enter",
    ) -> TerminalResult:
        """Send keystrokes to the terminal.

        Args:
            text: Literal text to type. Delivered verbatim and non-submitting
                via :meth:`TerminalBackend.send_text`, which chunks it as
                needed so special characters are never interpreted and large
                prompts arrive intact.
            keys: Named keys to press after the text, space-separated, in
                Omnigent's backend-neutral vocabulary. Defaults to ``"Enter"``.
                Set to ``""`` to type text without pressing any key after.
                Examples: ``"Enter"``, ``"Tab"``, ``"C-c"``, ``"Escape"``,
                ``"C-d"``, ``"Up"``.
        """
        if not self.running:
            return {"error": "Terminal is not running"}

        try:
            if text:
                await self._backend.send_text(text)

            if keys:
                if text:
                    await asyncio.sleep(0.05)
                await self._backend.send_keys(keys.split())
        except RuntimeError:
            self.running = False
            return {
                "error": (
                    f"Terminal {self.name}:{self.session_key} is no longer "
                    "running (tmux server exited)"
                )
            }

        return {"status": "sent"}

    async def read(self, scrollback: int = 0) -> TerminalResult:
        """Capture the terminal screen."""
        if not self.running:
            return {"error": "Terminal is not running"}

        try:
            result = await self._backend.capture(ansi=False, scrollback=scrollback)
        except RuntimeError:
            self.running = False
            return {
                "error": (
                    f"Terminal {self.name}:{self.session_key} is no longer "
                    "running (tmux server exited)"
                )
            }

        self._remember_pane_snapshot(result)
        return {
            "terminal": f"{self.name}:{self.session_key}",
            "screen": _strip_ansi(result),
            "scrollback_lines": scrollback,
        }

    def _bootstrap_egress_proxy(
        self,
        sandbox: SandboxPolicy,
        env: dict[str, str],
    ) -> SandboxPolicy:
        """Start the parent-side L7 egress proxy for this terminal.

        Wires the proxy lifecycle into the terminal so close()
        tears it down. Mutates ``env`` to inject ``HTTP_PROXY`` /
        ``HTTPS_PROXY`` / CA env vars and returns an updated
        :class:`SandboxPolicy` whose ``egress_relay_port`` /
        ``egress_socket_path`` are populated for the launcher.
        The caller must use the returned policy for
        ``create_exec_launcher`` (and ideally ``wrap_launcher_argv``
        too); the old policy lacks the relay handshake info.

        Idempotent against repeated calls: every call creates a
        new proxy and replaces ``self._egress_handle`` /
        ``self._egress_tmpdir``. ``close()`` is the only sanctioned
        teardown.

        :param sandbox: Active sandbox policy (caller has verified
            ``sandbox.active``).
        :param env: Mutable env dict that will be passed to the
            tmux subprocess.
        :returns: Updated policy with relay info baked in and the
            scratch tmpdir added to ``write_roots`` so bwrap
            bind-mounts it inside the namespace.
        """
        assert self.egress_rules, "caller checked self.egress_rules"
        self._egress_tmpdir = create_private_tmpdir()
        # Add the scratch tmpdir to write_roots BEFORE encoding the
        # policy into the launcher. Without this, bwrap won't bind
        # the tmpdir into the namespace and the CA bundle /
        # egress socket the launcher needs at activate time would
        # be invisible.
        sandbox = with_additional_write_roots(sandbox, [self._egress_tmpdir])
        self._egress_handle = start_egress_proxy(
            rules=self.egress_rules,
            tmpdir=self._egress_tmpdir,
            allow_private_destinations=self.egress_allow_private_destinations,
            # Terminal path uses ``require_auth=False``: tmux closes
            # inherited FDs before exec, so we have no out-of-band
            # channel for a Proxy-Authorization token. Embedding the
            # token in HTTP_PROXY (the alternative) would leak it via
            # ``ps -E`` on every shell child anyway. The relay's
            # other defenses (random ephemeral port, default-deny on
            # private destinations, allow-list per :attr:`egress_rules`)
            # still apply; see the controller's docstring for the
            # full trade-off discussion.
            require_auth=False,
        )
        apply_egress_env(
            env,
            relay_port=self._egress_handle.relay_port,
            ca_bundle_path=self._egress_handle.ca_bundle_path,
            auth_token=None,
        )
        return replace(
            sandbox,
            egress_relay_port=self._egress_handle.relay_port,
            egress_socket_path=str(self._egress_handle.socket_path),
        )

    async def close(self) -> None:
        """Kill the tmux session and clean up."""
        # Cancel both idle-watcher variants first so they don't race
        # the socket teardown. Order doesn't matter — they're
        # independent.
        await self._stop_idle_watcher()
        self._stop_idle_watcher_thread()

        if self.running:
            await self._backend.close()
            self.running = False

        if self.os_env is not None:
            self.os_env.close()

        # Stop the egress proxy + clean up its scratch tmpdir.
        # Order: stop first so the proxy isn't listening on a
        # socket inside a soon-to-be-deleted dir, then remove the
        # tmpdir.
        if self._egress_handle is not None:
            try:
                self._egress_handle.stop()
            except Exception:
                logger.exception(
                    "egress proxy stop failed for terminal %s:%s",
                    self.name,
                    self.session_key,
                )
            self._egress_handle = None
        if self._egress_tmpdir is not None:
            cleanup_private_tmpdir(self._egress_tmpdir)
            self._egress_tmpdir = None

        # Clean up the private dir (contains socket + fork).
        if self.private_dir.exists():
            shutil.rmtree(self.private_dir, ignore_errors=True)

    def start_idle_watcher(
        self,
        on_idle: Callable[[], None | Awaitable[None]],
        *,
        on_exit: Callable[[], None | Awaitable[None]] | None = None,
    ) -> None:
        """Start a background task that fires ``on_idle`` each time the pane
        becomes quiet (no change for ``_IDLE_THRESHOLD_SECONDS``).

        Edge-triggered: the callback fires once per idle transition.  It will
        fire again only after new output changes the pane and then stops again.
        The watcher is cancelled by ``close()``.
        """
        if not self.running:
            raise RuntimeError("Cannot start idle watcher before launch")
        if self._idle_task is not None and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_watch_loop(on_idle, on_exit=on_exit))

    async def _stop_idle_watcher(self) -> None:
        task = self._idle_task
        if task is None:
            return
        self._idle_task = None
        if task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _idle_watch_loop(
        self,
        on_idle: Callable[[], None | Awaitable[None]],
        *,
        on_exit: Callable[[], None | Awaitable[None]] | None = None,
    ) -> None:
        """
        Asyncio polling loop driving an :class:`_IdleDetector`.

        :param on_idle: Edge-triggered callback. May be sync or
            async; awaited if it returns a coroutine. Exceptions
            inside the callback log + stop the watcher.
        """
        detector = _IdleDetector()

        async def _fire(callback: Callable[[], None | Awaitable[None]], kind: str) -> bool:
            """Invoke a callback. Returns False if it raised and the watcher should exit."""
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "%s-notification callback failed for terminal %s:%s",
                    kind,
                    self.name,
                    self.session_key,
                )
                return False
            return True

        while self.running:
            await asyncio.sleep(_IDLE_POLL_INTERVAL_SECONDS)
            if not self.running:
                return
            try:
                snapshot = await self._backend.capture(ansi=True)
            except RuntimeError:
                # tmux server likely gone.
                self.running = False
                if on_exit is not None:
                    await _fire(on_exit, "exit")
                return

            self._remember_pane_snapshot(snapshot)
            if await self._pane_is_dead_async():
                # remain-on-exit kept the server alive after the inner CLI
                # exited; report the exit rather than treating the frozen pane
                # as an idle agent. Detach all clients so attached tmux attach
                # subprocesses (CLI direct attach, server-side bridge PTY) exit
                # naturally instead of hanging on the dead pane. Only relevant
                # when keep_alive_after_exit is set (remain-on-exit was enabled).
                if self.keep_alive_after_exit:
                    with contextlib.suppress(Exception):
                        await self._backend.detach_display_clients()
                self.running = False
                if on_exit is not None:
                    await _fire(on_exit, "exit")
                return
            if detector.tick(snapshot) and not await _fire(on_idle, "idle"):
                return

    def start_idle_watcher_thread(
        self,
        on_idle: Callable[[], None] | None = None,
        *,
        on_activity: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        idle_threshold_s: float | None = None,
        poll_interval_s: float | None = None,
        replace: bool = False,
    ) -> None:
        """
        Start a daemon thread driving idle/activity edges from the pane.

        Thread-based sibling of :meth:`start_idle_watcher` for callers
        without a long-lived event loop. The Omnigent ``sys_terminal_launch``
        path runs ``SysTerminalLaunchTool.invoke`` on a worker thread
        and drives :meth:`launch` via ``asyncio.run`` per call — that
        loop exits the moment ``launch`` returns, so an asyncio task
        started inside it dies. A daemon thread polling tmux via
        ``subprocess.run`` survives across launch / send / read tool
        calls and stops on :meth:`close` (or when the host process
        exits, since it's a daemon).

        Edge-triggered: ``on_idle`` fires once per idle transition (re-
        arms only after new output mutates the pane); ``on_activity``
        fires on every poll tick where the pane content changed — so at
        most once per *poll_interval_s*, which for the fast claude-native
        watcher (200ms) is up to ~5/sec while a pane redraws continuously.
        ``on_exit`` fires once when the tmux session disappears unexpectedly.
        Any further rate-limiting of activity (e.g. the runner's
        one-pulse-per-second ``session.terminal.activity`` throttle) is
        the caller's responsibility, not this watcher's. At least one
        callback should be provided; passing several is fine.

        :param on_idle: Optional sync callback invoked once per idle
            edge, or ``None`` to skip idle detection. Must not block the
            polling thread for long — invoked synchronously between
            snapshots.
        :param on_activity: Optional sync callback invoked on each tick
            the pane changed (the runner-determined "PTY had output"
            signal). Same non-blocking contract as *on_idle*.
        :param on_exit: Optional sync callback invoked when the watcher
            observes that tmux has disappeared. Same non-blocking contract
            as *on_idle*.
        :param idle_threshold_s: Per-watcher diff-track idle threshold in
            seconds passed to :class:`_IdleDetector`, e.g. ``1.0`` for the
            claude-native status watcher. ``None`` uses the module
            default :data:`_IDLE_THRESHOLD_SECONDS`.
        :param poll_interval_s: Per-watcher poll interval in seconds, e.g.
            ``0.2`` for the claude-native status watcher (snappier
            running/idle transitions). ``None`` uses the module default
            :data:`_IDLE_POLL_INTERVAL_SECONDS`.
        :param replace: When ``True``, replace any existing threaded watcher
            so callbacks can be rebound after terminal ownership transfer.
        :raises RuntimeError: When the instance is not currently
            running (caller forgot to ``await launch`` first).
        """
        if not self.running:
            raise RuntimeError("Cannot start idle watcher before launch")
        if on_idle is None and on_activity is None and on_exit is None:
            raise ValueError(
                "start_idle_watcher_thread requires at least one of "
                "on_idle / on_activity / on_exit — a watcher with none would poll "
                "tmux forever with no effect."
            )
        if self._idle_thread is not None and self._idle_thread.is_alive():
            if not replace:
                return
            self._stop_idle_watcher_thread()
        stop_event = threading.Event()
        self._idle_stop_event = stop_event
        self._idle_thread = threading.Thread(
            target=self._idle_watch_loop_threaded,
            args=(stop_event,),
            kwargs={
                "on_idle": on_idle,
                "on_activity": on_activity,
                "on_exit": on_exit,
                "idle_threshold_s": idle_threshold_s,
                "poll_interval_s": poll_interval_s,
            },
            name=f"terminal-idle-{self.name}-{self.session_key}",
            daemon=True,
        )
        self._idle_thread.start()

    def _idle_watch_loop_threaded(
        self,
        stop_event: threading.Event,
        *,
        on_idle: Callable[[], None] | None = None,
        on_activity: Callable[[], None] | None = None,
        on_exit: Callable[[], None] | None = None,
        idle_threshold_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        """
        Sync polling loop driving an :class:`_IdleDetector`.

        Runs on the daemon thread spawned by
        :meth:`start_idle_watcher_thread`. Stops cleanly when
        ``stop_event`` is set or when ``self.running`` flips to
        ``False`` (close path), and exits silently if ``tmux
        capture-pane`` fails (server likely gone).

        :param stop_event: Event the close path sets to signal
            shutdown. Doubles as the poll-interval sleep via
            :meth:`Event.wait` so the join window is bounded by
            one poll interval, not the full sleep.
        :param on_idle: Optional idle-edge callback (see
            :meth:`start_idle_watcher_thread`); skipped when ``None``.
        :param on_activity: Optional pane-changed callback; fired each
            tick the pane content changed. Skipped when ``None``.
        :param on_exit: Optional callback fired when tmux disappears.
            Skipped when ``None``.
        :param idle_threshold_s: Per-watcher diff-track idle threshold in
            seconds forwarded to :class:`_IdleDetector`, e.g. ``1.0``.
            ``None`` uses the module default.
        :param poll_interval_s: Seconds between polls, e.g. ``0.2`` for the
            claude-native status watcher. ``None`` uses the module default
            :data:`_IDLE_POLL_INTERVAL_SECONDS`.
        """
        detector = _IdleDetector(idle_threshold_s=idle_threshold_s)
        interval = poll_interval_s if poll_interval_s is not None else _IDLE_POLL_INTERVAL_SECONDS
        while self.running:
            # ``Event.wait`` doubles as the poll-interval sleep, so
            # ``stop_event.set()`` from :meth:`close` returns within
            # one tick instead of waiting out the full interval.
            if stop_event.wait(interval):
                return
            if not self.running:
                return
            snapshot = self._capture_pane_for_idle_or_none()
            if snapshot is None:
                self.running = False
                if on_exit is not None:
                    self._fire_watch_callback(on_exit, "exit")
                return
            self._remember_pane_snapshot(snapshot)
            if self._pane_is_dead():
                # The inner CLI exited but remain-on-exit kept the server, so
                # capture-pane still succeeds (the snapshot above is the final
                # frame, now remembered for diagnostics). Report the exit
                # deterministically instead of mistaking the frozen pane for an
                # idle agent and leaving the session hung. Detach all clients
                # so attached tmux attach subprocesses exit naturally. Only
                # relevant when keep_alive_after_exit is set.
                if self.keep_alive_after_exit:
                    with contextlib.suppress(Exception):
                        self._backend.detach_display_clients_sync()
                self.running = False
                if on_exit is not None:
                    self._fire_watch_callback(on_exit, "exit")
                return
            # A pane change that lands within the recent-interaction window
            # is a client-driven repaint (attach/detach reflow, focus,
            # mouse, keystroke — stamped via note_client_interaction), not
            # agent output, so the detector discounts it.
            suppress = (
                time.monotonic() - self._last_client_interaction_at
            ) < _CLIENT_INTERACTION_WINDOW_SECONDS
            idle_fired = detector.tick(snapshot, suppress_activity=suppress)
            # Activity edge first: a tick can both change the pane and
            # (much later) cross the idle threshold, but never both in
            # the same tick — a change resets the idle timer.
            if (
                on_activity is not None
                and detector.changed_this_tick
                and not self._fire_watch_callback(on_activity, "activity")
            ):
                return
            if (
                idle_fired
                and on_idle is not None
                and not self._fire_watch_callback(on_idle, "idle")
            ):
                return

    def _capture_pane_for_idle_or_none(self) -> str | None:
        """
        Capture the pane for an idle tick, or signal "host gone".

        :returns: The ANSI pane snapshot from the backend, or ``None`` when the
            capture raised — the threaded loop reads ``None`` as "stop
            watching, the host is no longer there".
        """
        try:
            return self._backend.capture_sync(ansi=True)
        except RuntimeError:
            return None

    def _pane_is_dead(self) -> bool:
        """
        Report whether the pane's process exited while tmux kept the pane.

        With ``remain-on-exit on`` (see
        :func:`_tmux_session_persistence_commands`) the private server survives
        the inner CLI's exit, so a *dead pane* — not a vanished server — is how
        a normal or early exit now presents. The threaded idle watcher uses this
        to report the exit deterministically once ``capture-pane`` still
        succeeds against the surviving server.

        :returns: ``True`` when the backend reports the inner process exited
            with the endpoint kept (:attr:`Liveness.INNER_EXITED`). ``False``
            when the pane is live, or when the probe fails / the server is
            already gone — the caller's capture step handles the
            vanished-server path.
        """
        return self._backend.liveness_sync() is Liveness.INNER_EXITED

    def _fire_watch_callback(self, callback: Callable[[], None], kind: str) -> bool:
        """
        Invoke a watcher edge callback, swallow + log on failure.

        :param callback: The user-supplied edge callback (idle or
            activity).
        :param kind: Label for logging, e.g. ``"idle"`` or
            ``"activity"``.
        :returns: ``True`` when the callback returned cleanly so
            the watcher continues; ``False`` when the callback
            raised (logged) so the watcher exits per the
            threaded-loop contract.
        """
        try:
            callback()
        except Exception:
            logger.exception(
                "%s-notification callback failed for terminal %s:%s",
                kind,
                self.name,
                self.session_key,
            )
            return False
        return True

    def _stop_idle_watcher_thread(self) -> None:
        """
        Signal the threaded watcher to stop and join with a timeout.

        Symmetrical to :meth:`_stop_idle_watcher` for the asyncio
        variant. Bounded by :data:`_IDLE_WATCHER_JOIN_TIMEOUT_S` so
        a wedged ``subprocess.run`` (rare — the only one in the loop
        body) doesn't block the close path indefinitely. After the
        timeout the thread keeps running, but it's a daemon — it
        will exit when the process does, and the next iteration's
        ``self.running`` check will short-circuit it anyway.
        """
        thread = self._idle_thread
        stop_event = self._idle_stop_event
        if thread is None:
            return
        self._idle_thread = None
        self._idle_stop_event = None
        if stop_event is not None:
            stop_event.set()
        if thread.is_alive():
            thread.join(timeout=_IDLE_WATCHER_JOIN_TIMEOUT_S)

    async def is_alive(self) -> bool:
        """
        Check if the terminal's inner process is still running.

        Probes the pane's ``#{pane_dead}`` flag rather than mere session
        existence: with ``remain-on-exit on`` (see
        :func:`_tmux_session_persistence_commands`) the session and server
        deliberately outlive the inner CLI's exit, so a live session no longer
        implies a live process. The terminal is alive only when the session
        exists AND its pane process has not exited.

        When the session is gone (probe exits non-zero), the pane is dead, or
        the probe cannot start, this marks ``self.running`` false. That side
        effect is intentional: subsequent pollers use the in-memory flag as a
        fast path instead of re-forking tmux after the process has exited.

        :returns: ``True`` when the session exists and its pane process is
            still running; otherwise ``False``.
        """
        if not self.running:
            return False
        # The backend's verdict distinguishes endpoint-gone from inner-exited
        # from a failed probe; any non-``ALIVE`` verdict means not-alive and
        # flips ``running`` off so later pollers short-circuit without
        # re-forking the multiplexer.
        if await self._backend.liveness() is Liveness.ALIVE:
            return True
        self.running = False
        return False

    async def _pane_is_dead_async(self) -> bool:
        """
        Async sibling of :meth:`_pane_is_dead` for the asyncio idle watcher.

        :returns: ``True`` when the backend reports
            :attr:`Liveness.INNER_EXITED`; ``False`` when the pane is live or
            the probe fails (server gone, which the caller's capture step
            handles).
        """
        return await self._backend.liveness() is Liveness.INNER_EXITED

    async def _tmux(self, *args: str) -> None:
        """Run a tmux command against this instance's server."""
        proc = await asyncio.create_subprocess_exec(
            *self._tmux_base_cmd(),
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"tmux command failed: {' '.join(args)}: {stderr.decode().strip()}")

    async def _tmux_output(self, *args: str) -> str:
        """Run a tmux command and return stdout."""
        proc = await asyncio.create_subprocess_exec(
            *self._tmux_base_cmd(),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"tmux command failed: {' '.join(args)}: {stderr.decode().strip()}")
        return stdout.decode()

    def _tmux_output_sync(self, *args: str) -> str:
        """
        Synchronous sibling of :meth:`_tmux_output`.

        Used by :meth:`_idle_watch_loop_threaded` because that
        watcher runs on a daemon thread without an event loop.
        Same error semantics as the async version: non-zero exit
        codes raise :class:`RuntimeError` carrying the stderr.

        :param args: Args to pass after ``tmux -S <socket>``,
            e.g. ``("capture-pane", "-t", "main", "-p", "-e")``.
        :returns: The captured stdout, decoded as UTF-8.
        :raises RuntimeError: When the tmux subprocess exits
            non-zero (typically because the server has gone away).
        """
        proc = subprocess.run([*self._tmux_base_cmd(), *args], capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"tmux command failed: {' '.join(args)}: {proc.stderr.decode().strip()}"
            )
        return proc.stdout.decode()


def _shell_quote(s: str) -> str:
    """Quote a string for shell use."""
    if not s:
        return "''"
    # Simple quoting for common cases.
    if re.match(r"^[a-zA-Z0-9_./:@=-]+$", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


@dataclass(frozen=True)
class TerminalCreateResult:
    """
    Result of :func:`create_terminal_instance`.

    :param instance: The freshly-constructed :class:`TerminalInstance`.
        Not yet launched — the caller is responsible for calling
        :meth:`TerminalInstance.launch` with the ``cwd`` below.
    :param cwd: The resolved working directory the caller should pass to
        :meth:`TerminalInstance.launch`. This is either the forked copy
        of the source tree (when ``spec.os_env.fork`` is true) or the
        original cwd resolved to an absolute path.
    """

    instance: TerminalInstance
    cwd: Path


def create_terminal_instance(
    name: str,
    session_key: str,
    spec: TerminalEnvSpec,
    *,
    parent_os_env_spec: OSEnvSpec | None = None,
    cwd_override: str | None = None,
    sandbox_override: str | None = None,
    conversation_link: str | None = None,
) -> TerminalCreateResult:
    """Create a terminal instance from a spec.

    Creates a private directory for the instance, optionally forks the
    filesystem, and prepares the tmux socket path.

    If the terminal spec has no ``os_env``, the parent's ``os_env`` is
    inherited (same cwd, same sandbox, no fork).  Terminals always have
    an ``os_env`` so their filesystems can be mounted.

    :param name: Logical terminal name from the agent spec (e.g. ``"bash"``).
    :param session_key: Per-session identifier used to scope tmux
        sockets and private directories, e.g. ``"s1"``.
    :param spec: The :class:`TerminalEnvSpec` describing the command,
        args, env, scrollback, and optional os_env for this terminal.
    :param parent_os_env_spec: The parent session's os_env spec, used
        when the terminal spec itself has no ``os_env`` and should
        inherit from the parent.
    :param cwd_override: Optional override for the terminal's starting
        working directory. When provided, takes precedence over the
        spec's cwd.
    :param sandbox_override: Optional override for the sandbox type,
        one of ``"none"`` or ``"linux_bwrap"``.
    :param conversation_link: Optional web UI link for the owning
        conversation, e.g. ``"/c/conv_abc123"``.
    :returns: A :class:`TerminalCreateResult` carrying the new instance
        and the resolved cwd to pass to ``launch()``.
    """
    # Select the multiplexer backend at the single construction point, by the
    # documented precedence (per-terminal spec → env → user config → platform
    # default). This is the one place platform support is decided: on native
    # Windows there is no backend yet, so selection raises a clear availability
    # error (a ``RuntimeError``, as the old hard-raise was) rather than an
    # incidental failure deeper in construction. ``ensure_available`` then
    # gates the chosen backend's binary with an install hint.
    backend_cls = select_terminal_backend_class(spec_backend=spec.terminal_backend)
    backend_cls.ensure_available()

    # Create the instance's private directory.
    private_dir = Path(tempfile.mkdtemp(prefix=_TERMINAL_DIR_PREFIX))
    socket_path = private_dir / "tmux.sock"
    # Record the owning process so a later startup can reap this tmux
    # server if we die without graceful shutdown (SIGKILL, harness
    # teardown) — see ``reap_orphaned_terminals``.
    (private_dir / _OWNER_PID_FILENAME).write_text(str(os.getpid()), encoding="utf-8")

    # Resolve os_env spec.  If none specified, inherit from parent.
    effective_os_env_spec = build_terminal_os_env_spec(
        spec,
        parent_os_env_spec=parent_os_env_spec,
        cwd_override=cwd_override,
        sandbox_override=sandbox_override,
    )

    os_env: OSEnvironment | None = None
    cwd: Path

    if effective_os_env_spec.fork:
        # Copy the directory tree for fork isolation.
        src_cwd = Path(effective_os_env_spec.cwd or os.getcwd()).resolve()
        fork_root = private_dir / "root"
        _copy_tree(src_cwd, fork_root)
        cwd = fork_root

        # Create an os_env pointing at the fork for mount support.
        # Use ``replace`` so any future OSEnvSpec field (e.g.
        # ``start_in_scratch``) is preserved without a code change here.
        forked_spec = replace(
            effective_os_env_spec,
            cwd=str(fork_root),
            fork=False,  # already forked
        )
        os_env = create_os_environment(forked_spec)
    else:
        cwd = Path(effective_os_env_spec.cwd or os.getcwd()).resolve()
        os_env = create_os_environment(effective_os_env_spec)

    # Resolve sandbox policy for the terminal process.
    sandbox: SandboxPolicy | None = None
    egress_rules: list[str] | None = None
    egress_allow_private: bool = False
    if effective_os_env_spec.sandbox is not None:
        sandbox_spec = effective_os_env_spec.sandbox
        if sandbox_spec.type != "none":
            sandbox = resolve_sandbox(effective_os_env_spec, cwd)
            if sandbox.active:
                # Add the private dir to write roots so a forked working
                # tree (``private_dir/root``) and the instance dir stay
                # writable inside the pane.
                sandbox = with_additional_write_roots(sandbox, [private_dir])
                # The tmux control socket lives inside that
                # now-writable ``private_dir``. Deny the sandboxed pane
                # from reaching it so it cannot ``tmux -S <sock> run-shell``
                # against the unsandboxed server. bwrap overlays /dev/null
                # onto the socket path; seatbelt emits a network-outbound
                # unix-socket deny (its default allow_network=true would
                # otherwise permit the connect).
                sandbox = with_denied_unix_sockets(sandbox, [socket_path])
        # Plumb the egress allow-list from the OSEnvSandboxSpec
        # onto the instance so :meth:`launch` can start a
        # parent-side MITM proxy. SandboxPolicy itself only carries
        # the *resolved* relay handshake fields, not the rule list
        # — see the policy docstring for the rationale (rules live
        # on the spec; resolved state on the policy).
        if sandbox_spec.egress_rules:
            egress_rules = list(sandbox_spec.egress_rules)
            egress_allow_private = bool(sandbox_spec.egress_allow_private_destinations)

    command = spec.command or "bash"
    if IS_WINDOWS and Path(command).name.lower() in ("bash", "bash.exe"):
        command = getattr(os_env, "shell_path", command)

    instance = TerminalInstance(
        name=name,
        session_key=session_key,
        socket_path=socket_path,
        private_dir=private_dir,
        os_env=os_env,
        command=command,
        args=list(spec.args),
        env=dict(spec.env),
        env_unset=list(spec.env_unset),
        inherit_env=spec.inherit_env,
        sandbox_policy=sandbox,
        conversation_link=conversation_link,
        egress_rules=egress_rules,
        egress_allow_private_destinations=egress_allow_private,
        scrollback=spec.scrollback,
        tmux_allow_passthrough=spec.tmux_allow_passthrough,
        tmux_start_on_attach=spec.tmux_start_on_attach,
        keep_alive_after_exit=spec.keep_alive_after_exit,
        terminal_transport=spec.terminal_transport,
        backend_name=backend_cls.name,
    )

    return TerminalCreateResult(instance=instance, cwd=cwd)
