"""A deterministic, scripted fake ``herdr`` CLI for the herdr backend tests.

:class:`~omnigent.inner.terminal.HerdrBackend` drives herdr purely as a
subprocess (one fresh process per call), so — exactly like
:mod:`tests.inner._fake_tmux` for tmux — this fake is a standalone, stdlib-only
script that models only the herdr CLI vocabulary the backend actually emits. It
must import nothing from ``omnigent`` so it can be launched by a bare
interpreter (the backend points :envvar:`OMNIGENT_HERDR_BIN` at
``[python, this_file]``; see the test module).

Server-per-named-session, on disk
---------------------------------

herdr runs a server per named session; the backend addresses each with an
explicit ``--session <name>``. This fake keeps each session's whole state in a
JSON sidecar named ``<session>.json`` under :envvar:`OMNIGENT_HERDR_STATE_DIR`.
Every ``herdr`` invocation is a fresh process, so the sidecar is the only thing
that persists between them — mirroring how a real per-session herdr server
persists across CLI commands. A session's first ``workspace create`` initializes
the sidecar (modeling herdr auto-starting the per-session server on first use).

Modeled vocabulary (nothing else)
----------------------------------

- ``api schema`` → the real binary's **text** report with a ``protocol: <N>``
  line (``N`` from :envvar:`OMNIGENT_HERDR_PROTOCOL`, so a test can simulate an
  old protocol). Models the real ``herdr api schema`` default output, which is
  human-readable text (NOT JSON) — the #12 protocol probe regex-extracts the
  number. Session-independent, but still invoked with an explicit ``--session``.
- ``workspace list`` / ``workspace create --label`` / ``workspace close
  --workspace`` — labels are NOT unique (a create never dedupes), so the backend
  can leave and later adopt same-label husks.
- ``tab create --workspace --cwd --cols --rows --command -- <argv...>`` — creates
  the tab and its single pane, pinning geometry.
- ``pane get <id>`` (liveness: alive / pane_not_found / workspace_not_found,
  plus ``agent_status``), ``pane read <id> --source S --format F [--lines N]``
  (screen snapshot: ``visible`` returns the whole viewport, ``recent``/
  ``recent-unwrapped`` tail ``--lines``; a small ``--lines`` models the historic
  empty-read quirk; ``--format ansi`` prepends an SGR marker; emitted with CRLF
  so the backend's normalization is exercised), ``pane send-text <id> <text>``
  (positional, non-submitting), ``pane send-keys <id> <key...>`` (positional
  named/plus-notation keys). The pane id is a **positional** argument (the
  spike-verified form), not a ``--pane`` flag.

Observability conventions (shared with :mod:`tests.inner._fake_tmux`)
---------------------------------------------------------------------

- An inner command equal to :data:`EXIT_SENTINEL` models a process that exits the
  instant it launches; herdr has no remain-on-exit, so its pane is immediately
  destroyed regardless of keep-alive → ``pane get`` reports ``pane_not_found``.
- A named ``Enter`` key appends :data:`SUBMIT_SENTINEL` to the screen, so a test
  can prove a literal ``send-text`` paste never submits.
- Every invocation's argv is appended to :envvar:`OMNIGENT_HERDR_LOG` (one JSON
  array per line) so a test can assert that EVERY herdr call carried an explicit
  ``--session`` (the never-ambient-targeting enforcement).

Later tickets can extend this fake (richer screen markers, more verbs) without
rewriting the model here.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Env vars the fake reads (set by the test that installs it).
# ---------------------------------------------------------------------------

#: Directory holding the per-session ``<session>.json`` state sidecars.
STATE_DIR_ENV_VAR = "OMNIGENT_HERDR_STATE_DIR"
#: File the fake appends every invocation's argv to (JSON-array lines).
LOG_ENV_VAR = "OMNIGENT_HERDR_LOG"
#: Wire protocol the fake's ``api schema`` report announces (default ``"16"``).
PROTOCOL_ENV_VAR = "OMNIGENT_HERDR_PROTOCOL"
#: Version string the fake's ``api schema`` report announces.
VERSION_ENV_VAR = "OMNIGENT_HERDR_VERSION"
#: Optional override for the ``agent_status`` every ``pane get`` reports, so a
#: test can flip the native busy/idle signal (incl. the lying-idle case where
#: native reads ``idle`` while the screen keeps changing). Unset → the pane's
#: own stored status (default ``"idle"``).
AGENT_STATUS_ENV_VAR = "OMNIGENT_HERDR_AGENT_STATUS"

DEFAULT_PROTOCOL = "16"
DEFAULT_VERSION = "0.7.4-preview"

#: ``pane read --format ansi`` prepends this SGR marker so a test can assert the
#: backend preserves ANSI escapes (herdr's ``--format ansi`` keeps 256-color
#: SGR). A lone LF-free escape so CRLF handling and tailing stay unaffected.
ANSI_MARKER = "\x1b[38;5;11m"

#: ``pane read --source recent|recent-unwrapped --lines N`` with ``N`` below this
#: returns an EMPTY capture, modeling the historic small-N empty-read quirk (NOT
#: reproduced on real 0.7.4, but the backend's fetch-large-tail-locally
#: workaround is regression-tested against it). ``visible`` ignores ``--lines``.
SMALL_N_EMPTY_THRESHOLD = 100

# An inner command whose sole argv token is this sentinel models a process that
# exits immediately on launch (herdr destroys its pane at once — no remain-on-
# exit). Mirrors the ``_fake_tmux`` convention.
EXIT_SENTINEL = "__omnigent_fake_herdr_exit__"

# Appended to the pane screen when a named ``Enter`` key is sent, so a test can
# prove a literal ``send-text`` paste never submits.
SUBMIT_SENTINEL = "[fake-herdr:submit]"


def key_marker(key: str) -> str:
    """Return the observable pane token the fake appends for a named *key*.

    A named-key send is otherwise invisible in a capture; the fake echoes each
    key (already translated to herdr's syntax by the backend, e.g. ``ctrl+c``)
    as ``<KEY>`` so tests can assert delivery.

    :param key: A herdr-syntax key token, e.g. ``"ctrl+c"`` or ``"Enter"``.
    :returns: The token appended to the pane screen, e.g. ``"<ctrl+c>"``.
    """
    return f"<{key}>"


# ---------------------------------------------------------------------------
# State sidecar helpers (pure; usable from the test process too).
# ---------------------------------------------------------------------------


def state_file(state_dir: str | Path, session: str) -> Path:
    """Return the JSON sidecar path holding *session*'s state."""
    return Path(state_dir) / f"{session}.json"


def _load(state_dir: str, session: str) -> dict[str, object] | None:
    """Load *session*'s state, or ``None`` when its server never started."""
    try:
        return json.loads(state_file(state_dir, session).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save(state_dir: str, session: str, state: dict[str, object]) -> None:
    """Persist *session*'s state to its sidecar (server started if absent)."""
    path = state_file(state_dir, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _blank_state(session: str) -> dict[str, object]:
    """Return an empty per-session server state (no workspaces yet)."""
    return {"session": session, "counter": 0, "workspaces": {}, "tabs": {}, "panes": {}}


def seed_workspace(state_dir: str | Path, session: str, label: str) -> str:
    """Pre-create a labeled workspace in *session*'s state (a husk).

    Used by the husk adopt/replace test to stage a restart leftover before the
    backend launches. Initializes the session server if absent.

    :returns: The new workspace id.
    """
    state = _load(str(state_dir), session) or _blank_state(session)
    wsid = _next_id(state, "ws")
    workspaces = state["workspaces"]
    assert isinstance(workspaces, dict)
    workspaces[wsid] = {"id": wsid, "label": label}
    _save(str(state_dir), session, state)
    return wsid


def read_log(log_path: str | Path) -> list[list[str]]:
    """Return every logged invocation's argv (``sys.argv[1:]``), in call order."""
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Argv parsing helpers.
# ---------------------------------------------------------------------------


def _next_id(state: dict[str, object], prefix: str) -> str:
    """Allocate a fresh ``<prefix><n>`` id, bumping the state counter."""
    counter = int(state.get("counter", 0)) + 1  # type: ignore[arg-type]
    state["counter"] = counter
    return f"{prefix}{counter}"


def _flag(tokens: list[str], name: str) -> str | None:
    """Return the value following ``name`` in *tokens*, or ``None``."""
    for i, token in enumerate(tokens):
        if token == name and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _command_argv(tokens: list[str]) -> list[str]:
    """Return the inner argv after a trailing ``--command --`` marker."""
    if "--command" not in tokens:
        return []
    rest = tokens[tokens.index("--command") + 1 :]
    if rest and rest[0] == "--":
        rest = rest[1:]
    return rest


def _log(argv: list[str]) -> None:
    """Append this invocation's argv to the log file, if one is configured."""
    log_path = os.environ.get(LOG_ENV_VAR)
    if not log_path:
        return
    with contextlib.suppress(OSError):
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(argv) + "\n")


# ---------------------------------------------------------------------------
# Subcommand handlers.
# ---------------------------------------------------------------------------


def _handle_api_schema() -> int:
    """Emit the ``api schema`` report (session-independent protocol gate).

    Models the **real** ``herdr api schema`` output, whose default form is
    human-readable text with a ``protocol: <N>`` line (verified against the real
    binary — it is NOT JSON by default; ``api schema --json`` prints the full
    schema). The #12 protocol gate regex-extracts the number from this text.
    """
    protocol = os.environ.get(PROTOCOL_ENV_VAR, DEFAULT_PROTOCOL)
    sys.stdout.write(
        "Herdr API schema\n"
        f"protocol: {protocol}\n"
        "schema_version: 1\n"
        "schemas: error_response, event, request, subscription_event, success_response\n"
        "\n"
        "Use `herdr api schema --json` to print the full schema.\n"
    )
    return 0


def _handle_workspace(state_dir: str, session: str, rest: list[str]) -> int:
    """Dispatch ``workspace list|create|close``."""
    action = rest[1] if len(rest) > 1 else ""
    if action == "list":
        state = _load(state_dir, session)
        workspaces = list((state or {}).get("workspaces", {}).values()) if state else []
        sys.stdout.write(json.dumps({"workspaces": workspaces}))
        return 0
    if action == "create":
        label = _flag(rest, "--label") or ""
        state = _load(state_dir, session) or _blank_state(session)
        wsid = _next_id(state, "ws")
        workspaces = state["workspaces"]
        assert isinstance(workspaces, dict)
        # Labels are intentionally NOT unique: a create never dedupes, so a
        # same-label husk survives here for the backend to adopt/replace.
        workspaces[wsid] = {"id": wsid, "label": label}
        _save(state_dir, session, state)
        sys.stdout.write(json.dumps({"workspace": {"id": wsid, "label": label}}))
        return 0
    if action == "close":
        wsid = _flag(rest, "--workspace") or ""
        state = _load(state_dir, session)
        if state is not None:
            workspaces = state["workspaces"]
            tabs = state["tabs"]
            panes = state["panes"]
            assert (
                isinstance(workspaces, dict) and isinstance(tabs, dict) and isinstance(panes, dict)
            )
            workspaces.pop(wsid, None)
            for tid in [t for t, tab in tabs.items() if tab.get("workspace") == wsid]:
                tabs.pop(tid, None)
            for pid in [p for p, pane in panes.items() if pane.get("workspace") == wsid]:
                panes.pop(pid, None)
            _save(state_dir, session, state)
        # Idempotent: closing an already-gone workspace succeeds quietly.
        return 0
    sys.stderr.write(f"unknown workspace action: {action}\n")
    return 2


def _handle_tab_create(state_dir: str, session: str, rest: list[str]) -> int:
    """Create a tab and its single pane under a workspace, pinning geometry."""
    state = _load(state_dir, session)
    if state is None:
        sys.stderr.write("workspace_not_found\n")
        return 1
    wsid = _flag(rest, "--workspace") or ""
    workspaces = state["workspaces"]
    assert isinstance(workspaces, dict)
    if wsid not in workspaces:
        sys.stderr.write("workspace_not_found\n")
        return 1
    command = _command_argv(rest)
    inner_exits = bool(command) and command[0] == EXIT_SENTINEL
    tid = _next_id(state, "tab")
    pid = _next_id(state, "pane")
    tabs = state["tabs"]
    panes = state["panes"]
    assert isinstance(tabs, dict) and isinstance(panes, dict)
    tabs[tid] = {"id": tid, "workspace": wsid, "pane": pid}
    panes[pid] = {
        "id": pid,
        "tab": tid,
        "workspace": wsid,
        # herdr has no remain-on-exit: a process that exits at once leaves a
        # destroyed pane (alive=False → pane_not_found), final screen lost.
        "alive": not inner_exits,
        "agent_status": "idle",
        "screen": "",
        "cols": _flag(rest, "--cols"),
        "rows": _flag(rest, "--rows"),
        "cwd": _flag(rest, "--cwd"),
        "command": command,
    }
    _save(state_dir, session, state)
    sys.stdout.write(json.dumps({"tab": {"id": tid}, "pane": {"id": pid}}))
    return 0


def _live_pane(state_dir: str, session: str, pid: str) -> dict[str, object] | None:
    """Return the live pane record for *pid*, or ``None`` if gone/destroyed."""
    state = _load(state_dir, session)
    if state is None:
        return None
    panes = state["panes"]
    assert isinstance(panes, dict)
    pane = panes.get(pid)
    if not isinstance(pane, dict) or not pane.get("alive"):
        return None
    return pane


def _handle_pane(state_dir: str, session: str, rest: list[str]) -> int:
    """Dispatch ``pane get|read|send-text|send-keys`` (positional pane id).

    ``rest`` is ``["pane", <action>, <pane-id>, ...]`` — the pane id is the
    first positional after the action (the spike-verified form), not a
    ``--pane`` flag.
    """
    action = rest[1] if len(rest) > 1 else ""
    pid = rest[2] if len(rest) > 2 else ""

    if action == "get":
        state = _load(state_dir, session)
        if state is None:
            sys.stdout.write(json.dumps({"result": "workspace_not_found"}))
            return 0
        pane = _live_pane(state_dir, session, pid)
        if pane is None:
            sys.stdout.write(json.dumps({"result": "pane_not_found"}))
            return 0
        # An env override lets a test flip the native busy/idle signal (incl. the
        # lying-idle case) without mutating the sidecar per pane.
        status = os.environ.get(AGENT_STATUS_ENV_VAR) or pane.get("agent_status")
        sys.stdout.write(json.dumps({"result": "alive", "agent_status": status}))
        return 0

    if action == "read":
        pane = _live_pane(state_dir, session, pid)
        if pane is None:
            sys.stderr.write("pane_not_found\n")
            return 1
        out = _rendered_read(pane, rest)
        # Write bytes (not text): on Windows a text-mode ``sys.stdout`` would
        # translate ``\n`` → ``\r\n`` on write and corrupt the CRLF we emit on
        # purpose. The buffer bypasses that so the backend receives exactly the
        # CRLF stream a real herdr capture carries.
        sys.stdout.buffer.write(out.replace("\n", "\r\n").encode("utf-8"))
        return 0

    if action in ("send-text", "send-keys"):
        # Look the pane up INSIDE the state we will save, so the screen mutation
        # persists (a fresh ``_load`` would be a different, discarded object).
        state = _load(state_dir, session)
        pane = None
        if state is not None:
            panes = state["panes"]
            assert isinstance(panes, dict)
            candidate = panes.get(pid)
            if isinstance(candidate, dict) and candidate.get("alive"):
                pane = candidate
        if pane is None:
            sys.stderr.write("pane_not_found\n")
            return 1
        screen = str(pane.get("screen", ""))
        if action == "send-text":
            # Non-submitting literal paste, delivered as the positional <text>.
            screen += rest[3] if len(rest) > 3 else ""
        else:
            for key in rest[3:]:  # positional (already backend-translated) keys
                screen += key_marker(key)
                if key == "Enter":
                    screen += SUBMIT_SENTINEL
        pane["screen"] = screen
        _save(state_dir, session, state)
        return 0

    sys.stderr.write(f"unknown pane action: {action}\n")
    return 2


def _rendered_read(pane: dict[str, object], rest: list[str]) -> str:
    """Render a ``pane read`` snapshot honoring ``--source``/``--format``/``--lines``.

    - ``--source visible`` (default) returns the whole stored viewport, ignoring
      ``--lines`` (herdr's ``visible`` behavior).
    - ``--source recent``/``recent-unwrapped`` tail the last ``--lines`` logical
      lines; a ``--lines`` below :data:`SMALL_N_EMPTY_THRESHOLD` returns EMPTY,
      modeling the historic small-N empty-read quirk the backend defends against.
    - ``--format ansi`` prepends :data:`ANSI_MARKER` (an SGR escape) so a test
      can assert the backend passes ANSI through.

    The stored screen is normalized to LF first so the CRLF the caller emits is
    purely a read-path (herdr-emission) artifact.
    """
    screen = str(pane.get("screen", "")).replace("\r\n", "\n").replace("\r", "\n")
    source = _flag(rest, "--source") or "visible"
    lines_flag = _flag(rest, "--lines")

    if source in ("recent", "recent-unwrapped"):
        n = int(lines_flag) if lines_flag is not None and lines_flag.isdigit() else None
        if n is not None and n < SMALL_N_EMPTY_THRESHOLD:
            body = ""  # small-N empty-read quirk
        elif n is not None:
            body = "\n".join(screen.split("\n")[-n:])
        else:
            body = screen
    else:  # "visible": whole viewport, --lines ignored
        body = screen

    if (_flag(rest, "--format") or "text") == "ansi" and body:
        body = ANSI_MARKER + body
    return body


def main(argv: list[str]) -> int:
    """Run one fake ``herdr`` invocation.

    :param argv: The process argv without the program name (``sys.argv[1:]``).
    :returns: A process exit code (``0`` success; non-zero mirrors a herdr
        failure such as a missing pane).
    """
    _log(argv)

    # Every backend invocation leads with an explicit ``--session <name>``.
    session = _flag(argv, "--session")
    rest = [
        t
        for i, t in enumerate(argv)
        if t != "--session" and (i == 0 or argv[i - 1] != "--session")
    ]
    if not rest:
        return 0
    command = rest[0]

    # ``api schema`` is session-independent (the protocol gate); handle it before
    # the session guard, exactly as the real static schema dump needs no session.
    if command == "api" and len(rest) > 1 and rest[1] == "schema":
        return _handle_api_schema()

    if session is None:
        # The backend never emits a bare subcommand; guard anyway.
        sys.stderr.write("missing --session\n")
        return 2
    state_dir = os.environ.get(STATE_DIR_ENV_VAR)
    if not state_dir:
        sys.stderr.write(f"{STATE_DIR_ENV_VAR} not set\n")
        return 2

    if command == "workspace":
        return _handle_workspace(state_dir, session, rest)
    if command == "tab":
        return _handle_tab_create(state_dir, session, rest)
    if command == "pane":
        return _handle_pane(state_dir, session, rest)

    sys.stderr.write(f"unknown herdr command: {command}\n")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main(sys.argv[1:]))
