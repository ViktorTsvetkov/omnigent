"""A deterministic, scripted fake ``tmux`` binary for the conformance suite.

The parametrized backend conformance suite
(:mod:`tests.inner.terminal_backend_conformance`) drives the *real*
:class:`~omnigent.inner.terminal.TmuxBackend` — the one that shells out to
``tmux`` — against this fake so the suite runs in CI with no real tmux
installed and no timing nondeterminism.

The fake is a standalone stdlib-only script (it must import nothing from
``omnigent`` so it can be launched by a bare shell shim; see
:func:`install_fake_tmux`). Each managed terminal addresses its own private
server via ``tmux -S <socket>``, so the fake keeps that server's whole state in
a JSON sidecar next to the socket path. Every ``tmux`` invocation is a fresh
process, so the sidecar is the only thing that persists between them — exactly
mirroring how a real tmux server persists across client commands.

Only the tmux vocabulary :class:`TmuxBackend` actually emits is modeled:
``set-option`` / ``set-window-option`` / ``set-hook`` / ``unbind-key`` (options,
mostly ignored), ``new-session`` (create the server), ``list-panes -F
#{pane_dead}`` (liveness), ``capture-pane`` (snapshot), ``send-keys`` (literal
paste and named keys), ``load-buffer`` / ``paste-buffer`` (the delivery surface's
bracketed non-submitting paste), ``kill-session`` (the delivery hard-stop),
``kill-server`` (teardown), and ``detach-client``.

Two conventions make otherwise-invisible behavior observable to the suite:

- A named ``Enter`` key appends :data:`SUBMIT_SENTINEL` to the screen, so the
  suite can assert that a literal paste (``send-keys -l``) never submits.
- An inner command equal to :data:`EXIT_SENTINEL` models a process that exits
  the instant it launches, so the suite can exercise the keep-alive-after-exit
  and liveness-verdict paths without real timing.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import sys
from pathlib import Path

# An inner command whose sole argv token is this sentinel models a process that
# exits immediately on launch. The conformance suite passes ``[EXIT_SENTINEL]``
# as the launch command to exercise inner-exit / keep-alive verdicts.
EXIT_SENTINEL = "__omnigent_fake_tmux_exit__"

# Appended to the pane screen when a named ``Enter`` key is sent. The suite
# asserts this is ABSENT after a multi-line ``send_text`` (proving the paste did
# not submit) and PRESENT after an explicit ``send_keys(["Enter"])``.
SUBMIT_SENTINEL = "[fake-tmux:submit]"

# The final screen the fake preserves after an inner process that exited while
# ``remain-on-exit`` (keep-alive) was on. Lets the suite prove the final frame
# stays capturable past inner exit.
FINAL_SCREEN_MARKER = "[fake-tmux:inner-exited]\n"


def key_marker(key: str) -> str:
    """Return the observable pane token the fake appends for a named *key*.

    A named key send is otherwise invisible in a capture; the fake echoes each
    key as ``<KEY>`` so the conformance suite can assert delivery.

    :param key: A named key in Omnigent's neutral vocabulary, e.g. ``"Escape"``.
    :returns: The token appended to the pane screen, e.g. ``"<Escape>"``.
    """
    return f"<{key}>"


def _state_path(socket_path: str) -> Path:
    """Return the JSON sidecar path holding this server's state."""
    return Path(socket_path + ".fakestate.json")


def _load(socket_path: str) -> dict[str, object] | None:
    """Load this server's state, or ``None`` when the server is gone."""
    path = _state_path(socket_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save(socket_path: str, state: dict[str, object]) -> None:
    """Persist this server's state to its JSON sidecar."""
    _state_path(socket_path).write_text(json.dumps(state), encoding="utf-8")


def _remove(socket_path: str) -> None:
    """Remove this server's state sidecar (idempotent server teardown)."""
    with contextlib.suppress(OSError):
        _state_path(socket_path).unlink()


def _split_semicolons(tokens: list[str]) -> list[list[str]]:
    """Split a flattened tmux command sequence on standalone ``;`` tokens.

    :meth:`TmuxBackend.launch` packs many commands into one invocation using a
    literal ``;`` argv separator; every other invocation is a single command.

    :param tokens: The argv after the ``-S``/``-f`` global options.
    :returns: One token list per command, dropping empty groups.
    """
    groups: list[list[str]] = [[]]
    for token in tokens:
        if token == ";":
            groups.append([])
        else:
            groups[-1].append(token)
    return [group for group in groups if group]


def _positionals(cmd: list[str]) -> list[str]:
    """Return the positional args of a ``send-keys`` command.

    Drops the command word, the ``-l`` literal flag, the ``-t <target>`` pair,
    and the ``-N <count>`` repeat pair, leaving the literal text (for ``-l``) or
    the key names.
    """
    positionals: list[str] = []
    i = 1  # skip the command word itself
    while i < len(cmd):
        token = cmd[i]
        if token == "-l":
            i += 1
        elif token in ("-t", "-N"):
            i += 2  # skip the target / repeat-count value too
        else:
            positionals.append(token)
            i += 1
    return positionals


def _repeat_count(cmd: list[str]) -> int:
    """Return the ``send-keys -N <count>`` repeat count, or ``1`` when absent.

    tmux's ``-N`` repeats the whole key sequence *count* times (cursor's
    composer-clear flood sends ``send-keys -N 200 BSpace``). A malformed or
    missing count reads as a single press.
    """
    if "-N" in cmd:
        idx = cmd.index("-N")
        if idx + 1 < len(cmd):
            try:
                return int(cmd[idx + 1])
            except ValueError:
                return 1
    return 1


def _handle_new_session(cmd: list[str], socket_path: str, *, remain_on_exit: bool) -> int:
    """Create this server for a ``new-session`` command.

    The inner command is the last argv token (``... -c <cwd> <inner_str>``). An
    inner equal to :data:`EXIT_SENTINEL` models an immediate exit: with
    keep-alive on the pane is kept but marked dead (``INNER_EXITED``); without
    it the session is destroyed and the server vanishes (``ENDPOINT_GONE``).
    """
    inner = cmd[-1].strip() if cmd else ""
    inner_exits = inner == EXIT_SENTINEL
    if inner_exits and not remain_on_exit:
        # exit-empty tears the session (and the private server) down at once.
        _remove(socket_path)
        return 0
    _save(
        socket_path,
        {
            "screen": FINAL_SCREEN_MARKER if inner_exits else "",
            "pane_dead": inner_exits and remain_on_exit,
        },
    )
    return 0


def _handle_list_panes(socket_path: str) -> int:
    """Emit ``#{pane_dead}`` (``0``/``1``), or fail when the server is gone."""
    state = _load(socket_path)
    if state is None:
        sys.stderr.write("no server running\n")
        return 1
    sys.stdout.write("1\n" if state.get("pane_dead") else "0\n")
    return 0


def _handle_capture(socket_path: str) -> int:
    """Print the pane screen, or fail when the server is gone."""
    state = _load(socket_path)
    if state is None:
        sys.stderr.write("no server running\n")
        return 1
    sys.stdout.write(str(state.get("screen", "")))
    return 0


def _handle_send_keys(cmd: list[str], socket_path: str) -> int:
    """Append literal text or named-key tokens to the pane screen.

    ``send-keys -l`` appends the literal text verbatim (non-submitting).
    ``send-keys <name>`` appends the observable ``<name>`` token, plus
    :data:`SUBMIT_SENTINEL` for ``Enter`` so a submit is detectable.
    """
    state = _load(socket_path)
    if state is None:
        sys.stderr.write("no server running\n")
        return 1
    screen = str(state.get("screen", ""))
    positionals = _positionals(cmd)
    if "-l" in cmd:
        screen += "".join(positionals)
    else:
        # ``-N <count>`` repeats the key sequence (cursor's Backspace flood).
        for _ in range(_repeat_count(cmd)):
            for key in positionals:
                screen += key_marker(key)
                if key == "Enter":
                    screen += SUBMIT_SENTINEL
    state["screen"] = screen
    _save(socket_path, state)
    return 0


def _handle_has_session(socket_path: str) -> int:
    """Exit ``0`` when this server/session exists, ``1`` when it is gone.

    Mirrors real ``tmux has-session``: the delivery liveness probe
    (:meth:`TmuxBackend.delivery_liveness_sync`) keys off the exit code.
    """
    return 0 if _load(socket_path) is not None else 1


def _handle_kill_server(socket_path: str) -> int:
    """Tear the server down; non-zero when it was already gone (tmux parity)."""
    existed = _load(socket_path) is not None
    _remove(socket_path)
    return 0 if existed else 1


def _handle_kill_session(socket_path: str) -> int:
    """Kill this session (the delivery hard-stop).

    Modeled like :func:`_handle_kill_server` for the fake's single-session
    private server: the session's pane is destroyed, so a following liveness
    probe finds no server and reports ``ENDPOINT_GONE``.
    """
    existed = _load(socket_path) is not None
    _remove(socket_path)
    return 0 if existed else 1


def _handle_load_buffer(cmd: list[str], socket_path: str) -> int:
    """Load a named paste buffer from a file (``load-buffer -b <name> <path>``).

    The delivery bracketed-paste path streams the payload through a file so it
    isn't capped by tmux's client->server command limit. The file's bytes are
    the CR-encoded paste payload; the fake decodes and stashes them under the
    buffer name for a following ``paste-buffer`` to deposit into the pane.
    """
    state = _load(socket_path)
    if state is None:
        sys.stderr.write("no server running\n")
        return 1
    name = cmd[cmd.index("-b") + 1] if "-b" in cmd else "0"
    try:
        payload = Path(cmd[-1]).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        sys.stderr.write("can't read buffer file\n")
        return 1
    buffers = state.setdefault("buffers", {})
    assert isinstance(buffers, dict)
    buffers[name] = payload
    _save(socket_path, state)
    return 0


def _handle_paste_buffer(cmd: list[str], socket_path: str) -> int:
    """Paste a named buffer into the pane WITHOUT submitting (``paste-buffer``).

    Appends the buffer's content to the screen — never a submit sentinel, so the
    conformance suite can prove a bracketed paste is non-submitting. ``-d`` drops
    the buffer after pasting (mirroring the real flag).
    """
    state = _load(socket_path)
    if state is None:
        sys.stderr.write("no server running\n")
        return 1
    name = cmd[cmd.index("-b") + 1] if "-b" in cmd else "0"
    buffers = state.setdefault("buffers", {})
    assert isinstance(buffers, dict)
    content = buffers.get(name, "")
    state["screen"] = str(state.get("screen", "")) + content
    if "-d" in cmd:
        buffers.pop(name, None)
    _save(socket_path, state)
    return 0


def main(argv: list[str]) -> int:
    """Run one fake ``tmux`` invocation.

    :param argv: The process argv without the program name (``sys.argv[1:]``).
    :returns: A process exit code (``0`` success, non-zero mirrors a tmux
        failure such as "no server running").
    """
    socket_path: str | None = None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "-S":
            socket_path = argv[i + 1]
            i += 2
        elif token == "-f":
            i += 2  # ignore the config path (always os.devnull)
        else:
            break
    if socket_path is None:
        return 0

    remain_on_exit = False
    rc = 0
    for cmd in _split_semicolons(argv[i:]):
        name = cmd[0]
        if name == "set-option":
            if "remain-on-exit" in cmd and "on" in cmd:
                remain_on_exit = True
        elif name == "new-session":
            rc = _handle_new_session(cmd, socket_path, remain_on_exit=remain_on_exit)
        elif name == "list-panes":
            rc = _handle_list_panes(socket_path)
        elif name == "capture-pane":
            rc = _handle_capture(socket_path)
        elif name == "send-keys":
            rc = _handle_send_keys(cmd, socket_path)
        elif name == "load-buffer":
            rc = _handle_load_buffer(cmd, socket_path)
        elif name == "paste-buffer":
            rc = _handle_paste_buffer(cmd, socket_path)
        elif name == "kill-server":
            rc = _handle_kill_server(socket_path)
        elif name == "kill-session":
            rc = _handle_kill_session(socket_path)
        elif name == "has-session":
            rc = _handle_has_session(socket_path)
        # set-window-option / set-hook / unbind-key / detach-client and any
        # other option command are inert no-ops for the fake.
        if rc != 0:
            return rc
    return rc


def install_fake_tmux(directory: Path) -> Path:
    """Write an executable ``tmux`` shim into *directory* and return its path.

    The shim is a POSIX shell script named ``tmux`` that execs this module
    through the current interpreter, so putting *directory* first on ``PATH``
    makes every ``tmux`` subprocess the backend spawns resolve to the fake.

    POSIX-only by construction: Windows ``CreateProcess`` cannot dispatch a
    bare-name script, and the tmux backend is POSIX-only anyway, so the
    conformance suite skips on Windows.

    :param directory: Directory to drop the ``tmux`` shim into (created if
        absent).
    :returns: The path to the installed shim.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "tmux"
    fake_module = Path(__file__).resolve()
    shim.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake_module))} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main(sys.argv[1:]))
