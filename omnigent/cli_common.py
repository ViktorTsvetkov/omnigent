"""Shared constants and guards for the CLI, importable without ``omnigent.cli``.

The native coding-agent subcommands live in :mod:`omnigent.cli_native`, which
``omnigent.cli`` imports at module load to register them on the ``cli`` group.
Click evaluates command decorators at import time, so any module-level name a
decorator references (``flag_value=``, help-string interpolation) must resolve
before the command object is built. Keeping those names here — in a leaf module
that imports nothing from ``omnigent.cli`` — lets both ``cli`` and ``cli_native``
import them without an import cycle.
"""

from __future__ import annotations

import click

from omnigent._platform import IS_WINDOWS

# Click ``flag_value`` for bare ``--resume`` (no arg). Must exist before any
# command's decorator evaluates.
RESUME_PICKER_SENTINEL = "__resume_picker__"

# Env var that force-enables native Claude startup timing marks. Referenced in a
# command's ``--profile-startup`` help string, so it is decorator-time state.
CLAUDE_STARTUP_PROFILE_ENV_VAR = "OMNIGENT_CLAUDE_STARTUP_PROFILE"


def reject_native_on_windows(harness: str) -> None:
    """Fail a native (tmux/PTY) harness command with an actionable message.

    The ``omnigent claude`` / ``codex`` / ``cursor`` native wrappers drive a
    private tmux server and PTY, which don't exist on Windows. Point users at
    the SDK harnesses / web UI instead of letting them hit a tmux crash.

    :param harness: The native command name, e.g. ``"claude"``.
    :raises click.ClickException: Always, when running on Windows.
    """
    if IS_WINDOWS:
        raise click.ClickException(
            f"`omnigent {harness}` (native tmux/PTY terminal) is not supported on "
            "Windows. Use an SDK-based harness via `omnigent run <agent.yaml>` "
            "or the web UI."
        )


def ensure_native_terminal_backend_available(harness: str) -> None:
    """Gate a hosting-only native harness on a usable terminal backend (Windows).

    Replaces the hard Windows rejection (:func:`reject_native_on_windows`) for a
    *side-channel* harness whose prompt/interrupt path is its own app-server, not
    the terminal — the terminal only needs to *host* the CLI. Such a harness can
    run wherever a platform-native terminal multiplexer backend is available, so
    on Windows this resolves the seam's backend for this platform and probes its
    dependencies via
    :meth:`~omnigent.inner.terminal.TerminalBackend.ensure_available`. ConPTY is
    the default and checks pywinpty; an override such as herdr checks its own
    requirements. An unavailable backend raises the seam's actionable error as
    a :class:`click.ClickException`.

    POSIX is left **byte-identical** to the old no-op: tmux hosting there is
    unchanged and its availability keeps failing at terminal-creation time as
    before, so this never adds an early gate on the POSIX path.

    :param harness: The native command name, e.g. ``"codex"`` (for messages).
    :raises click.ClickException: On Windows when no terminal backend can host
        the harness.
    """
    if not IS_WINDOWS:
        return
    from omnigent.inner.terminal import select_terminal_backend_class

    try:
        backend_cls = select_terminal_backend_class()
        backend_cls.ensure_available()
    except RuntimeError as exc:
        raise click.ClickException(
            f"`omnigent {harness}` cannot start on Windows: {exc}"
        ) from exc
