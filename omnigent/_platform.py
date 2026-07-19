"""Central, dependency-light platform flags and OS-portability helpers.

omnigent grew up on Linux/macOS and bakes a number of POSIX assumptions into
process management, shells, and user identity. This module is the single place
that answers "which OS are we on?" and provides the small portable primitives
that the rest of the package uses instead of branching on :data:`os.name`
ad hoc.

Keep this module import-cheap and free of heavy/optional dependencies: it is
imported very early (and on Windows it must import before any POSIX-only module
would otherwise crash), so it must never pull in ``fcntl``/``termios``/``pty``
or anything platform-specific at module top level.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
import re
import shutil
import sys
from contextlib import suppress
from pathlib import Path

_logger = logging.getLogger(__name__)


# Common global install dirs for npm/homebrew CLIs, probed when a binary isn't
# on ``PATH``. The host daemon snapshots ``PATH`` at spawn and never refreshes
# it, so a CLI installed into an nvm/npm/homebrew bin dir that only interactive
# shell init puts on ``PATH`` is invisible to ``shutil.which``.
def _cli_fallback_dirs() -> tuple[Path, ...]:
    """Return the global install dirs to probe when a CLI isn't on ``PATH``.

    Includes nvm's version-specific bin dirs (``~/.nvm/versions/node/*/bin``),
    where npm global installs land under nvm — the common driver of a CLI that
    a foreground shell sees but the daemon's frozen ``PATH`` doesn't.
    """
    home = Path.home()
    dirs = [
        home / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
        home / ".npm-global" / "bin",
    ]
    # nvm keeps global bins per Node version; newest first so a current install
    # wins over a stale one. Sort by parsed numeric version (so v10 > v9, not
    # the lexicographic order in which "v10" < "v9").
    nvm_versions = home / ".nvm" / "versions" / "node"
    with suppress(OSError):
        version_dirs = [p for p in nvm_versions.iterdir() if p.is_dir()]
        version_dirs.sort(key=lambda p: _parse_node_version(p.name), reverse=True)
        dirs.extend(p / "bin" for p in version_dirs)
    return tuple(dirs)


def _parse_node_version(name: str) -> tuple[int, ...]:
    """Parse an nvm version dir name (e.g. ``"v20.5.0"``) into a sortable tuple.

    Non-numeric or malformed names sort lowest (empty tuple) so real versions
    win over anything unparseable.
    """
    parts = name.lstrip("v").split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return ()


def resolve_cli_binary(name: str, *, env_var: str | None = None) -> str | None:
    """Resolve a CLI binary that may live off the process ``PATH``.

    Checks an optional ``env_var`` override first (an explicit path or a name
    on ``PATH``), then ``PATH`` via :func:`shutil.which`, then a ladder of
    common global install dirs (:func:`_cli_fallback_dirs`). This survives the
    host daemon's frozen ``PATH``, which omits nvm/npm/homebrew bin dirs that
    only interactive shell init adds. Returns ``None`` when none resolve; the
    caller decides whether that's fatal.

    :param name: The binary name, e.g. ``"codex"`` or ``"claude"``.
    :param env_var: Optional env var holding an override path/name, e.g.
        ``"OMNIGENT_CODEX_PATH"``.
    :returns: An absolute path to the executable, or ``None``.
    """
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            resolved = shutil.which(override)
            if resolved:
                return resolved
            if os.access(override, os.X_OK) and os.path.isfile(override):
                return override
            # A set-but-unresolvable override (typo, moved/non-executable
            # binary) silently falling through to PATH would launch a
            # *different* binary than intended — warn so the misconfig surfaces.
            _logger.warning(
                "%s=%r does not resolve to an executable file; falling back to PATH for %r.",
                env_var,
                override,
                name,
            )
    on_path = shutil.which(name)
    if on_path is not None:
        return on_path
    for directory in _cli_fallback_dirs():
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


#: True on native Windows (cmd/PowerShell), i.e. ``os.name == "nt"``. This is
#: *not* true under WSL, where Python reports a Linux platform.
IS_WINDOWS = os.name == "nt"
#: True on any POSIX host (Linux, macOS, BSD, WSL).
IS_POSIX = os.name == "posix"
#: True on Linux specifically (the only platform with bwrap + seccomp).
IS_LINUX = sys.platform.startswith("linux")
#: True on macOS specifically (the seatbelt sandbox platform).
IS_DARWIN = sys.platform == "darwin"

#: Non-sensitive Windows environment variables that a spawned omnigent
#: subprocess needs to function, for env-passthrough allowlists that otherwise
#: assume POSIX names. Python uppercases env keys on Windows, so these match
#: ``os.environ`` as stored; they are absent on POSIX, so including them in an
#: allowlist is a no-op there (only present vars pass through).
#:
#: - ``SYSTEMROOT`` is MANDATORY: Winsock loads its providers from
#:   ``%SystemRoot%\system32\mswsock.dll``, so a child without it dies at
#:   ``import asyncio`` with ``WinError 10106`` (WSAEPROVIDERFAILEDINIT).
#: - ``USERPROFILE`` / ``HOMEDRIVE`` / ``HOMEPATH`` let ``Path.home()`` /
#:   ``expanduser("~")`` resolve (the Windows analog of POSIX ``HOME``).
#: - ``APPDATA`` / ``LOCALAPPDATA`` are where Windows apps (keyring, pip, npm,
#:   …) keep per-user config and cache.
#: - ``TEMP`` / ``TMP`` let native tools extract temporary assets and addons
#:   (including Bun's OpenTUI library).
#: - The rest let a Windows process and shell resolve binaries normally.
#:
#: All are path/identity constants, not credentials — consistent with POSIX
#: ``HOME``/``PATH`` already being allowed.
WINDOWS_ENV_PASSTHROUGH: tuple[str, ...] = (
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "PROCESSOR_LEVEL",
    "PROCESSOR_REVISION",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "TEMP",
    "TMP",
)


def reconfigure_std_streams_for_windows() -> None:
    """Make this process's stdout/stderr tolerate any Unicode on Windows.

    A detached daemon / runner / background-server process on Windows inherits
    ``stdout``/``stderr`` that the parent redirected to a log file, and their
    text codec defaults to the legacy Windows code page (cp1252 / ``charmap``).
    Any of the codebase's many ``✓``/emoji ``print`` statements then raises
    ``UnicodeEncodeError`` on those streams — and in the host tunnel's serve loop
    that error unwinds into a *permanent* reconnect cycle, so the daemon never
    comes online (observed on the codex-native Windows path, whose daemon path
    was unreachable before Windows enablement).

    Reconfigure both streams to UTF-8 with ``errors="backslashreplace"`` so no
    conceivable character can ever crash a background process's output again.
    Best-effort and **Windows-only**: a stream that cannot be reconfigured (not a
    ``TextIOWrapper``, already detached, or ``None``) is left untouched. Call this
    once, as early as possible, at each daemon-spawned process's entry (it is
    invoked from :func:`omnigent.process_logging.configure_process_logging`, the
    shared setup every such process runs before its first output).

    :returns: None.
    """
    if not IS_WINDOWS:
        return
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def default_shell_argv(command: str) -> list[str]:
    """
    Build the argv to run ``command`` through the host's default shell.

    On POSIX this mirrors the long-standing behavior: prefer ``bash`` with
    ``--noprofile --norc`` (skip user rc files for a predictable environment),
    falling back to ``sh -c``. On Windows there is no ``/bin/sh``; route through
    ``cmd.exe`` (``%COMSPEC%``) with ``/c``.

    :param command: The shell command string to execute.
    :returns: An argv list suitable for :func:`subprocess.Popen` (no
        ``shell=True`` needed).
    """
    if IS_WINDOWS:
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/c", command]
    import shutil

    bash = shutil.which("bash")
    if bash:
        return [bash, "--noprofile", "--norc", "-c", command]
    sh = shutil.which("sh") or "/bin/sh"
    return [sh, "-c", command]


#: Interactive shells we honor from ``$SHELL`` for a user terminal. Anything
#: outside this set (or a ``$SHELL`` that doesn't resolve on PATH) falls back to
#: bash for a predictable pane.
_KNOWN_INTERACTIVE_SHELLS = frozenset({"bash", "zsh", "fish", "sh", "dash", "ksh", "tcsh"})

#: Mainstream interactive shells we proactively offer as launch choices (the
#: "New shell" picker), in display order. The user's ``$SHELL`` is always
#: offered first regardless (see :func:`installed_interactive_shells`); this is
#: the set of well-known alternatives we surface beyond it.
_OFFERED_INTERACTIVE_SHELLS = ("bash", "zsh", "fish")


def resolve_windows_interactive_shell_path(command: str) -> str | None:
    """Resolve a native Windows interactive shell to an absolute executable."""
    name = Path(command).name.lower()
    if name in {"cmd", "cmd.exe"}:
        windir = Path(os.environ.get("WINDIR") or os.environ.get("SYSTEMROOT") or r"C:\Windows")
        candidates = (windir / "System32" / "cmd.exe",)
    elif name in {"powershell", "powershell.exe"}:
        windir = Path(os.environ.get("WINDIR") or os.environ.get("SYSTEMROOT") or r"C:\Windows")
        candidates = (windir / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe",)
    elif name in {"pwsh", "pwsh.exe"}:
        roots = [Path(os.environ.get("PROGRAMFILES") or r"C:\Program Files") / "PowerShell"]
        literal_root = Path(r"C:\Program Files\PowerShell")
        if literal_root not in roots:
            roots.append(literal_root)
        candidates = tuple(
            path
            for root in roots
            for path in sorted(
                root.glob("*/pwsh.exe"),
                key=lambda item: tuple(int(part) for part in re.findall(r"\d+", item.parent.name)),
                reverse=True,
            )
        )
        path_on_path = shutil.which("pwsh")
        if path_on_path:
            candidates += (Path(path_on_path),)
    else:
        return None

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def default_interactive_shell() -> str:
    """
    Basename of the user's login shell for an interactive terminal.

    Reads ``$SHELL`` and keeps its basename when it names a known shell that
    resolves on PATH; otherwise falls back to ``"bash"``. Returns a basename
    (not the absolute ``$SHELL`` path) so it stays PATH-resolvable when the
    terminal launches under a runner on a different host than the one that read
    the env.

    :returns: A shell basename such as ``"zsh"``, ``"fish"``, or ``"bash"``.
    """
    if IS_WINDOWS:
        # Git Bash remains the default tool-compatible interactive shell.
        return "bash"
    import shutil

    name = os.path.basename(os.environ.get("SHELL", "")).strip()
    if name in _KNOWN_INTERACTIVE_SHELLS and shutil.which(name):
        return name
    return "bash"


def installed_interactive_shells() -> list[str]:
    """
    Ordered, deduped shell basenames to offer for a new interactive terminal.

    The user's login shell (:func:`default_interactive_shell`) comes first — so
    the "New shell" affordance can treat entry ``[0]`` as the click default —
    followed by any mainstream alternatives (bash/zsh/fish) that resolve on
    PATH. Always non-empty (the default is always present, and bash is the
    ultimate fallback).

    :returns: Basenames such as ``["zsh", "bash", "fish"]`` — the default first.
    """
    ordered = [default_interactive_shell()]
    if IS_WINDOWS:
        # ConPTY hosts native shells, but its runner has a deliberately stripped
        # PATH, so only offer executables found at resolvable absolute paths.
        powershell = "pwsh" if resolve_windows_interactive_shell_path("pwsh") else "powershell"
        if resolve_windows_interactive_shell_path(powershell):
            ordered.append(powershell)
        if resolve_windows_interactive_shell_path("cmd"):
            ordered.append("cmd")
        return ordered
    import shutil

    for name in _OFFERED_INTERACTIVE_SHELLS:
        if name not in ordered and shutil.which(name):
            ordered.append(name)
    return ordered


def stable_user_id() -> str:
    """
    A stable, filesystem-safe token identifying the current OS user.

    Used to namespace per-user scratch directories (e.g.
    ``omnigent-<id>`` / ``claude-<id>`` under the temp dir). On POSIX this is
    the numeric uid, matching historical behavior. Windows has no ``getuid``;
    derive a short hex digest from the login name so the value is stable across
    runs and safe to embed in a path.

    The digest is for path namespacing only — not security — so ``getuser``'s
    value never needs to be recoverable or collision-proof against an
    adversary; it just needs to be stable and filesystem-safe. SHA-256 with
    ``usedforsecurity=False`` documents that intent (and avoids flagging SHA-1).

    :returns: A short string with no path separators or shell-special chars.
    """
    if IS_POSIX and hasattr(os, "getuid"):
        return str(os.getuid())
    try:
        name = getpass.getuser()
    except (OSError, KeyError, ModuleNotFoundError):
        name = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    return hashlib.sha256(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def resolve_repo_symlink(path: Path) -> Path:
    """
    Follow a Git symlink that a no-symlink Windows checkout left as a text file.

    On Windows with ``core.symlinks=false`` (the default when Developer Mode is
    off and Git was not run elevated), Git materializes a repository symlink as a
    *regular file* whose entire content is the link target — e.g. the checked-out
    ``omnigent/resources/examples/polly`` is a 23-byte file containing
    ``../../../examples/polly`` rather than a link to that directory. Code that
    expects to open the linked directory then reads this stub instead (the
    symptom: ``expected YAML mapping at top level, got str``).

    Detect that exact shape — a small, single-line regular file whose content,
    resolved relative to the stub's parent, names an existing path — and return
    the real target. Everything else (real directories, real symlinks, genuine
    single-file specs, multi-line or unresolvable content) is returned
    unchanged. No-op off Windows, where the symlink is followed natively.

    :param path: The path as resolved from packaged resources.
    :returns: The dereferenced target on Windows when *path* is a Git-symlink
        stub, otherwise *path* unchanged.
    """
    if not IS_WINDOWS:
        return path
    try:
        if not path.is_file() or path.stat().st_size > 4096:
            return path
        body = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return path
    target = body.strip()
    if not target or "\n" in target:
        return path
    candidate = path.parent / target
    if candidate.exists():
        return candidate.resolve()
    return path
