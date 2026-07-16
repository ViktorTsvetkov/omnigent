"""An in-process fake :class:`~omnigent.inner.terminal.TerminalBackend`.

This is a full-protocol, in-memory stand-in for a terminal multiplexer backend,
built for higher-layer tests — the native bridges, the ``sys_terminal`` tools,
the registry/factory wiring — that need to run a real
:class:`~omnigent.inner.terminal.TerminalInstance` (or the
``create_terminal_instance`` factory) without a tmux server, a fake ``tmux``
binary on ``PATH``, patched availability flags, or subprocess argv sinks. Every
touchpoint that :class:`TmuxBackend` shells out for — launch, liveness, capture,
input, teardown, the status link, detach-on-exit — is a plain method mutating
in-memory state here, so the backend runs identically on POSIX and native
Windows and gives the program its first Windows-native
:class:`~tests.inner.terminal_backend_conformance.BackendConformanceSuite`
coverage.

Using it in a test
------------------

Two ways in, matching the two things tests need:

1. **Drive the backend directly** (bridge/tool unit tests, and the conformance
   suite). Construct one, ``await`` its protocol methods, and read back the
   recorded state::

       backend = FakeBackend()
       await backend.launch(request)
       await backend.send_text("hello")
       assert "hello" in await backend.capture()
       assert backend.sent_text == ["hello"]

2. **Run a real ``TerminalInstance`` on it** (registry/factory tests). Register
   the class under a name — with ``monkeypatch.setitem`` so it never leaks into
   production selection, exactly as the selection tests register their stub —
   then build an instance with that ``backend_name`` and script its backend::

       monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", FakeBackend)
       instance = TerminalInstance(..., backend_name="fake")
       backend = instance._backend  # the FakeBackend built by the ctor hook
       backend.push_liveness([Liveness.INNER_EXITED])

   The instance builds its backend through
   :meth:`~omnigent.inner.terminal.TerminalBackend.construct_for_instance`
   (overridden below), so no dispatcher edit is needed — the same seam a future
   ``HerdrBackend`` (#11) uses.

Scripting snapshots and liveness
--------------------------------

By default the fake computes its own answers from what it has been told to do:
after ``launch`` the endpoint is :attr:`Liveness.ALIVE`, ``capture`` returns the
accumulated screen, ``close`` makes it :attr:`Liveness.ENDPOINT_GONE`. Two
conventions make an inner exit expressible with no timing:

- A launch command whose sole token is :data:`EXIT_SENTINEL` models a process
  that exits the instant it launches — with ``keep_alive_after_exit`` the
  endpoint is kept and reports :attr:`Liveness.INNER_EXITED` with
  :data:`FINAL_SCREEN_MARKER` still capturable; without it the endpoint is gone
  (:attr:`Liveness.ENDPOINT_GONE`).
- A named ``Enter`` key appends :data:`SUBMIT_SENTINEL` to the screen, so a test
  can prove a literal ``send_text`` paste never submits while an explicit
  ``send_keys(["Enter"])`` does.

For tests that need a *sequence* of outcomes — a TUI's composer settling and
then its process exiting — override the auto answers with scripts:

- :meth:`push_snapshots` queues the screens successive ``capture`` /
  ``capture_sync`` calls return; a ``None`` entry makes that call raise
  ``RuntimeError`` (the "host endpoint went away" signal callers key off).
- :meth:`push_liveness` queues the verdicts successive ``liveness`` /
  ``liveness_sync`` calls return.
- :meth:`break_liveness_probe` makes every subsequent liveness probe report
  :attr:`Liveness.UNKNOWN`, modeling a probe that cannot run.

Both queues are sticky: once a script's last entry is reached it is returned for
every further call, so a scripted terminal state persists across a watcher's
polling loop. An empty (never-set) script falls back to the auto answer, which
is what keeps the conformance suite — which never scripts — driving real
computed behavior.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

from omnigent.inner.terminal import (
    Liveness,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalLaunchRequest,
)

# A launch command whose sole argv token is this sentinel models an inner
# process that exits the instant it launches (see the module docstring). Mirrors
# the ``_fake_tmux`` convention so the two doubles read the same way, but is
# defined locally so this in-process double imports nothing subprocess-shaped.
EXIT_SENTINEL = "__omnigent_fake_backend_exit__"

# Appended to the screen when a named ``Enter`` key is sent, so a test can prove
# a literal ``send_text`` paste never submits while an explicit Enter does.
SUBMIT_SENTINEL = "[fake-backend:submit]"

# The final screen the fake preserves after an inner process that exited while
# keep-alive was on — lets a test prove the final frame stays capturable past
# inner exit.
FINAL_SCREEN_MARKER = "[fake-backend:inner-exited]\n"


def key_marker(key: str) -> str:
    """Return the observable screen token the fake appends for a named *key*.

    A named-key send is otherwise invisible in a capture; the fake echoes each
    key as ``<KEY>`` so tests can assert delivery.

    :param key: A named key in Omnigent's neutral vocabulary, e.g. ``"Escape"``.
    :returns: The token appended to the screen, e.g. ``"<Escape>"``.
    """
    return f"<{key}>"


class FakeBackend(TerminalBackend):
    """Full-protocol, in-process terminal backend for higher-layer tests.

    Implements every :class:`TerminalBackend` method against in-memory state and
    records what it was asked to do (:attr:`sent_text`, :attr:`sent_keys`,
    :attr:`status_link_updates`, :attr:`detach_calls`, :attr:`launch_request`)
    for tests to assert on. See the module docstring for usage and scripting.
    """

    name = "fake"
    # Advertise no optional capabilities by default: a neutral double should not
    # imply a native popup / busy-state / push channel it does not emulate.
    # Tests that care pass a tailored ``capabilities`` to the constructor.
    capabilities = TerminalBackendCapabilities()
    # In-process, so it runs on both POSIX and native Windows — the point of the
    # fake for Windows-native conformance coverage.
    platforms = frozenset({"posix", "windows"})

    def __init__(
        self,
        *,
        socket_path: Path | None = None,
        target: str = "main",
        capabilities: TerminalBackendCapabilities | None = None,
    ) -> None:
        """Build a fresh, unlaunched fake backend.

        :param socket_path: Recorded for introspection; the fake keeps its state
            in memory and never touches the path. Accepted so the construction
            hook can pass it uniformly.
        :param target: Session/pane target name, recorded for introspection.
        :param capabilities: Optional capability override for this instance;
            defaults to the all-``False`` class declaration.
        """
        self.socket_path = socket_path
        self.target = target
        if capabilities is not None:
            self.capabilities = capabilities

        # --- Recorded interactions (for test assertions) --------------------
        self.launched = False
        self.closed = False
        self.launch_request: TerminalLaunchRequest | None = None
        self.sent_text: list[str] = []
        self.sent_keys: list[list[str]] = []
        self.status_link_updates: list[str | None] = []
        self.detach_calls = 0

        # --- Auto-computed state --------------------------------------------
        self._screen = ""
        self._endpoint_alive = True
        self._pane_dead = False
        self._probe_can_run = True
        self._status_link: str | None = None

        # --- Optional scripts (override the auto answers) -------------------
        self._snapshot_script: deque[str | None] = deque()
        self._liveness_script: deque[Liveness] = deque()

    # ------------------------------------------------------------------ hooks

    @classmethod
    def construct_for_instance(cls, *, socket_path: Path, target: str) -> TerminalBackend:
        """Build a fake bound to a terminal instance's endpoint (the ctor hook).

        Lets ``_construct_terminal_backend`` build this backend for a
        :class:`TerminalInstance` when it is registered under a name, with no
        dispatcher edit — the same seam #11's ``HerdrBackend`` uses.
        """
        return cls(socket_path=socket_path, target=target)

    @classmethod
    def ensure_available(cls) -> None:
        """No-op: an in-process backend is always available."""
        del cls

    # -------------------------------------------------------------- scripting

    def set_screen(self, screen: str) -> None:
        """Set the current auto-mode screen returned by ``capture``."""
        self._screen = screen

    def push_snapshots(self, snapshots: Sequence[str | None]) -> None:
        """Queue the screens successive captures return (sticky on the last).

        A ``None`` entry makes that ``capture`` / ``capture_sync`` call raise
        ``RuntimeError`` — the "host endpoint went away" signal that
        :class:`TerminalInstance`'s watchers and read path key off. Once the
        queue is down to its last entry that entry is returned for every further
        call, so a scripted state persists across a polling loop.
        """
        self._snapshot_script = deque(snapshots)

    def push_liveness(self, verdicts: Sequence[Liveness]) -> None:
        """Queue the verdicts successive liveness probes return (sticky).

        Consumed by both :meth:`liveness` and :meth:`liveness_sync`. Once down to
        the last verdict it is returned for every further probe, so a scripted
        terminal state (e.g. ``INNER_EXITED``) persists across a watcher loop.
        """
        self._liveness_script = deque(verdicts)

    def break_liveness_probe(self) -> None:
        """Make every subsequent liveness probe report ``UNKNOWN``.

        Models a probe that cannot run (the conformance safe-degradation case).
        """
        self._probe_can_run = False

    @staticmethod
    def _next_scripted(script: deque[object], auto: object) -> object:
        """Pop the next scripted value, staying sticky on the last, else *auto*.

        With more than one entry queued, consume the head; with exactly one,
        peek it (so it repeats for every further call); with none, return the
        auto-computed *auto* value.
        """
        if len(script) > 1:
            return script.popleft()
        if script:
            return script[0]
        return auto

    # -------------------------------------------------------------- protocol

    async def launch(self, request: TerminalLaunchRequest) -> None:
        """Record the launch and set state from the request's inner command."""
        self._apply_launch(request)

    def _apply_launch(self, request: TerminalLaunchRequest) -> None:
        """Set in-memory state for a launch (shared by any sync/async entry)."""
        self.launched = True
        self.launch_request = request
        self._status_link = request.status_link
        inner = request.command[0].strip() if request.command else ""
        if inner == EXIT_SENTINEL:
            if request.keep_alive_after_exit:
                # Keep the dead endpoint and its final frame (tmux
                # remain-on-exit parity) → INNER_EXITED.
                self._endpoint_alive = True
                self._pane_dead = True
                self._screen = FINAL_SCREEN_MARKER
            else:
                # The endpoint goes with the process → ENDPOINT_GONE.
                self._endpoint_alive = False
        else:
            self._endpoint_alive = True
            self._pane_dead = False
            self._screen = ""

    def _auto_liveness(self) -> Liveness:
        """Compute the liveness verdict from current in-memory state."""
        if not self._probe_can_run:
            return Liveness.UNKNOWN
        if not self._endpoint_alive:
            return Liveness.ENDPOINT_GONE
        if self._pane_dead:
            return Liveness.INNER_EXITED
        return Liveness.ALIVE

    async def liveness(self) -> Liveness:
        """Async liveness probe (scripted verdict, else auto-computed)."""
        return self.liveness_sync()

    def liveness_sync(self) -> Liveness:
        """Sync liveness probe (scripted verdict, else auto-computed)."""
        verdict = self._next_scripted(self._liveness_script, self._auto_liveness())
        assert isinstance(verdict, Liveness)
        return verdict

    async def close(self) -> None:
        """Tear the endpoint down. Idempotent; never raises."""
        self.closed = True
        self._endpoint_alive = False

    async def send_text(self, text: str) -> None:
        """Append literal *text* to the screen (non-submitting) and record it."""
        self._require_endpoint()
        self._screen += text
        self.sent_text.append(text)

    async def send_keys(self, keys: Sequence[str]) -> None:
        """Append each named key's marker to the screen; ``Enter`` submits."""
        self._require_endpoint()
        for key in keys:
            self._screen += key_marker(key)
            if key == "Enter":
                self._screen += SUBMIT_SENTINEL
        self.sent_keys.append(list(keys))

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Snapshot the screen (scripted, else the accumulated screen)."""
        return self._capture()

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        """Sync sibling of :meth:`capture`; same scripting and error semantics."""
        return self._capture()

    def _capture(self) -> str:
        """Return the next scripted snapshot, or the live screen.

        A scripted ``None`` — or a gone endpoint in auto mode — raises
        ``RuntimeError``, the "host went away" signal callers read as a stop.
        """
        if self._snapshot_script:
            snapshot = self._next_scripted(self._snapshot_script, self._screen)
            if snapshot is None:
                raise RuntimeError("fake terminal backend: host endpoint gone (scripted)")
            assert isinstance(snapshot, str)
            return snapshot
        if not self._endpoint_alive:
            raise RuntimeError("fake terminal backend: host endpoint gone")
        return self._screen

    async def set_status_link(self, link: str | None) -> None:
        """Record a status-link update (the fake always tracks it)."""
        self._status_link = link
        self.status_link_updates.append(link)

    async def detach_display_clients(self) -> None:
        """Record a detach request (async path)."""
        self.detach_calls += 1

    def detach_display_clients_sync(self) -> None:
        """Record a detach request (sync path)."""
        self.detach_calls += 1

    # --------------------------------------------------------------- helpers

    def _require_endpoint(self) -> None:
        """Raise ``RuntimeError`` when the endpoint is gone (input rejected)."""
        if not self._endpoint_alive:
            raise RuntimeError("fake terminal backend: host endpoint gone")
