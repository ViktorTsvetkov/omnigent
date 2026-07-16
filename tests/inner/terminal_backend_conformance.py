"""Parametrized conformance suite for :class:`TerminalBackend` adapters.

This module is the *mechanical definition* of what any terminal multiplexer
backend must do to be a drop-in behind the terminal seam. It is deliberately
backend-agnostic: it drives only the public
:class:`~omnigent.inner.terminal.TerminalBackend` surface (launch / liveness /
close / send_text / send_keys / capture) and asserts the contract each backend
promises — launch and kill semantics, the endpoint-exists-vs-agent-alive
:class:`~omnigent.inner.terminal.Liveness` verdict, keep-alive-after-exit,
non-submitting multi-line paste, named-key delivery, snapshot content, and safe
degradation to :attr:`Liveness.UNKNOWN` when a probe cannot run.

Opting a backend in
------------------------

The suite is a base class, :class:`BackendConformanceSuite`, plus a small
per-backend adapter, :class:`ConformanceAdapter`. A backend opts in with one
concrete test module:

1. Implement a :class:`ConformanceAdapter` that knows how to construct the
   backend against a deterministic test double (e.g. a scripted fake binary on
   ``PATH``) and how to express the few backend-specific probes the suite needs
   (an inner command that stays alive, one that exits immediately, how to break
   a liveness probe, and the observable paste/key conventions its double uses).
2. Add a ``test_*.py`` module whose test class subclasses
   :class:`BackendConformanceSuite` and overrides the ``adapter`` fixture to
   return that adapter.

See ``tests/inner/test_terminal_backend_conformance_tmux.py`` for the tmux
opt-in, which drives the real ``TmuxBackend`` against a scripted fake ``tmux``
binary (``tests/inner/_fake_tmux.py``). A future ``HerdrBackend`` opts in the
same way — a second adapter and a second one-class module — with no change to
this suite.

Capability-based differences
----------------------------

Where a behavior legitimately varies by backend, the adapter declares it rather
than the suite hard-coding tmux's answer. Today the only such axis is whether a
backend can preserve a *dead* endpoint after the inner process exits
(:attr:`ConformanceAdapter.preserves_dead_endpoint`): tmux keeps the pane
(``remain-on-exit`` → :attr:`Liveness.INNER_EXITED`), whereas a backend like
herdr destroys the pane on exit (:attr:`Liveness.ENDPOINT_GONE`). The
keep-alive test asserts the verdict the adapter declares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pytest

from omnigent.inner.terminal import (
    Liveness,
    TerminalBackend,
    TerminalLaunchRequest,
)


class ConformanceAdapter(ABC):
    """Backend-specific glue that lets one backend run the conformance suite.

    An adapter constructs the backend under test against a deterministic test
    double and expresses the handful of backend-specific probes the suite
    cannot phrase generically. Everything else in the suite is driven through
    the backend's public surface.
    """

    #: The concrete backend class under test (its name / capabilities are read
    #: for diagnostics and future capability-gated assertions).
    backend_cls: type[TerminalBackend]

    #: Substring the backend's test double writes to the pane ONLY when a real
    #: submit (an ``Enter`` key) happens. The paste test asserts it is absent.
    submit_sentinel: str

    #: The pane content the backend's double preserves after an inner process
    #: that exited under keep-alive. Only consulted when
    #: :attr:`preserves_dead_endpoint` is ``True``.
    final_screen_marker: str

    #: Whether the backend preserves a dead endpoint after the inner process
    #: exits under keep-alive (tmux ``remain-on-exit`` → ``INNER_EXITED``) or
    #: drops it (``ENDPOINT_GONE``). Backends that cannot keep a dead pane set
    #: this ``False``.
    preserves_dead_endpoint: bool = True

    #: Named keys used by the named-key delivery test. Chosen so every backend
    #: can express them; a backend lacking one narrows this tuple.
    sample_keys: tuple[str, ...] = ("Escape", "C-c")

    @abstractmethod
    def make_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TerminalBackend:
        """Install the test double and return a fresh, unlaunched backend.

        Implementations put any fake binary on ``PATH`` (via *monkeypatch*) and
        bind the backend to a private endpoint under *tmp_path*.
        """
        raise NotImplementedError

    @abstractmethod
    def launch_request(
        self, command: list[str], *, keep_alive_after_exit: bool = False
    ) -> TerminalLaunchRequest:
        """Build a launch request for *command* against the test double."""
        raise NotImplementedError

    @abstractmethod
    def alive_command(self) -> list[str]:
        """An inner command whose process stays running after launch."""
        raise NotImplementedError

    @abstractmethod
    def exiting_command(self) -> list[str]:
        """An inner command whose process has exited by the time launch returns."""
        raise NotImplementedError

    @abstractmethod
    def key_marker(self, key: str) -> str:
        """The observable pane token the double writes for a named *key*."""
        raise NotImplementedError

    @abstractmethod
    def break_probe(
        self, backend: TerminalBackend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Make a subsequent liveness probe unable to run (→ ``UNKNOWN``)."""
        raise NotImplementedError


class BackendConformanceSuite:
    """The backend-agnostic conformance tests.

    Not collected on its own (the name is not ``Test*``). A backend opts in by
    subclassing this in a ``test_*.py`` module and overriding the ``adapter``
    fixture. See the module docstring.
    """

    @pytest.fixture
    def adapter(self) -> ConformanceAdapter:
        """Return the backend adapter under test. Subclasses MUST override."""
        raise NotImplementedError("BackendConformanceSuite subclasses must override `adapter`")

    async def test_launch_makes_endpoint_alive(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After launch, both liveness probes report the endpoint ALIVE."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        try:
            assert await backend.liveness() == Liveness.ALIVE
            assert backend.liveness_sync() == Liveness.ALIVE
        finally:
            await backend.close()

    async def test_close_makes_endpoint_gone(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """close() tears the endpoint down; liveness then reports ENDPOINT_GONE."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        await backend.close()
        assert await backend.liveness() == Liveness.ENDPOINT_GONE

    async def test_capture_returns_pane_content(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A snapshot returns the pane screen, sync and async alike."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        try:
            await backend.send_text("hello conformance 123")
            assert "hello conformance 123" in await backend.capture()
            assert "hello conformance 123" in backend.capture_sync()
        finally:
            await backend.close()

    async def test_send_text_multiline_is_not_submitted(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-line paste lands intact and is NOT submitted.

        The whole point of ``send_text``: newlines are content, not Enter
        presses, so a pasted code block arrives verbatim for the caller to
        submit separately.
        """
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        try:
            pasted = "first line\nsecond line\nthird line"
            await backend.send_text(pasted)
            snapshot = await backend.capture()
            assert pasted in snapshot
            for line in pasted.split("\n"):
                assert line in snapshot
            assert adapter.submit_sentinel not in snapshot
        finally:
            await backend.close()

    async def test_send_keys_delivers_named_keys(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Named keys are delivered to the pane in the backend's key vocabulary."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        try:
            await backend.send_keys(list(adapter.sample_keys))
            snapshot = await backend.capture()
            for key in adapter.sample_keys:
                assert adapter.key_marker(key) in snapshot
        finally:
            await backend.close()

    async def test_enter_key_submits(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit Enter key submits, unlike a literal paste.

        Pairs with the paste test: proves the double distinguishes a literal
        send from a submit, so the paste test's absence-of-submit assertion is
        meaningful rather than vacuous.
        """
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        try:
            await backend.send_text("run me")
            assert adapter.submit_sentinel not in await backend.capture()
            await backend.send_keys(["Enter"])
            assert adapter.submit_sentinel in await backend.capture()
        finally:
            await backend.close()

    async def test_inner_exit_without_keep_alive_is_endpoint_gone(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without keep-alive, an inner exit takes the endpoint with it."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(
            adapter.launch_request(adapter.exiting_command(), keep_alive_after_exit=False)
        )
        try:
            assert await backend.liveness() == Liveness.ENDPOINT_GONE
        finally:
            await backend.close()

    async def test_keep_alive_after_exit_preserves_endpoint(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With keep-alive, an inner exit degrades per the backend's capability.

        Backends that can keep a dead pane report INNER_EXITED and keep the
        final screen capturable; backends that cannot report ENDPOINT_GONE.
        """
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(
            adapter.launch_request(adapter.exiting_command(), keep_alive_after_exit=True)
        )
        try:
            if adapter.preserves_dead_endpoint:
                assert await backend.liveness() == Liveness.INNER_EXITED
                assert adapter.final_screen_marker in await backend.capture()
            else:
                assert await backend.liveness() == Liveness.ENDPOINT_GONE
        finally:
            await backend.close()

    async def test_liveness_degrades_to_unknown_when_probe_cannot_run(
        self, adapter: ConformanceAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe that cannot run degrades to UNKNOWN, not a false ENDPOINT_GONE."""
        backend = adapter.make_backend(tmp_path, monkeypatch)
        await backend.launch(adapter.launch_request(adapter.alive_command()))
        # No close(): break_probe intentionally leaves the probe unable to run,
        # and the test double spawns no real process to reap.
        adapter.break_probe(backend, monkeypatch, tmp_path)
        assert await backend.liveness() == Liveness.UNKNOWN
        assert backend.liveness_sync() == Liveness.UNKNOWN
