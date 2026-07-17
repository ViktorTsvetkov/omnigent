"""Unit tests for terminal multiplexer backend selection.

Covers the selection contract added in the backend-abstraction work: the
name→class registry, the resolution precedence (per-terminal spec → env →
user config → platform default), the loud failure modes (unknown name,
platform mismatch, no-backend-for-platform), the binary availability gate, and
the factory/instance wiring that puts the selected backend to use.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest

import omnigent.inner.terminal as terminal_mod
from omnigent.inner.datamodel import OSEnvSpec, TerminalEnvSpec
from omnigent.inner.terminal import (
    HerdrBackend,
    Liveness,
    TerminalBackend,
    TerminalLaunchRequest,
    TmuxBackend,
    build_prompt_delivery,
    create_terminal_instance,
    resolve_terminal_backend_name,
    select_terminal_backend_class,
)


class _FakeBackend(TerminalBackend):
    """Minimal cross-platform backend for registry/selection tests.

    Implements the abstract surface with inert stubs — these tests exercise
    selection, not I/O, so the methods only need to exist for the class to
    instantiate. Doubles as the handoff shape for the conformance/FakeBackend
    tickets: registering a fake is one ``register_terminal_backend`` call.
    """

    name = "fake"
    capabilities = terminal_mod.TerminalBackendCapabilities()
    platforms = frozenset({"posix", "windows"})

    async def launch(self, request: TerminalLaunchRequest) -> None:
        del request

    async def liveness(self) -> Liveness:
        return Liveness.UNKNOWN

    def liveness_sync(self) -> Liveness:
        return Liveness.UNKNOWN

    async def close(self) -> None: ...

    async def send_text(self, text: str) -> None:
        del text

    async def send_keys(self, keys: Sequence[str]) -> None:
        del keys

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        del ansi, scrollback
        return ""

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        del ansi, scrollback
        return ""

    def send_text_sync(self, text: str) -> None:
        del text

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        del keys

    def paste_without_submit_sync(self, text: str) -> None:
        del text

    def kill_session_sync(self) -> None: ...


def _force_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make selection resolve as POSIX regardless of the test host."""
    monkeypatch.setattr(terminal_mod, "IS_WINDOWS", False)


def _force_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make selection resolve as native Windows regardless of the test host."""
    monkeypatch.setattr(terminal_mod, "IS_WINDOWS", True)


def _isolate_config(monkeypatch: pytest.MonkeyPatch, config_home: Path) -> None:
    """Point config + env reads at a scratch home with no stray override."""
    monkeypatch.setenv("OMNIGENT_CONFIG_HOME", str(config_home))
    monkeypatch.delenv(terminal_mod._TERMINAL_BACKEND_ENV_VAR, raising=False)


def _write_backend_config(config_home: Path, value: str | None) -> None:
    """Write ``terminal.backend: <value>`` into a scratch config.yaml.

    :param config_home: Directory used as ``OMNIGENT_CONFIG_HOME``.
    :param value: Backend value to persist, or ``None`` for no ``terminal``
        table at all.
    """
    body = "" if value is None else f"terminal:\n  backend: {value}\n"
    (config_home / "config.yaml").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_tmux_backend_is_registered() -> None:
    """The tmux backend registers itself at import under its ``name``."""
    assert terminal_mod._TERMINAL_BACKENDS["tmux"] is TmuxBackend


def test_register_terminal_backend_adds_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly registered backend becomes selectable by its name.

    Demonstrates the extension seam the herdr ticket (and the FakeBackend
    conformance tickets) use — registration is the only wiring needed above
    the seam.
    """
    monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", _FakeBackend)
    assert terminal_mod._TERMINAL_BACKENDS["fake"] is _FakeBackend


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


def test_absent_field_resolves_to_tmux_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The compatibility contract: no selection anywhere → tmux on POSIX.

    This is what keeps every persisted spec created before the backend field
    existed working unchanged.
    """
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)  # no config file written
    assert resolve_terminal_backend_name() == "tmux"


def test_spec_beats_env_beats_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Precedence: per-terminal spec → env var → user config."""
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    _write_backend_config(tmp_path, "cfgval")

    # Config only.
    assert resolve_terminal_backend_name() == "cfgval"

    # Env beats config.
    monkeypatch.setenv(terminal_mod._TERMINAL_BACKEND_ENV_VAR, "envval")
    assert resolve_terminal_backend_name() == "envval"

    # Spec beats env (and config).
    assert resolve_terminal_backend_name(spec_backend="specval") == "specval"


def test_blank_env_and_blank_config_fall_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Blank/whitespace values at a tier don't shadow lower tiers."""
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    _write_backend_config(tmp_path, '"   "')  # quoted whitespace string
    monkeypatch.setenv(terminal_mod._TERMINAL_BACKEND_ENV_VAR, "   ")
    # Both blank → platform default.
    assert resolve_terminal_backend_name() == "tmux"


def test_config_without_terminal_table_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A config file with no ``terminal`` table → platform default."""
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    _write_backend_config(tmp_path, None)
    assert resolve_terminal_backend_name() == "tmux"


# ---------------------------------------------------------------------------
# Platform default / availability
# ---------------------------------------------------------------------------


def test_windows_defaults_to_herdr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Native Windows with nothing selected → the herdr backend (#11).

    The Windows-native default the herdr adapter registers; binary/protocol
    availability is enforced separately by ``HerdrBackend.ensure_available`` at
    the factory, so resolution itself just names herdr.
    """
    _force_windows(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    assert resolve_terminal_backend_name() == "herdr"


def test_platform_without_default_raises_availability_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A platform with no default backend → clear availability RuntimeError.

    Both registered platforms now have a default (tmux on POSIX, herdr on
    Windows), so the no-default path is exercised via an unknown platform tag —
    the same loud failure any future unsupported platform would hit, distinct
    from the platform-mismatch error.
    """
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setattr(terminal_mod, "_current_platform_tag", lambda: "plan9")
    with pytest.raises(RuntimeError, match=r"No terminal multiplexer backend"):
        resolve_terminal_backend_name()


def test_select_returns_tmux_on_posix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """POSIX default selection yields the tmux backend class."""
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    assert select_terminal_backend_class() is TmuxBackend


# ---------------------------------------------------------------------------
# Loud failure modes
# ---------------------------------------------------------------------------


def test_unknown_backend_name_lists_known(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unknown name fails loudly and lists the known backends."""
    _force_posix(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        select_terminal_backend_class(spec_backend="does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "tmux" in message  # the known-backends list


def test_platform_incompatible_backend_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selecting tmux on native Windows fails with a platform error.

    tmux advertises ``platforms={'posix'}``; asking for it explicitly on
    Windows is a platform mismatch, distinct from the no-default case.
    """
    _force_windows(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="does not support this platform"):
        select_terminal_backend_class(spec_backend="tmux")


def test_registered_cross_platform_backend_selects_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A backend advertising windows support selects there without error.

    Guards that the platform gate keys off the backend's declared platforms,
    not a hardwired POSIX assumption — the property the herdr ticket relies on.
    """
    _force_windows(monkeypatch)
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", _FakeBackend)
    assert select_terminal_backend_class(spec_backend="fake") is _FakeBackend


# ---------------------------------------------------------------------------
# Binary availability gate
# ---------------------------------------------------------------------------


def test_tmux_ensure_available_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing tmux binary yields an actionable install-hint error."""
    monkeypatch.setattr(terminal_mod, "_tmux_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        TmuxBackend.ensure_available()
    message = str(excinfo.value)
    assert "tmux" in message
    assert "install" in message.lower()


def test_tmux_ensure_available_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ensure_available`` is a no-op when tmux is on PATH."""
    monkeypatch.setattr(terminal_mod, "_tmux_available", lambda: True)
    TmuxBackend.ensure_available()  # must not raise


def test_base_backend_ensure_available_is_noop() -> None:
    """The base hook defaults to a no-op for backends with no binary gate."""
    _FakeBackend.ensure_available()  # must not raise


# ---------------------------------------------------------------------------
# Construction seam + factory wiring
# ---------------------------------------------------------------------------


def test_instance_defaults_to_tmux_backend(tmp_path: Path) -> None:
    """Direct construction (test paths) still gets a tmux backend, unchanged."""
    instance = terminal_mod.TerminalInstance(
        name="bash",
        session_key="s1",
        socket_path=tmp_path / "tmux.sock",
        private_dir=tmp_path,
    )
    assert instance.backend_name == "tmux"
    assert isinstance(instance._backend, TmuxBackend)


def test_construct_terminal_backend_unknown_name_raises(tmp_path: Path) -> None:
    """The construction seam rejects an unregistered name loudly."""
    with pytest.raises(RuntimeError, match="Unknown terminal backend"):
        terminal_mod._construct_terminal_backend(
            "nope", socket_path=tmp_path / "s.sock", target="main"
        )


def test_build_prompt_delivery_binds_advertised_herdr_pane(tmp_path: Path) -> None:
    """A herdr advertisement reconstructs delivery against its live pane id."""
    delivery = build_prompt_delivery(
        socket_path=tmp_path / "terminal.endpoint",
        target="w1:p2",
        backend_name="herdr",
    )

    assert isinstance(delivery.backend, HerdrBackend)
    assert delivery.backend._pane_id == "w1:p2"


def test_build_prompt_delivery_without_backend_remains_tmux(tmp_path: Path) -> None:
    """Legacy advertisements retain the byte-identical tmux default."""
    socket_path = tmp_path / "tmux.sock"
    delivery = build_prompt_delivery(socket_path=socket_path, target="main")

    assert isinstance(delivery.backend, TmuxBackend)
    assert delivery.backend._socket_path == socket_path
    assert delivery.backend._target == "main"


def test_construct_terminal_backend_registered_but_unwired_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered backend with no constructor branch raises NotImplementedError.

    The #11 hook: herdr registers, then adds its branch to
    ``_construct_terminal_backend``. Until a branch exists, construction fails
    loudly rather than silently falling back to tmux.
    """
    monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", _FakeBackend)
    with pytest.raises(NotImplementedError, match="constructor is not"):
        terminal_mod._construct_terminal_backend(
            "fake", socket_path=tmp_path / "s.sock", target="main"
        )


def test_factory_sets_backend_name_from_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The factory selects tmux on POSIX and stamps it on the instance.

    Runs the real factory with the platform forced to POSIX and tmux faked as
    present, so the selection path is exercised end-to-end and the instance's
    backend reflects the choice.
    """
    _force_posix(monkeypatch)
    monkeypatch.setattr(terminal_mod, "_tmux_available", lambda: True)
    spec = TerminalEnvSpec(
        command="bash",
        os_env=OSEnvSpec(type="caller_process", cwd=str(tmp_path)),
    )
    result = create_terminal_instance(name="bash", session_key="s1", spec=spec)
    try:
        assert result.instance.backend_name == "tmux"
        assert isinstance(result.instance._backend, TmuxBackend)
    finally:
        shutil.rmtree(result.instance.private_dir, ignore_errors=True)


def test_factory_propagates_explicit_spec_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``terminal_backend`` on the spec drives factory selection.

    A cross-platform fake backend is registered and named on the spec; the
    factory selects it over the platform default, proving precedence tier 1 is
    honored at the real construction point.
    """
    _force_posix(monkeypatch)
    monkeypatch.setitem(terminal_mod._TERMINAL_BACKENDS, "fake", _FakeBackend)
    # Wire a constructor branch for the fake so construction succeeds.
    real_construct = terminal_mod._construct_terminal_backend

    def _construct(name: str, *, socket_path: Path, target: str) -> TerminalBackend:
        if name == "fake":
            return _FakeBackend()
        return real_construct(name, socket_path=socket_path, target=target)

    monkeypatch.setattr(terminal_mod, "_construct_terminal_backend", _construct)
    spec = TerminalEnvSpec(
        command="bash",
        os_env=OSEnvSpec(type="caller_process", cwd=str(tmp_path)),
        terminal_backend="fake",
    )
    result = create_terminal_instance(name="bash", session_key="s1", spec=spec)
    try:
        assert result.instance.backend_name == "fake"
        assert isinstance(result.instance._backend, _FakeBackend)
    finally:
        shutil.rmtree(result.instance.private_dir, ignore_errors=True)
