from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.windows_only


@pytest.fixture
def policy():
    from omnigent.inner.sandbox import SandboxPolicy

    return SandboxPolicy(
        backend_type="windows_jobobject",
        active=True,
        read_roots=None,
        write_roots=[],
        write_files=[],
        allow_network=True,
    )


def test_build_low_il_token_returns_closeable_handle() -> None:
    from omnigent.inner.windows_sandbox import build_low_il_token

    token = build_low_il_token()
    assert token
    assert ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(token))


def test_label_write_root_allows_low_il_child_write(tmp_path: Path) -> None:
    from omnigent.inner.windows_sandbox import (
        build_low_il_token,
        create_process_low_il,
        label_write_root_low,
        wait_process,
    )

    label_write_root_low(tmp_path)
    output = tmp_path / "written-by-low-il.txt"
    token = build_low_il_token()
    try:
        _, process = create_process_low_il(
            token,
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')",
                str(output),
            ],
            tmp_path,
            os.environ,
        )
        assert wait_process(process) == 0
    finally:
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(token))
    assert output.read_text() == "ok"


def test_wrap_launcher_argv_is_noop_without_write_grants(policy, tmp_path: Path) -> None:
    from omnigent.inner.windows_jobobject_sandbox import WindowsJobObjectSandboxBackend

    argv = [sys.executable, "-c", "pass"]
    assert WindowsJobObjectSandboxBackend().wrap_launcher_argv(argv, policy, tmp_path) is argv


def test_wrap_launcher_argv_uses_low_il_wrapper_with_write_grant(policy, tmp_path: Path) -> None:
    from omnigent.inner.windows_jobobject_sandbox import WindowsJobObjectSandboxBackend
    from omnigent.inner.windows_sandbox_launch import decode_policy

    policy.write_roots = [tmp_path]
    command = [sys.executable, "-c", "pass"]
    wrapped = WindowsJobObjectSandboxBackend().wrap_launcher_argv(command, policy, tmp_path)

    assert wrapped[:3] == [sys.executable, "-m", "omnigent.inner.windows_sandbox_launch"]
    decoded, cwd = decode_policy(wrapped[3])
    assert decoded.write_roots == [tmp_path]
    assert cwd == tmp_path
    assert wrapped[4:] == ["--", *command]
