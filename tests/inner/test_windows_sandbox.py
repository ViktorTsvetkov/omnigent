from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import json
import msvcrt
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from omnigent._platform import IS_WINDOWS

pytestmark = pytest.mark.skipif(not IS_WINDOWS, reason="native Windows sandbox tests")


@pytest.fixture
def policy(tmp_path: Path):
    from omnigent.inner.sandbox import SandboxPolicy

    return SandboxPolicy(
        backend_type="windows_jobobject",
        active=True,
        read_roots=None,
        write_roots=[tmp_path],
        write_files=[],
        allow_network=True,
    )


def _launch(policy, argv: list[str], cwd: Path | None = None, env=None):
    from omnigent.inner.windows_sandbox_process import launch_low_integrity_process

    return launch_low_integrity_process(
        argv,
        policy,
        cwd=cwd or policy.write_roots[0],
        env=env or os.environ,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _finish(result) -> tuple[int, str, str]:
    result.process.stdin.close()
    stdout = result.process.stdout.read()
    stderr = result.process.stderr.read()
    code = result.process.wait(timeout=15)
    result.containment.close()
    return code, stdout, stderr


def _label_sddl(path: Path) -> str:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    descriptor = wintypes.LPVOID()
    error = advapi32.GetNamedSecurityInfoW(
        str(path), 1, 0x10, None, None, None, None, ctypes.byref(descriptor)
    )
    if error:
        raise ctypes.WinError(error)
    text = wintypes.LPWSTR()
    try:
        if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, 1, 0x10, ctypes.byref(text), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return text.value or ""
    finally:
        if text:
            kernel32.LocalFree(text)
        kernel32.LocalFree(descriptor)


def test_windows_argv_quoting_matches_subprocess() -> None:
    from omnigent.inner.windows_sandbox_process import quote_windows_argv

    argv = [r"C:\Program Files\Python\python.exe", "space here", 'quote"here', "é雪", "tail\\"]
    assert quote_windows_argv(argv) == subprocess.list2cmdline(argv)


def test_unicode_environment_block_is_sorted_and_double_nul() -> None:
    from omnigent.inner.windows_sandbox_process import build_environment_block

    block = build_environment_block({"z": "last", "Alpha": "é雪", "b": "two"})
    assert block == "Alpha=é雪\0b=two\0z=last\0\0"


def test_read_roots_and_network_deny_fail_before_spawn(tmp_path: Path) -> None:
    from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
    from omnigent.inner.windows_jobobject_sandbox import WindowsJobObjectSandboxBackend

    backend = WindowsJobObjectSandboxBackend()
    with pytest.raises(ValueError, match=r"AppContainer/C3.*not implemented"):
        backend.resolve(OSEnvSpec(sandbox=OSEnvSandboxSpec(read_paths=["read"])), tmp_path)
    with pytest.raises(ValueError, match=r"network denial.*not implemented"):
        backend.resolve(
            OSEnvSpec(sandbox=OSEnvSandboxSpec(read_paths=None, allow_network=False)), tmp_path
        )


def test_native_launch_owner_keeps_launcher_argv_unwrapped(policy, tmp_path: Path) -> None:
    from omnigent.inner.windows_jobobject_sandbox import WindowsJobObjectSandboxBackend

    argv = [sys.executable, "-c", "pass"]
    assert WindowsJobObjectSandboxBackend().wrap_launcher_argv(argv, policy, tmp_path) is argv


def test_child_is_suspended_until_job_assignment(policy, tmp_path: Path, monkeypatch) -> None:
    import omnigent.inner.windows_sandbox_process as process_module
    from omnigent.inner.windows_jobobject_sandbox import assign_process_handle_to_job

    sentinel = tmp_path / "sentinel.txt"

    def checked_assign(handle: int):
        assert not sentinel.exists()
        return assign_process_handle_to_job(handle)

    monkeypatch.setattr(process_module, "assign_process_handle_to_job", checked_assign)
    result = _launch(
        policy,
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ran')",
            str(sentinel),
        ],
    )
    assert _finish(result)[0] == 0
    assert sentinel.read_text() == "ran"


def test_only_redirected_stdio_handles_are_inherited(policy) -> None:
    read_fd, write_fd = os.pipe()
    leaked = msvcrt.get_osfhandle(read_fd)
    os.set_handle_inheritable(leaked, True)
    try:
        code = (
            "import ctypes,ctypes.wintypes as w,sys; flags=w.DWORD(); "
            "ok=ctypes.windll.kernel32.GetHandleInformation(w.HANDLE(int(sys.argv[1])), "
            "ctypes.byref(flags)); print(int(bool(ok)))"
        )
        result = _launch(policy, [sys.executable, "-c", code, str(leaked)])
        returncode, stdout, stderr = _finish(result)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert (returncode, stdout.strip(), stderr) == (0, "0", "")


def test_stdio_exit_code_terminate_and_unicode_launch(policy, tmp_path: Path) -> None:
    cwd = tmp_path / "space é雪"
    cwd.mkdir()
    before = _label_sddl(cwd)
    policy.write_roots = [cwd]
    env = dict(os.environ)
    env["OMNI_UNICODE"] = "value é雪"
    result = _launch(
        policy,
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            "import os,sys; print(sys.argv[1]); "
            "print(os.environ['OMNI_UNICODE'], file=sys.stderr); "
            "print(input()); sys.exit(7)",
            "arg é雪 space",
        ],
        cwd,
        env,
    )
    result.process.stdin.write("input line\n")
    result.process.stdin.flush()
    result.process.stdin.close()
    captured_stdout = result.process.stdout.read()
    captured_stderr = result.process.stderr.read()
    returncode = result.process.wait(15)
    assert (returncode, captured_stdout.splitlines(), captured_stderr.strip()) == (
        7,
        ["arg é雪 space", "input line"],
        "value é雪",
    )
    result.containment.close()
    restored = _label_sddl(cwd)
    assert "LW" not in restored
    if before:
        assert restored == before

    sleeper = _launch(policy, [sys.executable, "-c", "import time; time.sleep(60)"], cwd)
    sleeper.process.terminate()
    assert sleeper.process.wait(15) == 1
    sleeper.containment.close()
    assert "LW" not in _label_sddl(cwd)


def test_low_il_write_jail_read_semantics_and_label_restore(tmp_path: Path) -> None:
    from omnigent.inner.sandbox import SandboxPolicy

    allowed = tmp_path / "allowed"
    forbidden = tmp_path / "forbidden"
    allowed.mkdir()
    forbidden.mkdir()
    exact = tmp_path / "exact.txt"
    exact.write_text("old")
    seed = forbidden / "seed.txt"
    seed.write_text("readable")
    before_allowed = _label_sddl(allowed)
    before_exact = _label_sddl(exact)
    policy = SandboxPolicy("windows_jobobject", True, None, [allowed], [exact], True)
    script = r"""
import json, pathlib, sys
allowed, exact, forbidden, seed, profile = map(pathlib.Path, sys.argv[1:])
out = {}
def attempt(name, fn):
    try: fn(); out[name] = 'allowed'
    except OSError as exc: out[name] = f'denied:{exc.winerror or exc.errno}'
attempt('create_allowed', lambda: (allowed/'new.txt').write_text('ok'))
attempt('delete_allowed', lambda: (allowed/'new.txt').unlink())
attempt('exact_file', lambda: exact.write_text('updated'))
attempt('outside_write', lambda: (forbidden/'bad.txt').write_text('bad'))
attempt('profile_write', lambda: profile.write_text('bad'))
attempt('outside_read', lambda: seed.read_text())
print(json.dumps(out))
"""
    profile_target = Path.home() / f"omnigent-low-il-forbidden-{os.getpid()}.txt"
    result = _launch(
        policy,
        [
            sys.executable,
            "-c",
            script,
            str(allowed),
            str(exact),
            str(forbidden),
            str(seed),
            str(profile_target),
        ],
        allowed,
    )
    code, stdout, stderr = _finish(result)
    assert (code, stderr) == (0, "")
    evidence = json.loads(stdout)
    assert evidence["create_allowed"] == "allowed"
    assert evidence["delete_allowed"] == "allowed"
    assert evidence["exact_file"] == "allowed"
    assert evidence["outside_write"].startswith("denied:")
    assert evidence["profile_write"].startswith("denied:")
    assert evidence["outside_read"] == "allowed"
    assert exact.read_text() == "updated"
    assert not profile_target.exists()
    restored_allowed = _label_sddl(allowed)
    restored_exact = _label_sddl(exact)
    assert "LW" not in restored_allowed
    assert "LW" not in restored_exact
    if before_allowed:
        assert restored_allowed == before_allowed
    if before_exact:
        assert restored_exact == before_exact


def test_overlapping_label_leases_restore_only_after_last_close(tmp_path: Path) -> None:
    from omnigent.inner.windows_security import grant_low_integrity_write_root

    before = _label_sddl(tmp_path)
    first = grant_low_integrity_write_root(tmp_path)
    second = grant_low_integrity_write_root(tmp_path)
    assert "LW" in _label_sddl(tmp_path)
    first.close()
    assert "LW" in _label_sddl(tmp_path)
    second.close()
    restored = _label_sddl(tmp_path)
    assert "LW" not in restored
    if before:
        assert restored == before


def test_known_ambient_low_integrity_holes_are_explicit(policy, tmp_path: Path) -> None:
    candidates = [Path.home() / "AppData" / "LocalLow", Path(os.environ["TEMP"]) / "Low"]
    existing = [path for path in candidates if path.is_dir()]
    result = _launch(
        policy,
        [
            sys.executable,
            "-c",
            "import json,pathlib,sys; out={}; "
            'exec("for p in map(pathlib.Path,sys.argv[1:]):\\n'
            " q=p/'omnigent-low-il-hole.txt'\\n"
            " try:q.write_text('hole');q.unlink();out[str(p)]='allowed'\\n"
            " except OSError as e:out[str(p)]=f'denied:{e.winerror or e.errno}'\"); "
            "print(json.dumps(out))",
            *map(str, existing),
        ],
    )
    code, stdout, _ = _finish(result)
    assert code == 0
    observed = json.loads(stdout)
    assert set(observed) == {str(path) for path in existing}
    assert set(observed.values()) <= {"allowed"}


def test_job_close_kills_child_and_grandchild(policy, tmp_path: Path) -> None:
    import psutil

    pids = tmp_path / "pids.json"
    code = (
        "import json,os,pathlib,subprocess,sys,time; "
        "g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),g.pid])); "
        "time.sleep(60)"
    )
    result = _launch(policy, [sys.executable, "-c", code, str(pids)])
    deadline = time.monotonic() + 10
    while not pids.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid, grandchild_pid = json.loads(pids.read_text())
    result.containment.close()
    deadline = time.monotonic() + 10
    while (
        any(psutil.pid_exists(pid) for pid in (child_pid, grandchild_pid))
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    assert not psutil.pid_exists(child_pid)
    assert not psutil.pid_exists(grandchild_pid)


def test_handle_count_returns_to_baseline_after_normal_and_failed_launch(
    policy, monkeypatch
) -> None:
    import omnigent.inner.windows_sandbox_process as process_module

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    count = wintypes.DWORD()
    assert kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count))
    baseline = count.value
    result = _launch(policy, [sys.executable, "-c", "pass"])
    assert _finish(result)[0] == 0

    def fail_assignment(_handle: int):
        raise OSError("injected job failure")

    monkeypatch.setattr(process_module, "assign_process_handle_to_job", fail_assignment)
    with pytest.raises(OSError, match="injected job failure"):
        _launch(policy, [sys.executable, "-c", "pass"])
    assert kernel32.GetProcessHandleCount(kernel32.GetCurrentProcess(), ctypes.byref(count))
    assert count.value <= baseline + 2


def test_existing_appcontainer_profile_is_deleted_on_close() -> None:
    """C3 primitives remain covered even though C3 integration is out of scope."""
    from omnigent.inner.windows_sandbox import create_appcontainer_profile

    name = f"omnigent-test-{uuid.uuid4().hex}"
    profile = create_appcontainer_profile(name)
    assert profile.sid
    derived = create_appcontainer_profile(name)
    assert not derived.created
    derived.close()
    profile.close()

    recreated = create_appcontainer_profile(name)
    try:
        assert recreated.created
    finally:
        recreated.close()


def test_existing_appcontainer_process_can_use_granted_root(tmp_path: Path) -> None:
    """The C1+C2 launch changes do not regress existing C3 primitives."""
    from omnigent.inner.windows_sandbox import (
        cleanup_appcontainer,
        create_appcontainer_profile,
        create_process_appcontainer,
        wait_process,
    )

    profile = create_appcontainer_profile(f"omnigent-test-{uuid.uuid4().hex}")
    grants = []
    output = tmp_path / "appcontainer.txt"
    try:
        _, process, grants = create_process_appcontainer(
            profile.sid,
            [
                sys._base_executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('ok')",
                str(output),
            ],
            tmp_path,
            os.environ,
            [],
            [tmp_path],
        )
        assert wait_process(process) == 0
    finally:
        cleanup_appcontainer(profile, grants)
    assert output.read_text() == "ok"
