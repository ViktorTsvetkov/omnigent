"""
Windows Job Object sandbox backend.

The Windows platform default. Unlike the Linux (``bwrap``) and macOS
(``seatbelt``) backends, which wrap the helper argv with a launcher that sets up
mount namespaces / SBPL filesystem isolation *before* exec, Windows has no
equivalent OS primitive for filesystem or network isolation. What Windows
*does* provide is the **Job Object**: a kernel container that owns a set of
processes, enforces resource limits, and — with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` — guarantees the whole process tree is
terminated when the last handle to the job closes.

So this backend trades a different set of guarantees than its POSIX siblings:

- **Provided**: process-tree containment (the helper and every descendant are
  killed together when Omnigent closes the job handle, including on an
  unexpected Omnigent crash, since the OS closes handles of dead processes) and
  optional CPU/memory ceilings.
- **Provided**: Low-integrity write confinement. ``write_paths`` and
  ``write_files`` receive reversible mandatory labels before launch.
- **NOT provided**: read confinement, network isolation, or syscall filtering.
  Unsupported requests fail before spawning instead of degrading silently.

A Job Object cannot prepend a launcher to argv the way ``bwrap`` does. The
backend therefore owns native process creation: it creates the child suspended,
assigns the Job, and resumes only after containment succeeds. ``post_spawn``
remains as a compatibility path for callers that already created a process.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

from .datamodel import OSEnvSandboxSpec, OSEnvSpec
from .sandbox import (
    ContainmentHandle,
    SandboxBackend,
    SandboxLaunchResult,
    SandboxPolicy,
    register_backend,
)

_LOGGER = logging.getLogger(__name__)

# Win32 constants (winnt.h). JobObjectExtendedLimitInformation is class 9.
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _JobHandle:
    """Owns a Windows Job Object handle; closing it kills the contained tree.

    Held by the parent for the helper's lifetime. Closing (explicitly or via
    ``with``) closes the kernel handle; with ``KILL_ON_JOB_CLOSE`` set, that
    terminates every still-running process in the job.
    """

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    def close(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        if not ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle)):
            _LOGGER.debug("windows_jobobject: CloseHandle failed for job handle")

    def __enter__(self) -> _JobHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def assign_process_handle_to_job(process_handle: int) -> _JobHandle:
    """Create a kill-on-close Job and assign an already-open process handle."""
    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error(), "CreateJobObjectW")
    try:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(job),
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error(), "SetInformationJobObject")
        if not kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(job), wintypes.HANDLE(process_handle)
        ):
            raise ctypes.WinError(ctypes.get_last_error(), "AssignProcessToJobObject")
        result = _JobHandle(int(job))
        job = None
        return result
    finally:
        if job:
            kernel32.CloseHandle(wintypes.HANDLE(job))


class WindowsJobObjectSandboxBackend(SandboxBackend):
    """Process-containment backend for Windows via Job Objects.

    See the module docstring for the guarantees this does and does not provide.
    """

    type_name = "windows_jobobject"
    isolation_tier = "windows_low_il"

    def resolve(self, spec: OSEnvSpec, cwd: Path) -> SandboxPolicy:
        """
        Build a :class:`SandboxPolicy` for the Job Object backend.

        The policy is marked ``active`` so the helper runs through the
        sandbox spawn path (private ``$TMPDIR``, ``start_in_scratch``
        support). The native launch owner enforces write roots at Low IL,
        assigns a Job while suspended, and only then resumes the helper.

        :param spec: The agent's :class:`OSEnvSpec`.
        :param cwd: Effective working directory of the helper.
        :returns: A populated :class:`SandboxPolicy`.
        :raises OSError: If the host is not Windows.
        """
        if os_name() != "nt":
            raise OSError(
                "windows_jobobject sandbox is only available on Windows. "
                "Configure os_env.sandbox.type='linux_bwrap' on Linux, "
                "'darwin_seatbelt' on macOS, or 'none' to disable sandboxing."
            )

        sandbox_spec = spec.sandbox or OSEnvSandboxSpec(type=self.type_name)
        read_roots: list[Path] | None = None
        if sandbox_spec.read_paths is not None:
            read_roots = [(cwd / r).resolve() for r in sandbox_spec.read_paths]
        write_roots = [(cwd / w).resolve() for w in (sandbox_spec.write_paths or [])]
        write_files = [(cwd / f).resolve() for f in (sandbox_spec.write_files or [])]

        policy = SandboxPolicy(
            backend_type=self.type_name,
            active=True,
            read_roots=read_roots,
            write_roots=write_roots,
            write_files=write_files,
            allow_network=sandbox_spec.allow_network,
            env_passthrough=(
                list(sandbox_spec.env_passthrough)
                if sandbox_spec.env_passthrough is not None
                else None
            ),
            credential_proxy=sandbox_spec.credential_proxy,
        )
        if policy.read_roots is not None:
            raise ValueError(
                "windows_low_il cannot enforce read_paths; AppContainer/C3 is required "
                "for read confinement and is not implemented yet"
            )
        if not policy.allow_network:
            raise ValueError(
                "windows_low_il cannot enforce allow_network=false; AppContainer/C3 is "
                "required for network denial and is not implemented yet"
            )
        _LOGGER.debug("windows sandbox resolved tier=%s", self.isolation_tier)
        return policy

    def wrap_launcher_argv(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        cwd: Path,
        chdir: Path | None = None,
        target: str | None = None,
    ) -> list[str]:
        """Return argv unchanged; the native Windows backend owns creation."""
        del policy, cwd, chdir, target
        return argv

    def launch_windows(
        self,
        argv: list[str],
        policy: SandboxPolicy,
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: int,
        stdout: int,
        stderr: int,
        text: bool,
        bufsize: int,
    ) -> SandboxLaunchResult:
        """Launch Low-IL suspended, assign its Job, and only then resume."""
        from .windows_sandbox_process import launch_low_integrity_process

        return launch_low_integrity_process(
            argv,
            policy,
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=text,
            bufsize=bufsize,
        )

    def activate(self, policy: SandboxPolicy) -> None:
        """No-op: containment is applied by :meth:`post_spawn` from the parent.

        A Job Object is assigned to a process from outside it, so there is
        nothing for the in-helper ``activate`` step to do.
        """
        del policy

    def post_spawn(self, policy: SandboxPolicy, pid: int) -> ContainmentHandle | None:
        """Compatibility path for callers that already created the child."""
        del policy
        kernel32 = ctypes.windll.kernel32
        proc_handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, wintypes.DWORD(pid)
        )
        if not proc_handle:
            _LOGGER.warning(
                "windows_jobobject: OpenProcess(pid=%d) failed (err=%d); continuing uncontained.",
                pid,
                ctypes.get_last_error(),
            )
            return None
        try:
            try:
                return assign_process_handle_to_job(int(proc_handle))
            except OSError as exc:
                _LOGGER.warning(
                    "windows_jobobject: job assignment for pid %d failed (%s); "
                    "continuing without Job Object containment.",
                    pid,
                    exc,
                )
                return None
        finally:
            kernel32.CloseHandle(wintypes.HANDLE(proc_handle))


def os_name() -> str:
    """Indirection so tests can monkeypatch the platform check."""
    import os

    return os.name


# Configure ctypes prototypes once at import. Setting argtypes/restype keeps the
# HANDLE/DWORD widths correct on 64-bit Python (pointers must not be truncated
# to 32-bit ints).
def _configure_prototypes() -> None:
    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]


if hasattr(ctypes, "windll"):
    _configure_prototypes()

register_backend(WindowsJobObjectSandboxBackend())
