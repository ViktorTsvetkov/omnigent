r"""Spawn-time owner for suspended, Low-IL Windows sandbox processes."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import io
import msvcrt
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol, TextIO, cast

if os.name != "nt":
    raise ImportError("windows_sandbox_process is only available on Windows")

from .sandbox import SandboxLaunchResult, SandboxPolicy
from .windows_jobobject_sandbox import assign_process_handle_to_job
from .windows_security import (
    WindowsLaunchSecurity,
    prepare_appcontainer_launch,
    prepare_low_integrity_launch,
)

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
STARTF_USESTDHANDLES = 0x00000100
HANDLE_FLAG_INHERIT = 0x00000001
STILL_ACTIVE = 259
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
INFINITE = 0xFFFFFFFF


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


def quote_windows_argv(argv: list[str]) -> str:
    """Encode argv using CPython's Windows command-line quoting rules."""
    if not argv:
        raise ValueError("argv must not be empty")
    return subprocess.list2cmdline(argv)


def build_environment_block(env: Mapping[str, str]) -> str:
    """Return a sorted, double-NUL-terminated Unicode environment block."""
    entries = sorted(env.items(), key=lambda item: item[0].upper())
    return "\0".join(f"{key}={value}" for key, value in entries) + "\0\0"


@dataclass
class _SharedHandle:
    value: int | None

    def close(self) -> None:
        if self.value is None:
            return
        value, self.value = self.value, None
        if not kernel32.CloseHandle(wintypes.HANDLE(value)):
            raise ctypes.WinError(ctypes.get_last_error(), "CloseHandle")


class _WindowsProcessHandle:
    """Popen-compatible subset backed by a native process handle."""

    def __init__(
        self,
        pid: int,
        process: _SharedHandle,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO,
        argv: list[str],
    ) -> None:
        self.pid = pid
        self._process = process
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.args = argv
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._process.value is None:
            return self.returncode
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(
            wintypes.HANDLE(self._process.value), ctypes.byref(code)
        ):
            raise ctypes.WinError(ctypes.get_last_error(), "GetExitCodeProcess")
        if code.value != STILL_ACTIVE:
            self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is not None:
            return self.returncode
        if self._process.value is None:
            raise RuntimeError("process handle is closed")
        milliseconds = (
            INFINITE if timeout is None else max(0, min(round(timeout * 1000), 0xFFFFFFFE))
        )
        result = kernel32.WaitForSingleObject(wintypes.HANDLE(self._process.value), milliseconds)
        if result == WAIT_TIMEOUT:
            assert timeout is not None
            raise subprocess.TimeoutExpired(self.args, timeout)
        if result != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error(), "WaitForSingleObject")
        code = self.poll()
        assert code is not None
        return code

    def terminate(self) -> None:
        self._terminate(1)

    def kill(self) -> None:
        self._terminate(1)

    def _terminate(self, code: int) -> None:
        if self.poll() is not None:
            return
        assert self._process.value is not None
        if not kernel32.TerminateProcess(wintypes.HANDLE(self._process.value), code):
            raise ctypes.WinError(ctypes.get_last_error(), "TerminateProcess")


class _Closeable(Protocol):
    def close(self) -> None: ...


@dataclass
class _WindowsContainment:
    job: _Closeable
    security: WindowsLaunchSecurity
    process: _SharedHandle
    thread: _SharedHandle

    def close(self) -> None:
        errors: list[BaseException] = []
        try:
            self.job.close()
        except OSError as exc:
            errors.append(exc)
        if self.process.value is not None:
            kernel32.WaitForSingleObject(wintypes.HANDLE(self.process.value), 10_000)
        for owner in (self.thread, self.process, self.security):
            try:
                owner.close()
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> _WindowsContainment:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _pipe(
    *,
    parent_reads: bool,
    text: bool,
    bufsize: int,
    appcontainer_sid: wintypes.LPVOID | None = None,
) -> tuple[TextIO | BinaryIO, int]:
    read_handle = wintypes.HANDLE()
    write_handle = wintypes.HANDLE()
    descriptor = wintypes.LPVOID()
    sid_text = wintypes.LPWSTR()
    try:
        if appcontainer_sid:
            if not advapi32.ConvertSidToStringSidW(appcontainer_sid, ctypes.byref(sid_text)):
                raise ctypes.WinError(ctypes.get_last_error(), "ConvertSidToStringSidW")
            sddl = f"D:(A;;GA;;;SY)(A;;GA;;;OW)(A;;GRGW;;;{sid_text.value})"
            if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, ctypes.byref(descriptor), None
            ):
                raise ctypes.WinError(
                    ctypes.get_last_error(),
                    "ConvertStringSecurityDescriptorToSecurityDescriptorW(pipe)",
                )
        attributes = _SECURITY_ATTRIBUTES(
            ctypes.sizeof(_SECURITY_ATTRIBUTES), descriptor or None, True
        )
        if not kernel32.CreatePipe(
            ctypes.byref(read_handle), ctypes.byref(write_handle), ctypes.byref(attributes), 0
        ):
            raise ctypes.WinError(ctypes.get_last_error(), "CreatePipe")
    finally:
        if sid_text:
            kernel32.LocalFree(sid_text)
        if descriptor:
            kernel32.LocalFree(descriptor)
    parent_handle = read_handle if parent_reads else write_handle
    child_handle = write_handle if parent_reads else read_handle
    try:
        if not kernel32.SetHandleInformation(parent_handle, HANDLE_FLAG_INHERIT, 0):
            raise ctypes.WinError(ctypes.get_last_error(), "SetHandleInformation")
        flags = os.O_RDONLY if parent_reads else os.O_WRONLY
        assert parent_handle.value is not None
        fd = msvcrt.open_osfhandle(int(parent_handle.value), flags | os.O_BINARY)
        parent_handle = wintypes.HANDLE()
        raw: BinaryIO = os.fdopen(fd, "rb" if parent_reads else "wb", buffering=0)
        if text:
            stream: TextIO | BinaryIO = io.TextIOWrapper(
                raw,
                encoding="utf-8",
                errors="strict",
                line_buffering=bufsize == 1,
                write_through=bufsize == 0,
            )
        else:
            stream = raw
        assert child_handle.value is not None
        return stream, int(child_handle.value)
    except BaseException:
        if parent_handle:
            kernel32.CloseHandle(parent_handle)
        kernel32.CloseHandle(child_handle)
        raise


def launch_windows_sandbox_process(
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
    """Create suspended, assign to a Job, then resume; all failures are closed."""
    if (stdin, stdout, stderr) != (subprocess.PIPE, subprocess.PIPE, subprocess.PIPE):
        raise ValueError("Windows sandbox launch currently requires redirected PIPE stdio")
    if not text:
        raise ValueError("Windows sandbox helper launch requires text=True")
    security = (
        prepare_appcontainer_launch(policy, executable=argv[0])
        if policy.backend_type == "windows_appcontainer"
        else prepare_low_integrity_launch(policy)
    )
    streams: list[TextIO | BinaryIO] = []
    child_handles: list[int] = []
    process = _SharedHandle(None)
    thread = _SharedHandle(None)
    job = None
    try:
        parent_stdin, child_stdin = _pipe(
            parent_reads=False,
            text=text,
            bufsize=bufsize,
            appcontainer_sid=security.appcontainer_sid,
        )
        streams.append(parent_stdin)
        child_handles.append(child_stdin)
        parent_stdout, child_stdout = _pipe(
            parent_reads=True,
            text=text,
            bufsize=bufsize,
            appcontainer_sid=security.appcontainer_sid,
        )
        streams.append(parent_stdout)
        child_handles.append(child_stdout)
        parent_stderr, child_stderr = _pipe(
            parent_reads=True,
            text=text,
            bufsize=bufsize,
            appcontainer_sid=security.appcontainer_sid,
        )
        streams.append(parent_stderr)
        child_handles.append(child_stderr)
        startup = _STARTUPINFOEXW()
        startup.StartupInfo = _STARTUPINFOW(
            cb=ctypes.sizeof(_STARTUPINFOEXW),
            dwFlags=STARTF_USESTDHANDLES,
            hStdInput=wintypes.HANDLE(child_stdin),
            hStdOutput=wintypes.HANDLE(child_stdout),
            hStdError=wintypes.HANDLE(child_stderr),
        )
        attribute_size = ctypes.c_size_t()
        attribute_count = 2 if security.security_capabilities is not None else 1
        kernel32.InitializeProcThreadAttributeList(
            None, attribute_count, 0, ctypes.byref(attribute_size)
        )
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        startup.lpAttributeList = ctypes.cast(attribute_buffer, wintypes.LPVOID)
        if not kernel32.InitializeProcThreadAttributeList(
            startup.lpAttributeList, attribute_count, 0, ctypes.byref(attribute_size)
        ):
            raise ctypes.WinError(ctypes.get_last_error(), "InitializeProcThreadAttributeList")
        inherited_handles = (wintypes.HANDLE * 3)(child_stdin, child_stdout, child_stderr)
        if not kernel32.UpdateProcThreadAttribute(
            startup.lpAttributeList,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inherited_handles,
            ctypes.sizeof(inherited_handles),
            None,
            None,
        ):
            kernel32.DeleteProcThreadAttributeList(startup.lpAttributeList)
            raise ctypes.WinError(
                ctypes.get_last_error(), "UpdateProcThreadAttribute(handle list)"
            )
        if security.security_capabilities is not None and not kernel32.UpdateProcThreadAttribute(
            startup.lpAttributeList,
            0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
            ctypes.byref(security.security_capabilities),
            ctypes.sizeof(security.security_capabilities),
            None,
            None,
        ):
            kernel32.DeleteProcThreadAttributeList(startup.lpAttributeList)
            raise ctypes.WinError(
                ctypes.get_last_error(), "UpdateProcThreadAttribute(security capabilities)"
            )
        info = _PROCESS_INFORMATION()
        command = ctypes.create_unicode_buffer(quote_windows_argv(argv))
        launch_env = dict(env)
        if policy.write_roots:
            launch_env["TEMP"] = os.fspath(policy.write_roots[0])
            launch_env["TMP"] = os.fspath(policy.write_roots[0])
        environment = ctypes.create_unicode_buffer(build_environment_block(launch_env))
        try:
            flags = CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT
            if security.token is not None:
                assert security.token.value is not None
                created = advapi32.CreateProcessAsUserW(
                    wintypes.HANDLE(security.token.value), None, command, None, None, True,
                    flags, environment, os.fspath(cwd), ctypes.byref(startup.StartupInfo),
                    ctypes.byref(info),
                )
                api = "CreateProcessAsUserW"
            else:
                created = kernel32.CreateProcessW(
                    None, command, None, None, True, flags, environment, os.fspath(cwd),
                    ctypes.byref(startup.StartupInfo), ctypes.byref(info),
                )
                api = "CreateProcessW"
            if not created:
                raise ctypes.WinError(ctypes.get_last_error(), api)
        finally:
            kernel32.DeleteProcThreadAttributeList(startup.lpAttributeList)
        process.value = int(info.hProcess)
        thread.value = int(info.hThread)
        for handle in child_handles:
            kernel32.CloseHandle(wintypes.HANDLE(handle))
        child_handles.clear()
        job = assign_process_handle_to_job(process.value)
        if kernel32.ResumeThread(wintypes.HANDLE(thread.value)) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error(), "ResumeThread")
        proc = _WindowsProcessHandle(
            int(info.dwProcessId),
            process,
            cast(TextIO, parent_stdin),
            cast(TextIO, parent_stdout),
            cast(TextIO, parent_stderr),
            argv,
        )
        containment = _WindowsContainment(job, security, process, thread)
        return SandboxLaunchResult(proc, containment)
    except BaseException:
        if process.value is not None:
            kernel32.TerminateProcess(wintypes.HANDLE(process.value), 1)
        if job is not None:
            job.close()
        if process.value is not None:
            kernel32.WaitForSingleObject(wintypes.HANDLE(process.value), 10_000)
        thread.close()
        process.close()
        for stream in streams:
            stream.close()
        for handle in child_handles:
            kernel32.CloseHandle(wintypes.HANDLE(handle))
        security.close()
        raise


launch_low_integrity_process = launch_windows_sandbox_process


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
kernel32.CreatePipe.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(_SECURITY_ATTRIBUTES),
    wintypes.DWORD,
]
kernel32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
kernel32.ResumeThread.restype = wintypes.DWORD
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
kernel32.InitializeProcThreadAttributeList.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.UpdateProcThreadAttribute.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.c_size_t,
    wintypes.LPVOID,
    ctypes.c_size_t,
    wintypes.LPVOID,
    wintypes.LPVOID,
]
kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
kernel32.CreateProcessW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.BOOL,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPCWSTR,
    ctypes.POINTER(_STARTUPINFOW),
    ctypes.POINTER(_PROCESS_INFORMATION),
]
advapi32.CreateProcessAsUserW.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.BOOL,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPCWSTR,
    ctypes.POINTER(_STARTUPINFOW),
    ctypes.POINTER(_PROCESS_INFORMATION),
]
advapi32.ConvertSidToStringSidW.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.LPWSTR),
]
advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.DWORD),
]
