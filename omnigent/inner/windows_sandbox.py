"""Low-integrity Windows process-launch primitives.

Low IL provides a write jail through mandatory ``NO_WRITE_UP``; it does not
confine reads. Known Low-IL locations such as
``%USERPROFILE%\AppData\LocalLow`` and ``%TEMP%\Low`` remain writable.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import subprocess
from collections.abc import Mapping, Sequence

if os.name != "nt":
    raise ImportError("windows_sandbox is only available on Windows")


TOKEN_ALL_ACCESS = 0x000F01FF
DISABLE_MAX_PRIVILEGE = 0x1
TOKEN_INTEGRITY_LEVEL = 25
SE_GROUP_INTEGRITY = 0x20
SDDL_REVISION_1 = 1
SE_FILE_OBJECT = 1
LABEL_SECURITY_INFORMATION = 0x10
CREATE_UNICODE_ENVIRONMENT = 0x400
INFINITE = 0xFFFFFFFF


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


class STARTUPINFOW(ctypes.Structure):
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


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


def _raise_last_error(name: str, error: int | None = None) -> None:
    raise ctypes.WinError(error if error is not None else ctypes.get_last_error(), name)


def build_low_il_token() -> int:
    """Build a low-IL restricted derivative of this process's token."""
    token = wintypes.HANDLE()
    restricted = wintypes.HANDLE()
    low_sid = wintypes.LPVOID()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), TOKEN_ALL_ACCESS, ctypes.byref(token)
    ):
        _raise_last_error("OpenProcessToken")
    try:
        if not advapi32.CreateRestrictedToken(
            token,
            DISABLE_MAX_PRIVILEGE,
            0,
            None,
            0,
            None,
            0,
            None,
            ctypes.byref(restricted),
        ):
            _raise_last_error("CreateRestrictedToken")
        if not advapi32.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(low_sid)):
            _raise_last_error("ConvertStringSidToSidW")
        label = TOKEN_MANDATORY_LABEL(SID_AND_ATTRIBUTES(low_sid, SE_GROUP_INTEGRITY))
        sid_length = advapi32.GetLengthSid(low_sid)
        if not advapi32.SetTokenInformation(
            restricted,
            TOKEN_INTEGRITY_LEVEL,
            ctypes.byref(label),
            ctypes.sizeof(TOKEN_MANDATORY_LABEL) + sid_length,
        ):
            _raise_last_error("SetTokenInformation")
        result = int(restricted.value)
        restricted = wintypes.HANDLE()
        return result
    finally:
        if low_sid:
            kernel32.LocalFree(low_sid)
        if restricted:
            kernel32.CloseHandle(restricted)
        kernel32.CloseHandle(token)


def label_write_root_low(path: os.PathLike[str] | str) -> None:
    """Apply an inheritable low mandatory label to a writable path."""
    security_descriptor = wintypes.LPVOID()
    sacl_present = wintypes.BOOL()
    sacl_defaulted = wintypes.BOOL()
    sacl = wintypes.LPVOID()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        "S:(ML;OICI;NW;;;LW)", SDDL_REVISION_1, ctypes.byref(security_descriptor), None
    ):
        _raise_last_error("ConvertStringSecurityDescriptorToSecurityDescriptorW")
    try:
        if not advapi32.GetSecurityDescriptorSacl(
            security_descriptor,
            ctypes.byref(sacl_present),
            ctypes.byref(sacl),
            ctypes.byref(sacl_defaulted),
        ):
            _raise_last_error("GetSecurityDescriptorSacl")
        error = advapi32.SetNamedSecurityInfoW(
            os.fspath(path), SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION, None, None, None, sacl
        )
        if error:
            _raise_last_error("SetNamedSecurityInfoW", error)
    finally:
        kernel32.LocalFree(security_descriptor)


def _environment_block(env: Mapping[str, str]) -> ctypes.Array[ctypes.c_wchar]:
    entries = (
        f"{key}={value}" for key, value in sorted(env.items(), key=lambda item: item[0].upper())
    )
    return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")


def create_process_low_il(
    token: int,
    argv: Sequence[str],
    cwd: os.PathLike[str] | str | None,
    env: Mapping[str, str],
) -> tuple[int, int]:
    """Launch argv with a low-IL token and inherited standard handles."""
    if not argv:
        raise ValueError("argv must not be empty")
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
    environment = _environment_block(env)
    startup = STARTUPINFOW(cb=ctypes.sizeof(STARTUPINFOW))
    process_info = PROCESS_INFORMATION()
    if not advapi32.CreateProcessAsUserW(
        wintypes.HANDLE(token),
        None,
        command_line,
        None,
        None,
        True,
        CREATE_UNICODE_ENVIRONMENT,
        environment,
        os.fspath(cwd) if cwd is not None else None,
        ctypes.byref(startup),
        ctypes.byref(process_info),
    ):
        _raise_last_error("CreateProcessAsUserW")
    kernel32.CloseHandle(process_info.hThread)
    return int(process_info.dwProcessId), int(process_info.hProcess)


def wait_process(process_handle: int) -> int:
    """Wait for a process handle, close it, and return its exit code."""
    handle = wintypes.HANDLE(process_handle)
    try:
        result = kernel32.WaitForSingleObject(handle, INFINITE)
        if result != 0:
            _raise_last_error("WaitForSingleObject")
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            _raise_last_error("GetExitCodeProcess")
        return int(exit_code.value)
    finally:
        kernel32.CloseHandle(handle)


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.argtypes = []
kernel32.LocalFree.restype = wintypes.HANDLE
kernel32.LocalFree.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.GetExitCodeProcess.restype = wintypes.BOOL
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
advapi32.CreateRestrictedToken.restype = wintypes.BOOL
advapi32.CreateRestrictedToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.HANDLE),
]
advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPVOID)]
advapi32.GetLengthSid.restype = wintypes.DWORD
advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
advapi32.SetTokenInformation.restype = wintypes.BOOL
advapi32.SetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.DWORD),
]
advapi32.GetSecurityDescriptorSacl.restype = wintypes.BOOL
advapi32.GetSecurityDescriptorSacl.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.BOOL),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.BOOL),
]
advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
advapi32.SetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
]
advapi32.CreateProcessAsUserW.restype = wintypes.BOOL
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
    ctypes.POINTER(STARTUPINFOW),
    ctypes.POINTER(PROCESS_INFORMATION),
]
