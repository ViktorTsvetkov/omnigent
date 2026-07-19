r"""Native Windows restricted process-launch primitives.

Low IL provides a write jail through mandatory ``NO_WRITE_UP``; it does not
confine reads. Known Low-IL locations such as
``%USERPROFILE%\AppData\LocalLow`` and ``%TEMP%\Low`` remain writable.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
ERROR_ALREADY_EXISTS = 183
ERROR_ALREADY_EXISTS_HRESULT = ctypes.c_long(0x800700B7).value
GENERIC_ALL = 0x10000000
GENERIC_READ = 0x80000000
GENERIC_EXECUTE = 0x20000000
DACL_SECURITY_INFORMATION = 0x4
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
UNPROTECTED_DACL_SECURITY_INFORMATION = 0x20000000
SE_DACL_PROTECTED = 0x1000
GRANT_ACCESS = 1
TRUSTEE_IS_SID = 0
TRUSTEE_IS_UNKNOWN = 0
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
EXTENDED_STARTUPINFO_PRESENT = 0x00080000


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


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", wintypes.LPVOID),
        ("Capabilities", wintypes.LPVOID),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class TRUSTEE_W(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", wintypes.LPVOID),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", wintypes.LPWSTR),
    ]


class EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", TRUSTEE_W),
    ]


@dataclass
class AppContainerProfile:
    """Own an AppContainer SID and delete a newly-created profile on close."""

    name: str
    sid: int
    created: bool

    def close(self) -> None:
        if self.sid:
            kernel32.LocalFree(wintypes.HANDLE(self.sid))
            self.sid = 0
        if self.created:
            result = userenv.DeleteAppContainerProfile(self.name)
            if result:
                _raise_last_error("DeleteAppContainerProfile", result & 0xFFFF)
            self.created = False


@dataclass
class AppContainerAccessGrant:
    """Snapshot an object's DACL and restore it when closed."""

    path: str
    security_descriptor: int
    dacl: int
    protected: bool

    def close(self) -> None:
        if not self.security_descriptor:
            return
        flags = DACL_SECURITY_INFORMATION | (
            PROTECTED_DACL_SECURITY_INFORMATION
            if self.protected
            else UNPROTECTED_DACL_SECURITY_INFORMATION
        )
        error = advapi32.SetNamedSecurityInfoW(
            self.path,
            SE_FILE_OBJECT,
            flags,
            None,
            None,
            wintypes.LPVOID(self.dacl),
            None,
        )
        descriptor, self.security_descriptor = self.security_descriptor, 0
        kernel32.LocalFree(wintypes.HANDLE(descriptor))
        if error:
            _raise_last_error("SetNamedSecurityInfoW", error)


def cleanup_appcontainer(
    profile: AppContainerProfile, grants: Sequence[AppContainerAccessGrant]
) -> None:
    """Restore all ACL snapshots and delete the ephemeral profile."""
    errors: list[OSError] = []
    for grant in reversed(grants):
        try:
            grant.close()
        except OSError as exc:
            errors.append(exc)
    try:
        profile.close()
    except OSError as exc:
        errors.append(exc)
    if errors:
        raise errors[0]


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


def create_appcontainer_profile(name: str) -> AppContainerProfile:
    """Create a capability-free AppContainer profile, or derive its SID."""
    sid = wintypes.LPVOID()
    result = userenv.CreateAppContainerProfile(
        name, name, "Omnigent ephemeral sandbox", None, 0, ctypes.byref(sid)
    )
    created = result == 0
    if result in (ERROR_ALREADY_EXISTS, ERROR_ALREADY_EXISTS_HRESULT):
        result = userenv.DeriveAppContainerSidFromAppContainerName(name, ctypes.byref(sid))
    if result:
        _raise_last_error("CreateAppContainerProfile", result & 0xFFFF)
    return AppContainerProfile(name, int(sid.value), created)


def grant_appcontainer_access(
    path: os.PathLike[str] | str,
    sid: int,
    access: int,
    *,
    inherit: bool = True,
) -> AppContainerAccessGrant:
    """Add inheritable AppContainer and ALL APPLICATION PACKAGES allow ACEs."""
    target = os.fspath(path)
    descriptor = wintypes.LPVOID()
    old_dacl = wintypes.LPVOID()
    error = advapi32.GetNamedSecurityInfoW(
        target,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(old_dacl),
        None,
        ctypes.byref(descriptor),
    )
    if error:
        _raise_last_error("GetNamedSecurityInfoW", error)

    all_apps = wintypes.LPVOID()
    new_dacl = wintypes.LPVOID()
    try:
        if not advapi32.ConvertStringSidToSidW("S-1-15-2-1", ctypes.byref(all_apps)):
            _raise_last_error("ConvertStringSidToSidW")
        inheritance = (
            SUB_CONTAINERS_AND_OBJECTS_INHERIT if inherit and Path(target).is_dir() else 0
        )
        entries = (EXPLICIT_ACCESS_W * 2)()
        for entry, entry_sid in zip(entries, (sid, int(all_apps.value)), strict=True):
            entry.grfAccessPermissions = access
            entry.grfAccessMode = GRANT_ACCESS
            entry.grfInheritance = inheritance
            entry.Trustee.TrusteeForm = TRUSTEE_IS_SID
            entry.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
            entry.Trustee.ptstrName = ctypes.cast(wintypes.LPVOID(entry_sid), wintypes.LPWSTR)
        error = advapi32.SetEntriesInAclW(2, entries, old_dacl, ctypes.byref(new_dacl))
        if error:
            _raise_last_error("SetEntriesInAclW", error)
        error = advapi32.SetNamedSecurityInfoW(
            target,
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            None,
            None,
            new_dacl,
            None,
        )
        if error:
            _raise_last_error("SetNamedSecurityInfoW", error)
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not advapi32.GetSecurityDescriptorControl(
            descriptor, ctypes.byref(control), ctypes.byref(revision)
        ):
            _raise_last_error("GetSecurityDescriptorControl")
        return AppContainerAccessGrant(
            target,
            int(descriptor.value),
            int(old_dacl.value) if old_dacl else 0,
            bool(control.value & SE_DACL_PROTECTED),
        )
    except BaseException:
        kernel32.LocalFree(descriptor)
        raise
    finally:
        if new_dacl:
            kernel32.LocalFree(new_dacl)
        if all_apps:
            kernel32.LocalFree(all_apps)


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


def create_process_appcontainer(
    sid: int,
    argv: Sequence[str],
    cwd: os.PathLike[str] | str | None,
    env: Mapping[str, str],
    read_roots: Sequence[os.PathLike[str] | str],
    write_roots: Sequence[os.PathLike[str] | str],
) -> tuple[int, int, list[AppContainerAccessGrant]]:
    """Grant launch dependencies and start argv in a no-capability AppContainer."""
    if not argv:
        raise ValueError("argv must not be empty")
    executable = Path(argv[0]).resolve(strict=False)
    launch_reads = [executable.parent, Path(sys.base_prefix)]
    if cwd is not None:
        launch_reads.append(Path(cwd).resolve(strict=False))
    pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        launch_reads.append(pyvenv_cfg)
    launch_reads.extend(
        Path(arg).resolve(strict=False).parent
        for arg in argv[1:]
        if arg.lower().endswith((".py", ".pyw"))
    )
    grants: list[AppContainerAccessGrant] = []
    seen: set[tuple[str, int, bool]] = set()
    try:
        requested = [
            *((path, GENERIC_READ | GENERIC_EXECUTE) for path in launch_reads),
            *((path, GENERIC_READ | GENERIC_EXECUTE) for path in read_roots),
            *((path, GENERIC_ALL) for path in write_roots),
        ]
        for path, access in requested:
            key = (os.path.normcase(os.path.abspath(os.fspath(path))), access, True)
            if key in seen:
                continue
            seen.add(key)
            grants.append(grant_appcontainer_access(path, sid, access))

        capabilities = SECURITY_CAPABILITIES(wintypes.LPVOID(sid), None, 0, 0)
        size = ctypes.c_size_t()
        kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_buffer, wintypes.LPVOID)
        if not kernel32.InitializeProcThreadAttributeList(
            attribute_list, 1, 0, ctypes.byref(size)
        ):
            _raise_last_error("InitializeProcThreadAttributeList")
        try:
            if not kernel32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                ctypes.byref(capabilities),
                ctypes.sizeof(capabilities),
                None,
                None,
            ):
                _raise_last_error("UpdateProcThreadAttribute")
            command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
            environment = _environment_block(env)
            startup = STARTUPINFOEXW()
            startup.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            startup.lpAttributeList = attribute_list
            process_info = PROCESS_INFORMATION()
            if not kernel32.CreateProcessW(
                str(executable),
                command_line,
                None,
                None,
                True,
                CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                environment,
                os.fspath(cwd) if cwd is not None else None,
                ctypes.byref(startup.StartupInfo),
                ctypes.byref(process_info),
            ):
                _raise_last_error("CreateProcessW")
            kernel32.CloseHandle(process_info.hThread)
            return int(process_info.dwProcessId), int(process_info.hProcess), grants
        finally:
            kernel32.DeleteProcThreadAttributeList(attribute_list)
    except BaseException:
        for grant in reversed(grants):
            grant.close()
        raise


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
kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
kernel32.InitializeProcThreadAttributeList.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
kernel32.UpdateProcThreadAttribute.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.c_size_t,
    wintypes.LPVOID,
    ctypes.c_size_t,
    wintypes.LPVOID,
    wintypes.LPVOID,
]
kernel32.DeleteProcThreadAttributeList.restype = None
kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
kernel32.CreateProcessW.restype = wintypes.BOOL
kernel32.CreateProcessW.argtypes = [
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
advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
advapi32.GetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.LPVOID),
]
advapi32.SetEntriesInAclW.restype = wintypes.DWORD
advapi32.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,
    ctypes.POINTER(EXPLICIT_ACCESS_W),
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.LPVOID),
]
advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
advapi32.GetSecurityDescriptorControl.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.WORD),
    ctypes.POINTER(wintypes.DWORD),
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

userenv = ctypes.WinDLL("userenv", use_last_error=True)
userenv.CreateAppContainerProfile.restype = wintypes.LONG
userenv.CreateAppContainerProfile.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
]
userenv.DeriveAppContainerSidFromAppContainerName.restype = wintypes.LONG
userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(wintypes.LPVOID),
]
userenv.DeleteAppContainerProfile.restype = wintypes.LONG
userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
