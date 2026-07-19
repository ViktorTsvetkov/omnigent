r"""Low-integrity token and reversible mandatory-label leases for Windows."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sandbox import SandboxPolicy

if os.name != "nt":
    raise ImportError("windows_security is only available on Windows")

TOKEN_ALL_ACCESS = 0x000F01FF
DISABLE_MAX_PRIVILEGE = 0x1
TOKEN_INTEGRITY_LEVEL = 25
SE_GROUP_INTEGRITY = 0x20
SDDL_REVISION_1 = 1
SE_FILE_OBJECT = 1
LABEL_SECURITY_INFORMATION = 0x10


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class _TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", _SID_AND_ATTRIBUTES)]


def _raise(name: str, error: int | None = None) -> None:
    raise ctypes.WinError(error if error is not None else ctypes.get_last_error(), name)


@dataclass
class OwnedHandle:
    """Own one CloseHandle-compatible kernel handle."""

    value: int | None

    def close(self) -> None:
        if self.value is None:
            return
        value, self.value = self.value, None
        if not kernel32.CloseHandle(wintypes.HANDLE(value)):
            _raise("CloseHandle")


@dataclass
class LabelLease:
    """Restore the mandatory label that existed before a temporary grant."""

    path: str
    key: str | None

    def close(self) -> None:
        if self.key is None:
            return
        _release_label(self.key)
        self.key = None


@dataclass
class _LabelState:
    path: str
    descriptor: ctypes.Array[ctypes.c_char]
    inherit: bool
    references: int = 1


_LABEL_LOCK = threading.Lock()
_LABEL_STATES: dict[str, _LabelState] = {}


def _release_label(key: str) -> None:
    with _LABEL_LOCK:
        state = _LABEL_STATES[key]
        state.references -= 1
        if state.references:
            return
        if not advapi32.SetFileSecurityW(
            state.path, LABEL_SECURITY_INFORMATION, ctypes.byref(state.descriptor)
        ):
            state.references = 1
            _raise("SetFileSecurityW (restore label)")
        del _LABEL_STATES[key]


@dataclass
class WindowsLaunchSecurity:
    """Token and label mutations needed for a Low-IL launch."""

    token: OwnedHandle
    label_leases: list[LabelLease] = field(default_factory=list)

    def close(self) -> None:
        errors: list[BaseException] = []
        for lease in reversed(self.label_leases):
            try:
                lease.close()
            except OSError as exc:
                errors.append(exc)
        self.label_leases.clear()
        try:
            self.token.close()
        except OSError as exc:
            errors.append(exc)
        if errors:
            raise errors[0]


def create_low_integrity_token() -> OwnedHandle:
    """Create a Low-IL restricted derivative of Omnigent's own token."""
    current = wintypes.HANDLE()
    restricted = wintypes.HANDLE()
    low_sid = wintypes.LPVOID()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), TOKEN_ALL_ACCESS, ctypes.byref(current)
    ):
        _raise("OpenProcessToken")
    try:
        if not advapi32.CreateRestrictedToken(
            current, DISABLE_MAX_PRIVILEGE, 0, None, 0, None, 0, None, ctypes.byref(restricted)
        ):
            _raise("CreateRestrictedToken")
        if not advapi32.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(low_sid)):
            _raise("ConvertStringSidToSidW")
        label = _TOKEN_MANDATORY_LABEL(_SID_AND_ATTRIBUTES(low_sid, SE_GROUP_INTEGRITY))
        size = ctypes.sizeof(_TOKEN_MANDATORY_LABEL) + advapi32.GetLengthSid(low_sid)
        if not advapi32.SetTokenInformation(
            restricted, TOKEN_INTEGRITY_LEVEL, ctypes.byref(label), size
        ):
            _raise("SetTokenInformation(TokenIntegrityLevel)")
        assert restricted.value is not None
        value = int(restricted.value)
        restricted = wintypes.HANDLE()
        return OwnedHandle(value)
    finally:
        if low_sid:
            kernel32.LocalFree(low_sid)
        if restricted:
            kernel32.CloseHandle(restricted)
        kernel32.CloseHandle(current)


def _grant_low_label(path: Path, *, inherit: bool) -> LabelLease:
    target = os.fspath(path)
    key = os.path.normcase(os.path.abspath(target))
    with _LABEL_LOCK:
        existing = _LABEL_STATES.get(key)
        if existing is not None:
            if existing.inherit != inherit:
                raise ValueError(f"conflicting Low-label leases for {path}")
            existing.references += 1
            return LabelLease(target, key)

        needed = wintypes.DWORD()
        advapi32.GetFileSecurityW(
            target, LABEL_SECURITY_INFORMATION, None, 0, ctypes.byref(needed)
        )
        if not needed.value:
            _raise("GetFileSecurityW(label size)")
        descriptor = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetFileSecurityW(
            target, LABEL_SECURITY_INFORMATION, descriptor, needed, ctypes.byref(needed)
        ):
            _raise("GetFileSecurityW(label)")

        new_descriptor = wintypes.LPVOID()
        new_sacl = wintypes.LPVOID()
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        sddl = "S:(ML;OICI;NW;;;LW)" if inherit else "S:(ML;;NW;;;LW)"
        try:
            if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, SDDL_REVISION_1, ctypes.byref(new_descriptor), None
            ):
                _raise("ConvertStringSecurityDescriptorToSecurityDescriptorW")
            if not advapi32.GetSecurityDescriptorSacl(
                new_descriptor,
                ctypes.byref(present),
                ctypes.byref(new_sacl),
                ctypes.byref(defaulted),
            ):
                _raise("GetSecurityDescriptorSacl")
            error = advapi32.SetNamedSecurityInfoW(
                target, SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION, None, None, None, new_sacl
            )
            if error:
                _raise("SetNamedSecurityInfoW(label)", error)
            _LABEL_STATES[key] = _LabelState(target, descriptor, inherit)
            return LabelLease(target, key)
        finally:
            if new_descriptor:
                kernel32.LocalFree(new_descriptor)


def grant_low_integrity_write_root(path: Path) -> LabelLease:
    """Grant inheritable Low-IL writes to an existing directory."""
    if not path.is_dir():
        raise ValueError(f"Windows write_root must be an existing directory: {path}")
    return _grant_low_label(path, inherit=True)


def grant_low_integrity_write_file(path: Path) -> LabelLease:
    """Grant Low-IL writes to one existing non-container object."""
    if not path.is_file():
        raise ValueError(f"Windows write_file must be an existing file: {path}")
    return _grant_low_label(path, inherit=False)


def prepare_low_integrity_launch(policy: SandboxPolicy) -> WindowsLaunchSecurity:
    """Validate a write-only policy, lower a token, and lease its Low labels."""
    read_roots = policy.read_roots
    if read_roots is not None:
        raise ValueError(
            "Windows Low-IL sandbox cannot enforce read_roots; AppContainer/C3 is required "
            "for read confinement and is not implemented yet"
        )
    roots = [Path(path).resolve(strict=False) for path in policy.write_roots]
    files = [Path(path).resolve(strict=False) for path in policy.write_files]
    security = WindowsLaunchSecurity(create_low_integrity_token())
    try:
        for root in roots:
            security.label_leases.append(grant_low_integrity_write_root(root))
        for path in files:
            if path.exists():
                security.label_leases.append(grant_low_integrity_write_file(path))
                continue
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError(
                    f"Windows write_file does not exist and its parent is not covered by a "
                    f"write_root: {path}"
                )
        return security
    except BaseException:
        security.close()
        raise


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
kernel32.LocalFree.restype = wintypes.HLOCAL
advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
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
advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.LPVOID)]
advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
advapi32.GetLengthSid.restype = wintypes.DWORD
advapi32.SetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
]
advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.DWORD),
]
advapi32.GetSecurityDescriptorSacl.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.BOOL),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.BOOL),
]
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
advapi32.SetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
]
advapi32.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
advapi32.GetFileSecurityW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
