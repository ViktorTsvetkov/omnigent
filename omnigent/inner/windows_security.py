r"""Low-integrity token and reversible mandatory-label leases for Windows."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import importlib.metadata
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

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
DACL_SECURITY_INFORMATION = 0x4
GRANT_ACCESS = 1
SET_ACCESS = 2
REVOKE_ACCESS = 4
NO_INHERITANCE = 0
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 3
TRUSTEE_IS_SID = 0
TRUSTEE_IS_UNKNOWN = 0
FILE_GENERIC_READ = 0x00120089
FILE_GENERIC_EXECUTE = 0x001200A0
FILE_GENERIC_WRITE = 0x00120116
DELETE = 0x00010000
APP_CONTAINER_PROFILE_NAME = "omnigent-sandbox-v1"
TREE_SEC_INFO_SET = 0x1
PROGRESS_INVOKE_NEVER = 1
_RX_STATE_PATH = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    / "omnigent"
    / "sandbox"
    / f"{APP_CONTAINER_PROFILE_NAME}-rx-roots.json"
)


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]


class _TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", _SID_AND_ATTRIBUTES)]


class _TRUSTEE_W(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", wintypes.LPVOID),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", wintypes.LPWSTR),
    ]


class _EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _TRUSTEE_W),
    ]


class SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", wintypes.LPVOID),
        ("Capabilities", ctypes.POINTER(_SID_AND_ATTRIBUTES)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


def _raise(name: str, error: int | None = None) -> NoReturn:
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
class AclLease:
    """Restore a DACL after the last concurrent lease is released."""

    path: str
    key: str | None

    def close(self) -> None:
        if self.key is not None:
            _release_acl(self.key)
            self.key = None


@dataclass
class _AclState:
    path: str
    descriptor: ctypes.Array[ctypes.c_char]
    access_mask: int
    references: int = 1


_ACL_LOCK = threading.Lock()
_ACL_STATES: dict[str, _AclState] = {}


def _release_acl(key: str) -> None:
    with _ACL_LOCK:
        state = _ACL_STATES[key]
        state.references -= 1
        if state.references:
            return
        if not advapi32.SetFileSecurityW(
            state.path, DACL_SECURITY_INFORMATION, ctypes.byref(state.descriptor)
        ):
            state.references = 1
            _raise("SetFileSecurityW (restore DACL)")
        del _ACL_STATES[key]


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

    token: OwnedHandle | None
    label_leases: list[LabelLease] = field(default_factory=list)
    acl_leases: list[AclLease] = field(default_factory=list)
    appcontainer_sid: wintypes.LPVOID | None = None
    security_capabilities: SECURITY_CAPABILITIES | None = None

    def close(self) -> None:
        errors: list[BaseException] = []
        for label_lease in reversed(self.label_leases):
            try:
                label_lease.close()
            except OSError as exc:
                errors.append(exc)
        for acl_lease in reversed(self.acl_leases):
            try:
                acl_lease.close()
            except OSError as exc:
                errors.append(exc)
        self.label_leases.clear()
        self.acl_leases.clear()
        if self.token is not None:
            try:
                self.token.close()
            except OSError as exc:
                errors.append(exc)
        if self.appcontainer_sid:
            kernel32.LocalFree(self.appcontainer_sid)
            self.appcontainer_sid = None
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


def _sid_from_string(value: str) -> wintypes.LPVOID:
    sid = wintypes.LPVOID()
    if not advapi32.ConvertStringSidToSidW(value, ctypes.byref(sid)):
        _raise("ConvertStringSidToSidW")
    return sid


def _explicit_access_entries(
    sids: list[wintypes.LPVOID], access_mask: int, access_mode: int, inheritance: int
) -> ctypes.Array[_EXPLICIT_ACCESS_W]:
    """Build one EXPLICIT_ACCESS_W per SID granting the same mask/mode/inheritance."""
    entries = (_EXPLICIT_ACCESS_W * len(sids))()
    for entry, sid in zip(entries, sids, strict=True):
        entry.grfAccessPermissions = access_mask
        entry.grfAccessMode = access_mode
        entry.grfInheritance = inheritance
        entry.Trustee.TrusteeForm = TRUSTEE_IS_SID
        entry.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
        entry.Trustee.ptstrName = ctypes.cast(sid, wintypes.LPWSTR)
    return entries


def _effective_rights(old_dacl: wintypes.LPVOID, sid: wintypes.LPVOID) -> tuple[int, int]:
    """Return (error, granted-mask) for one SID against an existing DACL."""
    trustee = _TRUSTEE_W(
        None, 0, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, ctypes.cast(sid, wintypes.LPWSTR)
    )
    effective = wintypes.DWORD()
    error = advapi32.GetEffectiveRightsFromAclW(
        old_dacl, ctypes.byref(trustee), ctypes.byref(effective)
    )
    return error, effective.value


def _grant_acl(path: Path, sids: list[wintypes.LPVOID], access_mask: int) -> AclLease:
    target = os.fspath(path)
    key = os.path.normcase(os.path.abspath(target))
    with _ACL_LOCK:
        existing = _ACL_STATES.get(key)
        if existing is not None:
            if existing.access_mask != access_mask:
                raise ValueError(f"conflicting AppContainer ACL leases for {path}")
            existing.references += 1
            return AclLease(target, key)
        needed = wintypes.DWORD()
        advapi32.GetFileSecurityW(target, DACL_SECURITY_INFORMATION, None, 0, ctypes.byref(needed))
        if not needed.value:
            _raise("GetFileSecurityW(DACL size)")
        descriptor = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetFileSecurityW(
            target, DACL_SECURITY_INFORMATION, descriptor, needed, ctypes.byref(needed)
        ):
            _raise("GetFileSecurityW(DACL)")
        old_dacl = wintypes.LPVOID()
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        if not advapi32.GetSecurityDescriptorDacl(
            descriptor, ctypes.byref(present), ctypes.byref(old_dacl), ctypes.byref(defaulted)
        ):
            _raise("GetSecurityDescriptorDacl")
        inheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT if path.is_dir() else NO_INHERITANCE
        entries = _explicit_access_entries(sids, access_mask, GRANT_ACCESS, inheritance)
        new_dacl = wintypes.LPVOID()
        error = advapi32.SetEntriesInAclW(len(entries), entries, old_dacl, ctypes.byref(new_dacl))
        if error:
            _raise("SetEntriesInAclW", error)
        try:
            error = advapi32.SetNamedSecurityInfoW(
                target, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None, new_dacl, None
            )
            if error:
                _raise("SetNamedSecurityInfoW(DACL)", error)
            _ACL_STATES[key] = _AclState(target, descriptor, access_mask)
            return AclLease(target, key)
        finally:
            if new_dacl:
                kernel32.LocalFree(new_dacl)


def _parents(path: Path) -> list[Path]:
    result: list[Path] = []
    current = path.parent
    while current != current.parent:
        result.append(current)
        current = current.parent
    return result


def _grant_tree(
    root: Path,
    sids: list[wintypes.LPVOID],
    access_mask: int,
) -> None:
    """Persist an inherited SID grant and propagate it to existing objects once."""
    _set_persistent_acl(root, sids, access_mask, SET_ACCESS, propagate=True)


def _set_persistent_acl(
    path: Path,
    sids: list[wintypes.LPVOID],
    access_mask: int,
    access_mode: int,
    *,
    propagate: bool = False,
    inherit: bool | None = None,
) -> None:
    """Converge ACEs for stable profile SIDs without changing other trustees."""
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
        _raise(f"GetNamedSecurityInfoW(DACL): {target}", error)
    if access_mode == SET_ACCESS:
        system_roots = [
            Path(os.environ.get("SYSTEMROOT", r"C:\Windows")).resolve(),
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")).resolve(),
        ]
        if any(path == root or path.is_relative_to(root) for root in system_roots):
            all_apps = _sid_from_string("S-1-15-2-1")
            try:
                error, granted = _effective_rights(old_dacl, all_apps)
                if not error and granted & access_mask == access_mask:
                    kernel32.LocalFree(descriptor)
                    return
            finally:
                kernel32.LocalFree(all_apps)
        error, granted = _effective_rights(old_dacl, sids[0])
        if error:
            kernel32.LocalFree(descriptor)
            _raise(f"GetEffectiveRightsFromAclW: {target}", error)
        if granted & access_mask == access_mask:
            kernel32.LocalFree(descriptor)
            return
    should_inherit = path.is_dir() if inherit is None else inherit
    inheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT if should_inherit else NO_INHERITANCE
    entries = _explicit_access_entries(sids, access_mask, access_mode, inheritance)
    new_dacl = wintypes.LPVOID()
    try:
        error = advapi32.SetEntriesInAclW(len(entries), entries, old_dacl, ctypes.byref(new_dacl))
        if error:
            _raise(f"SetEntriesInAclW(persistent RX): {target}", error)
        if propagate and path.is_dir():
            error = advapi32.TreeSetNamedSecurityInfoW(
                target,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_dacl,
                None,
                TREE_SEC_INFO_SET,
                None,
                PROGRESS_INVOKE_NEVER,
                None,
            )
        else:
            error = advapi32.SetNamedSecurityInfoW(
                target, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None, new_dacl, None
            )
        if error:
            _raise(f"SetNamedSecurityInfoW(persistent RX): {target}", error)
    finally:
        if new_dacl:
            kernel32.LocalFree(new_dacl)
        if descriptor:
            kernel32.LocalFree(descriptor)


def _runtime_read_paths(argv0: str | None = None) -> list[Path]:
    """Return narrow loader/import roots needed by the Omnigent helper."""
    from packaging.requirements import Requirement

    import omnigent

    executable = Path(argv0 or sys.executable).resolve()
    package = Path(omnigent.__file__).resolve().parent
    candidates = [executable.parent, Path(sys.base_prefix).resolve(), package]
    pyvenv_cfg = Path(sys.prefix) / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        candidates.append(pyvenv_cfg.resolve())
    discovered_sites = {
        Path(entry).resolve()
        for entry in sys.path
        if entry and "site-packages" in Path(entry).parts and Path(entry).exists()
    }
    site_packages = {
        site
        for site in discovered_sites
        if not any(other != site and site.is_relative_to(other) for other in discovered_sites)
    }
    pending = ["omnigent"]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        key = name.casefold().replace("_", "-")
        if key in visited:
            continue
        visited.add(key)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for file in distribution.files or []:
            resolved = Path(str(distribution.locate_file(file))).resolve(strict=False)
            for site in site_packages:
                if resolved.is_relative_to(site):
                    relative = resolved.relative_to(site)
                    if relative.parts:
                        candidate = site / relative.parts[0]
                        if candidate.exists() and not candidate.name.endswith(".dist-info"):
                            candidates.append(candidate)
                    break
        for requirement_text in distribution.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    for entry in sys.path:
        if not entry or entry.startswith("__editable__"):
            continue
        candidate = Path(entry).resolve(strict=False)
        if candidate.exists() and not any(
            candidate.is_relative_to(site) for site in site_packages
        ):
            if (
                candidate != prefix
                and candidate != base_prefix
                and not candidate.is_relative_to(base_prefix)
            ):
                candidates.append(candidate)
    return list(dict.fromkeys(candidates))


def create_or_open_appcontainer_profile() -> wintypes.LPVOID:
    """Create the stable installation profile, or derive its existing SID.

    The versioned identity is intentionally stable: concurrent helpers share it,
    and explicit uninstall/migration code may call cleanup_appcontainer_profile.
    """
    sid = wintypes.LPVOID()
    result = userenv.CreateAppContainerProfile(
        APP_CONTAINER_PROFILE_NAME,
        "Omnigent sandbox v1",
        "Omnigent admin-free filesystem and network sandbox",
        None,
        0,
        ctypes.byref(sid),
    )
    if result == 0:
        return sid
    if ctypes.c_uint32(result).value != 0x800700B7:
        raise OSError(ctypes.c_uint32(result).value, "CreateAppContainerProfile")
    if userenv.DeriveAppContainerSidFromAppContainerName(
        APP_CONTAINER_PROFILE_NAME, ctypes.byref(sid)
    ) == 0:
        return sid
    _raise("DeriveAppContainerSidFromAppContainerName")


def cleanup_appcontainer_profile() -> None:
    """Remove the stable profile after its durable grants have been removed."""
    error = userenv.DeleteAppContainerProfile(APP_CONTAINER_PROFILE_NAME)
    if error not in (0, 2, 1168):
        _raise("DeleteAppContainerProfile", error)


def _read_rx_state() -> tuple[set[Path], set[Path]]:
    try:
        raw = json.loads(_RX_STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set(), set()
    if isinstance(raw, list):
        # Pre-C3 development manifests did not distinguish propagation scope.
        # Treat them conservatively as root-only to avoid sweeping ancestors.
        return set(), {Path(str(value)) for value in raw}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid AppContainer RX state: {_RX_STATE_PATH}")
    trees = raw.get("trees", [])
    traversal = raw.get("traversal", [])
    if not isinstance(trees, list) or not isinstance(traversal, list):
        raise ValueError(f"invalid AppContainer RX state: {_RX_STATE_PATH}")
    return (
        {Path(str(value)) for value in trees},
        {Path(str(value)) for value in traversal},
    )


def _write_rx_state(trees: set[Path], traversal: set[Path]) -> None:
    _RX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _RX_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "trees": sorted(map(os.fspath, trees)),
                "traversal": sorted(map(os.fspath, traversal)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, _RX_STATE_PATH)


def setup_appcontainer_read_grants(
    read_roots: list[Path],
    *,
    executable: str | None = None,
    traversal_targets: list[Path] | None = None,
) -> None:
    """Ensure durable, idempotent RX grants for the stable profile SID."""
    sid = create_or_open_appcontainer_profile()
    try:
        explicit = {Path(path).resolve(strict=True) for path in read_roots}
        runtime = set(_runtime_read_paths(executable))
        runtime = {
            path
            for path in runtime
            if not any(
                other != path and other.is_dir() and path.is_relative_to(other)
                for other in runtime
            )
        }
        requested = runtime | explicit
        traversal_sources = requested | {
            Path(path).resolve(strict=False) for path in (traversal_targets or [])
        }
        traversal = {parent for path in traversal_sources for parent in _parents(path)}
        configured_trees, configured_traversal = _read_rx_state()
        mask = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
        for path in sorted((runtime - configured_trees) | explicit, key=os.fspath):
            _grant_tree(path, [sid], mask)
        for path in sorted(traversal, key=os.fspath):
            _set_persistent_acl(
                path, [sid], FILE_GENERIC_EXECUTE, SET_ACCESS, inherit=False
            )
        _write_rx_state(
            configured_trees | requested,
            configured_traversal | traversal,
        )
    finally:
        kernel32.LocalFree(sid)


def teardown_appcontainer_read_grants() -> None:
    """Remove this profile SID's durable ACEs, then remove profile metadata."""
    sid = wintypes.LPVOID()
    if userenv.DeriveAppContainerSidFromAppContainerName(
        APP_CONTAINER_PROFILE_NAME, ctypes.byref(sid)
    ) != 0:
        _RX_STATE_PATH.unlink(missing_ok=True)
        return
    try:
        trees, traversal = _read_rx_state()
        for root in sorted(trees, key=os.fspath, reverse=True):
            if not root.exists():
                continue
            _set_persistent_acl(root, [sid], 0, REVOKE_ACCESS, propagate=root.is_dir())
        for root in sorted(traversal - trees, key=os.fspath, reverse=True):
            if root.exists():
                _set_persistent_acl(root, [sid], 0, REVOKE_ACCESS)
        _RX_STATE_PATH.unlink(missing_ok=True)
    finally:
        kernel32.LocalFree(sid)


def teardown_appcontainer() -> None:
    """Explicitly remove durable SID grants and the stable profile."""
    teardown_appcontainer_read_grants()
    cleanup_appcontainer_profile()


def prepare_appcontainer_launch(
    policy: SandboxPolicy, *, executable: str | None = None
) -> WindowsLaunchSecurity:
    """Lease required ACLs and construct zero-capability AppContainer security."""
    if policy.allow_network:
        raise ValueError(
            "windows_appcontainer only supports allow_network=false with zero capabilities; "
            "select windows_jobobject for the Low-IL network-open tier"
        )
    if policy.egress_relay_port is not None or policy.egress_socket_path is not None:
        raise ValueError(
            "windows_appcontainer cannot combine hard network deny with active egress: "
            "AppContainer blocks loopback without an admin-gated exemption, so the relay "
            "would hang"
        )
    sid = create_or_open_appcontainer_profile()
    security = WindowsLaunchSecurity(None, appcontainer_sid=sid)
    try:
        setup_appcontainer_read_grants(
            policy.read_roots or [],
            executable=executable,
            traversal_targets=[*policy.write_roots, *policy.write_files],
        )
        modify = FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE
        for path in policy.write_roots:
            resolved = Path(path).resolve(strict=True)
            security.acl_leases.append(_grant_acl(resolved, [sid], modify))
        for path in policy.write_files:
            resolved = Path(path).resolve(strict=True)
            security.acl_leases.append(_grant_acl(resolved, [sid], modify))
        capabilities = SECURITY_CAPABILITIES(sid, None, 0, 0)
        security.security_capabilities = capabilities
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
advapi32.GetSecurityDescriptorDacl.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.BOOL),
    ctypes.POINTER(wintypes.LPVOID),
    ctypes.POINTER(wintypes.BOOL),
]
advapi32.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,
    ctypes.POINTER(_EXPLICIT_ACCESS_W),
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.LPVOID),
]
advapi32.GetEffectiveRightsFromAclW.argtypes = [
    wintypes.LPVOID,
    ctypes.POINTER(_TRUSTEE_W),
    ctypes.POINTER(wintypes.DWORD),
]
advapi32.TreeSetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.LPVOID,
    ctypes.c_int,
    wintypes.LPVOID,
]
advapi32.TreeSetNamedSecurityInfoW.restype = wintypes.DWORD
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
userenv = ctypes.WinDLL("userenv", use_last_error=True)
userenv.CreateAppContainerProfile.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.LPVOID),
]
userenv.CreateAppContainerProfile.restype = ctypes.c_long
userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(wintypes.LPVOID),
]
userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
userenv.DeleteAppContainerProfile.restype = ctypes.c_long
