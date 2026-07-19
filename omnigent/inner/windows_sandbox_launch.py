"""Wrapper launcher for native Windows restricted process tiers."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import site
import sys
import uuid
from pathlib import Path
from typing import Any

if os.name != "nt":
    raise ImportError("windows_sandbox_launch is only available on Windows")

from .sandbox import SandboxPolicy
from .windows_sandbox import (
    build_low_il_token,
    cleanup_appcontainer,
    create_appcontainer_profile,
    create_process_appcontainer,
    create_process_low_il,
    label_write_root_low,
    wait_process,
)


def encode_policy(policy: SandboxPolicy, cwd: Path) -> str:
    payload = {"policy": policy.to_jsonable(), "cwd": str(cwd)}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_policy(encoded: str) -> tuple[SandboxPolicy, Path]:
    payload: dict[str, Any] = json.loads(base64.urlsafe_b64decode(encoded).decode())
    return SandboxPolicy.from_jsonable(payload["policy"]), Path(payload["cwd"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    policy, cwd = decode_policy(args.policy)
    if policy.read_roots is not None or not policy.allow_network:
        profile = create_appcontainer_profile(f"omnigent-{uuid.uuid4().hex}")
        grants = []
        try:
            appcontainer_command = list(command)
            appcontainer_env = dict(os.environ)
            runtime_reads = list(policy.read_roots or [])
            if os.path.normcase(os.path.abspath(appcontainer_command[0])) == os.path.normcase(
                os.path.abspath(sys.executable)
            ) and os.path.normcase(sys._base_executable) != os.path.normcase(sys.executable):
                appcontainer_command[0] = sys._base_executable
                site_packages = [
                    path for path in site.getsitepackages() if Path(path).name == "site-packages"
                ]
                if appcontainer_command[1:3] == ["-m", "omnigent.inner.os_env"]:
                    importlib.import_module("omnigent.inner.os_env")
                for module in tuple(sys.modules.values()):
                    filename = getattr(module, "__file__", None)
                    if not filename:
                        continue
                    module_path = Path(filename).resolve(strict=False)
                    for site_path_text in site_packages:
                        site_path = Path(site_path_text).resolve(strict=False)
                        try:
                            relative = module_path.relative_to(site_path)
                        except ValueError:
                            continue
                        package_path = site_path / relative.parts[0]
                        runtime_reads.append(package_path)
                        break
                existing = appcontainer_env.get("PYTHONPATH")
                appcontainer_env["PYTHONPATH"] = os.pathsep.join(
                    [*site_packages, *([existing] if existing else [])]
                )
            _, process_handle, grants = create_process_appcontainer(
                profile.sid,
                appcontainer_command,
                cwd,
                appcontainer_env,
                runtime_reads,
                [*policy.write_roots, *policy.write_files],
            )
            return wait_process(process_handle)
        finally:
            cleanup_appcontainer(profile, grants)

    token = build_low_il_token()
    try:
        for path in [*policy.write_roots, *policy.write_files]:
            label_write_root_low(path)
        _, process_handle = create_process_low_il(token, command, cwd, os.environ)
        return wait_process(process_handle)
    finally:
        import ctypes
        import ctypes.wintypes as wintypes

        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(token))


if __name__ == "__main__":
    sys.exit(main())
