"""Wrapper launcher for the Windows low-integrity write jail."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

if os.name != "nt":
    raise ImportError("windows_sandbox_launch is only available on Windows")

from .sandbox import SandboxPolicy
from .windows_sandbox import (
    build_low_il_token,
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
