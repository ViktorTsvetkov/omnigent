"""Native-Windows terminal backend driven by psmux."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .terminal import (
    Liveness,
    TerminalBackend,
    TerminalBackendCapabilities,
    TerminalLaunchRequest,
)


class PsmuxBackend(TerminalBackend):
    """Host one terminal in an isolated psmux server namespace."""

    name = "psmux"
    capabilities = TerminalBackendCapabilities()
    platforms = frozenset({"windows"})

    BIN_ENV_VAR = "OMNIGENT_PSMUX_BIN"
    DEFAULT_BIN = "psmux"
    _PROBE_TIMEOUT_S = 15.0
    _LITERAL_CHARS_PER_CALL = 1024

    def __init__(self, *, socket_path: str | Path, target: str = "main") -> None:
        self._socket_path = Path(socket_path)
        self._target = target
        identity = str(self._socket_path.resolve()).casefold().encode("utf-8")
        self._namespace = f"omnigent-{hashlib.sha256(identity).hexdigest()[:20]}"
        self._config_path = self._socket_path.parent / "psmux.conf"

    @classmethod
    def construct_for_instance(cls, *, socket_path: Path, target: str) -> TerminalBackend:
        return cls(socket_path=socket_path, target=target)

    @classmethod
    def _command_prefix(cls) -> list[str]:
        raw = os.environ.get(cls.BIN_ENV_VAR, cls.DEFAULT_BIN).strip() or cls.DEFAULT_BIN
        if raw.startswith("["):
            try:
                tokens = json.loads(raw)
            except json.JSONDecodeError:
                return [raw]
            if (
                isinstance(tokens, list)
                and tokens
                and all(isinstance(token, str) for token in tokens)
            ):
                return tokens
        return [raw]

    @classmethod
    def ensure_available(cls) -> None:
        argv = [*cls._command_prefix(), "--version"]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                check=False,
                timeout=cls._PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"psmux is not installed or not spawnable ({argv[0]!r}: {exc}). "
                "Install psmux and ensure it is on PATH, or point "
                f"{cls.BIN_ENV_VAR} at psmux.exe."
            ) from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"psmux version probe failed (rc={proc.returncode}): {stderr}")

    def _base_cmd(self) -> list[str]:
        return [
            *self._command_prefix(),
            "-L",
            self._namespace,
            "-f",
            str(self._config_path),
        ]

    @staticmethod
    def _normalize_output(data: bytes) -> str:
        return data.decode(errors="replace").replace("\r\n", "\n").replace("\r", "\n")

    async def _run(
        self,
        *args: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._base_cmd(),
            *args,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = self._normalize_output(stderr).strip()
            raise RuntimeError(f"psmux command failed: {' '.join(args)}: {detail}")

    async def _run_output(self, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            *self._base_cmd(),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = self._normalize_output(stderr).strip()
            raise RuntimeError(f"psmux command failed: {' '.join(args)}: {detail}")
        return self._normalize_output(stdout)

    def _run_sync(self, *args: str) -> None:
        proc = subprocess.run([*self._base_cmd(), *args], capture_output=True, check=False)
        if proc.returncode != 0:
            detail = self._normalize_output(proc.stderr).strip()
            raise RuntimeError(f"psmux command failed: {' '.join(args)}: {detail}")

    def _run_output_sync(self, *args: str) -> str:
        proc = subprocess.run([*self._base_cmd(), *args], capture_output=True, check=False)
        if proc.returncode != 0:
            detail = self._normalize_output(proc.stderr).strip()
            raise RuntimeError(f"psmux command failed: {' '.join(args)}: {detail}")
        return self._normalize_output(proc.stdout)

    async def launch(self, request: TerminalLaunchRequest) -> None:
        cols, rows = request.size
        config = [f"set-option -g history-limit {request.scrollback}"]
        if request.keep_alive_after_exit:
            config.append("set-option -g remain-on-exit on")
        self._config_path.write_text("\n".join(config) + "\n", encoding="utf-8")
        await self._run(
            "new-session",
            "-d",
            "-s",
            self._target,
            "-x",
            str(cols),
            "-y",
            str(rows),
            "--",
            *request.command,
            cwd=request.cwd,
            env=request.env,
        )

    @staticmethod
    def _liveness_verdict(returncode: int | None, stdout: str) -> Liveness:
        if returncode is None:
            return Liveness.UNKNOWN
        if returncode != 0:
            return Liveness.ENDPOINT_GONE
        panes = stdout.split()
        if not panes:
            return Liveness.ENDPOINT_GONE
        return Liveness.INNER_EXITED if "1" in panes else Liveness.ALIVE

    async def liveness(self) -> Liveness:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._base_cmd(),
                "list-panes",
                "-t",
                self._target,
                "-F",
                "#{pane_dead}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
        except OSError:
            return Liveness.UNKNOWN
        return self._liveness_verdict(proc.returncode, self._normalize_output(stdout))

    def liveness_sync(self) -> Liveness:
        try:
            proc = subprocess.run(
                [
                    *self._base_cmd(),
                    "list-panes",
                    "-t",
                    self._target,
                    "-F",
                    "#{pane_dead}",
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return Liveness.UNKNOWN
        return self._liveness_verdict(proc.returncode, self._normalize_output(proc.stdout))

    async def close(self) -> None:
        with contextlib.suppress(OSError, RuntimeError):
            await self._run("kill-server")

    async def send_text(self, text: str) -> None:
        for start in range(0, len(text), self._LITERAL_CHARS_PER_CALL):
            chunk = text[start : start + self._LITERAL_CHARS_PER_CALL]
            await self._run("send-keys", "-t", self._target, "-l", chunk)

    async def send_keys(self, keys: Sequence[str]) -> None:
        if keys:
            await self._run("send-keys", "-t", self._target, *keys)

    def send_text_sync(self, text: str) -> None:
        for start in range(0, len(text), self._LITERAL_CHARS_PER_CALL):
            chunk = text[start : start + self._LITERAL_CHARS_PER_CALL]
            self._run_sync("send-keys", "-t", self._target, "-l", chunk)

    def send_keys_sync(self, keys: Sequence[str]) -> None:
        if keys:
            self._run_sync("send-keys", "-t", self._target, *keys)

    def paste_without_submit_sync(self, text: str) -> None:
        self.send_text_sync(text)

    def kill_session_sync(self) -> None:
        with contextlib.suppress(OSError, RuntimeError):
            self._run_sync("kill-server")

    def _capture_args(self, *, ansi: bool, scrollback: int) -> list[str]:
        args = ["capture-pane", "-t", self._target, "-p"]
        if ansi:
            args.append("-e")
        if scrollback > 0:
            args.extend(["-S", f"-{scrollback}"])
        return args

    async def capture(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        return await self._run_output(*self._capture_args(ansi=ansi, scrollback=scrollback))

    def capture_sync(self, *, ansi: bool = False, scrollback: int = 0) -> str:
        return self._run_output_sync(*self._capture_args(ansi=ansi, scrollback=scrollback))

    async def cursor_position(self) -> tuple[int, int]:
        """Return the active pane cursor as zero-based ``(column, row)``."""
        output = await self._run_output(
            "display-message",
            "-p",
            "-t",
            self._target,
            "#{cursor_x},#{cursor_y}",
        )
        try:
            column, row = output.strip().split(",", maxsplit=1)
            return int(column), int(row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"psmux returned an invalid cursor position: {output!r}") from exc
