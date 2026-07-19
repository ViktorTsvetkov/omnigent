from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from omnigent._platform import IS_WINDOWS
from omnigent.inner.datamodel import OSEnvSandboxSpec, OSEnvSpec
from omnigent.inner.os_env import create_os_environment

pytestmark = pytest.mark.skipif(not IS_WINDOWS, reason="native Windows sandbox only")


def test_real_helper_enforces_write_root_and_keeps_reads_open(tmp_path: Path) -> None:
    """Exercise C1+C2 through the public caller-process environment."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    seeded = tmp_path / "seeded.txt"
    seeded.write_text("readable")
    environment = create_os_environment(
        OSEnvSpec(
            type="caller_process",
            cwd=str(tmp_path),
            sandbox=OSEnvSandboxSpec(
                type="windows_jobobject",
                read_paths=None,
                write_paths=["allowed"],
                allow_network=True,
            ),
        )
    )
    assert environment is not None
    try:
        write = asyncio.run(environment.write("allowed/result.txt", "enforced"))
        read = asyncio.run(environment.read("seeded.txt"))
        denied = asyncio.run(environment.write("outside.txt", "blocked"))
    finally:
        environment.close()

    assert write["created"] is True
    assert read["content"] == "readable"
    assert "blocked by sandbox" in denied["error"]
    assert (allowed / "result.txt").read_text() == "enforced"
    assert not (tmp_path / "outside.txt").exists()
