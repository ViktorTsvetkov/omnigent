"""Platform gate tests for session workspace validation."""

from unittest.mock import Mock

import pytest

from omnigent.errors import OmnigentError
from omnigent.server.routes import sessions


async def test_windows_absolute_workspace_rejected_on_posix() -> None:
    """POSIX restores the upstream absolute-path guard byte-for-byte."""
    request = Mock()
    request.app.state.host_registry.get_host_platform.return_value = "linux"

    with pytest.raises(OmnigentError, match="absolute path starting with /"):
        await sessions._validate_session_workspace(
            user_id=None,
            host_id="host_test",
            workspace=r"C:\Repos\omnigent",
            agent=None,
            agent_cache=None,
            request=request,
        )


async def test_windows_absolute_workspace_passes_guard_on_windows() -> None:
    """Windows absolute paths reach validation beyond the POSIX guard."""
    request = Mock()
    request.app.state.host_registry.get_host_platform.return_value = "windows"

    with pytest.raises(OmnigentError, match="requires an agent cache"):
        await sessions._validate_session_workspace(
            user_id=None,
            host_id="host_test",
            workspace=r"C:\Repos\omnigent",
            agent=None,
            agent_cache=None,
            request=request,
        )
