import json
import time

import pytest
from agentic_mcp_foundation import LifecycleError
from agentic_mcp_foundation.transport import rpc


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    """Never touch a user's real runtime, and retire only test-created coordinators."""
    monkeypatch.setenv("LOCAL_VISION_RUNTIME_DIR", str(tmp_path / "runtime"))
    yield
    for path in tmp_path.rglob("endpoint.json"):
        deadline = time.monotonic() + 5
        while path.exists() and time.monotonic() < deadline:
            try:
                rpc(json.loads(path.read_text()), "retire", timeout=1)
                break
            except (OSError, ValueError, LifecycleError):
                time.sleep(0.05)
