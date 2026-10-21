import asyncio
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest
from agentic_mcp_foundation import LifecycleError
from agentic_mcp_foundation.transport import LoopbackServer
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from local_vision_mcp.coordinator import Coordinator, RemoteService, digest_file
from local_vision_mcp.setup import recommend


@pytest.fixture
def backend():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version":"fixture"}')

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            if self.path == "/api/chat":
                calls.append(request)
                if "wait-for-loss" in json.dumps(request):
                    time.sleep(2)
                try:
                    self.wfile.write(
                        json.dumps(
                            {
                                "message": {
                                    "content": '{"count":2}' if request.get("format") else "red"
                                }
                            }
                        ).encode()
                    )
                except BrokenPipeError:
                    pass
            elif self.path == "/api/show":
                self.wfile.write(b'{"capabilities":["vision"]}')
            else:
                self.wfile.write(b'{"status":"success"}\n')

    server = LoopbackServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()
    server.server_close()
    thread.join(3)


def test_local_vision_reconnect_receipt_media_identity_and_probe(tmp_path, backend):
    url, calls = backend
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ollama_url": url, "omni_enabled": False}))
    image = tmp_path / "red.png"
    Image.new("RGB", (96, 96), "red").save(image)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "local_vision_mcp"],
        env={
            "LOCAL_VISION_CONFIG": str(config),
            "LOCAL_VISION_ROOTS": str(tmp_path),
            "LOCAL_VISION_RUNTIME_DIR": str(tmp_path / "runtime"),
        },
    )
    request = {"path": str(image), "question": "What color?", "idempotency_key": "stable-image"}

    async def wait(client, job_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = await client.call_tool("get_media_job", {"job_id": job_id})
            assert not response.isError
            job = json.loads(response.content[0].text)
            if job["state"] in ("succeeded", "failed"):
                assert job["state"] == "succeeded", job
                return job
            await asyncio.sleep(0.05)
        pytest.fail("Timed out waiting for analysis")

    async def run():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            response = await client.call_tool("submit_media_job", request)
            assert not response.isError
            job = json.loads(response.content[0].text)
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            recovered = await wait(client, job["id"])
            assert recovered["result"]["answer"] == "red"
            assert recovered["result"]["media_sha256"] == digest_file(image)
            replay = await client.call_tool("submit_media_job", request)
            assert json.loads(replay.content[0].text)["id"] == job["id"]
            assert len(calls) == 1
            schema = {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            }
            structured = await client.call_tool(
                "submit_media_job",
                {**request, "idempotency_key": "structured", "response_schema": schema},
            )
            structured_job = json.loads(structured.content[0].text)
            structured_result = (await wait(client, structured_job["id"]))["result"]
            assert structured_result["output_validation"]["valid"]
            assert calls[-1]["format"] == schema
            conflict_schema = await client.call_tool(
                "submit_media_job",
                {
                    **request,
                    "idempotency_key": "structured",
                    "response_schema": {**schema, "required": []},
                },
            )
            assert conflict_schema.isError
            status = await client.call_tool("local_media_status", {})
            data = json.loads(status.content[0].text)
            assert data["capabilities"]["image"]["ready"]
            assert not data["capabilities"]["audio"]["ready"]
            assert data["capabilities"]["qualified"] is False
            Image.new("RGB", (96, 96), "blue").save(image)
            conflict = await client.call_tool("submit_media_job", request)
            assert conflict.isError
            verify = await client.call_tool("verify_local_media", {"idempotency_key": "probe"})
            probe = json.loads(verify.content[0].text)
            assert (await wait(client, probe["id"]))["result"]["passed"]

    asyncio.run(run())


def test_different_roots_cannot_recover_each_others_jobs(tmp_path):
    config = recommend({"ram_gb": 8, "cuda_vram_gb": 0})
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = Coordinator(config, (a,), tmp_path / "coordinator-a")
    second = Coordinator(config, (b,), tmp_path / "coordinator-b")
    try:
        job = first.store.create("private", "analyze", {"path": "private"})
        with pytest.raises(LifecycleError, match="No job"):
            second.store.get(job["id"])
    finally:
        first.service.close()
        second.service.close()


def test_dead_coordinator_recovery_preserves_results_without_replay(tmp_path, backend):
    url, calls = backend
    config = recommend({"ram_gb": 8, "cuda_vram_gb": 0})
    config["ollama_url"] = url
    first = RemoteService(config, (tmp_path,))
    image = tmp_path / "input.png"
    Image.new("RGB", (96, 96), "red").save(image)
    request = {"path": str(image), "question": "Color?", "kind": "image"}
    job = first.submit_job(request, "completed-before-loss")
    deadline = time.monotonic() + 5
    while first.get_job(job["id"])["state"] != "succeeded":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    pid = first.call("health")["pid"]
    unfinished = first.submit_job({**request, "question": "wait-for-loss"}, "running-at-loss")
    while len(calls) != 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    # Dedicated test coordinator; never a discovered production process.
    os.kill(pid, signal.SIGKILL)
    recovered = first.get_job(job["id"])
    second = RemoteService(config, (tmp_path,))
    assert recovered["state"] == "succeeded"
    assert recovered["result"]["answer"] == "red"
    assert second.call("health")["pid"] != pid
    assert second.get_job(unfinished["id"])["state"] == "interrupted"
    assert second.submit_job(request, "completed-before-loss")["id"] == job["id"]
    assert len(calls) == 2
