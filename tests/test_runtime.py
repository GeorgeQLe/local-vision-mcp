import asyncio
import io
import json
import socket
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import zstandard
from agentic_mcp_foundation import processes
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from local_vision_mcp.runtime import Runtime, extract_archive
from local_vision_mcp.setup import recommend


@pytest.fixture
def runtime(tmp_path):
    config = recommend({"ram_gb": 8, "cuda_vram_gb": 0})
    instance = Runtime(config, "http://127.0.0.1:11434", tmp_path / "runtime")
    yield instance
    instance.close()


def wait_job(runtime):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with runtime.lock:
            if runtime.job["state"] != "running":
                return dict(runtime.job)
        time.sleep(0.01)
    pytest.fail("Management job did not complete")


def archive_bytes(name="bin/ollama", link=None):
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w") as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = link
            archive.addfile(member)
        else:
            data = b"#!/bin/sh\nexit 0\n"
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    return content.getvalue()


@pytest.mark.parametrize("name,link", [("../escape", None), ("bin/ollama", "/tmp/escape")])
def test_archive_rejects_escaping_paths(tmp_path, name, link):
    archive = tmp_path / "archive.tar"
    archive.write_bytes(archive_bytes(name, link))
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(archive, tmp_path / "install")


def test_atomic_install_and_failed_update_preserves_current(runtime, monkeypatch):
    runtime.directory.mkdir()
    runtime.job = {"state": "running"}
    monkeypatch.setattr("local_vision_mcp.runtime.platform.system", lambda: "Linux")
    monkeypatch.setattr("local_vision_mcp.runtime.platform.machine", lambda: "x86_64")
    compressed = zstandard.ZstdCompressor().compress(archive_bytes())
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200, content=compressed)

    client = httpx.Client
    monkeypatch.setattr(
        "local_vision_mcp.runtime.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    runtime.install()
    selected = (runtime.directory / "current").resolve()
    assert (selected / "bin/ollama").stat().st_mode & 0o111
    assert requests == ["https://ollama.com/download/ollama-linux-amd64.tar.zst"]
    compressed = b"invalid archive"
    with pytest.raises(zstandard.ZstdError):
        runtime.install()
    assert (runtime.directory / "current").resolve() == selected
    assert len(list(runtime.directory.glob("release-*"))) == 1


def test_ensure_deduplicates_and_reuses_external_service(runtime, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(runtime, "version", lambda: "test")
    monkeypatch.setattr(runtime, "install", lambda: pytest.fail("Unexpected install"))
    monkeypatch.setattr(runtime, "start", lambda: pytest.fail("Unexpected start"))

    def pull():
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(runtime, "pull", pull)
    first = runtime.launch("ensure")
    assert entered.wait(3)
    second = runtime.launch("ensure")
    assert first["id"] == second["id"]
    release.set()
    assert wait_job(runtime)["state"] == "succeeded"
    assert not runtime.status()["owns_server"]


def test_update_refuses_to_restart_external_service(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda: "old")
    monkeypatch.setattr(runtime, "install", lambda: pytest.fail("Unexpected install"))
    runtime.launch("update")
    job = wait_job(runtime)
    assert job["state"] == "failed"
    assert "another process" in job["error"]


def test_update_owned_runtime_orders_install_restart_pull(runtime, monkeypatch):
    calls = []
    runtime.process = type("Process", (), {"poll": lambda self: None})()
    monkeypatch.setattr(runtime, "version", lambda: "old")

    def prepare(activate=True):
        calls.append("install")
        release = runtime.directory / "new-release"
        release.mkdir()
        return release

    monkeypatch.setattr(runtime, "install", prepare)
    monkeypatch.setattr(runtime, "healthcheck", lambda: calls.append("healthcheck"))
    for name in ("stop", "start", "pull"):
        monkeypatch.setattr(runtime, name, lambda name=name: calls.append(name))
    runtime.launch("update")
    assert wait_job(runtime)["state"] == "succeeded"
    assert calls == ["install", "stop", "start", "pull", "healthcheck"]
    runtime.process = None


def test_failed_update_healthcheck_rolls_back_to_previous_release(runtime, monkeypatch):
    runtime.directory.mkdir()
    previous = runtime.directory / "previous"
    candidate = runtime.directory / "candidate"
    previous.mkdir()
    candidate.mkdir()
    runtime.activate(previous)
    runtime.process = type("Process", (), {"poll": lambda self: None})()
    monkeypatch.setattr(runtime, "version", lambda: "old")
    monkeypatch.setattr(runtime, "install", lambda activate=True: candidate)
    started = []
    monkeypatch.setattr(
        runtime, "start", lambda: started.append((runtime.directory / "current").resolve())
    )
    monkeypatch.setattr(runtime, "stop", lambda: None)
    monkeypatch.setattr(runtime, "pull", lambda: None)

    def fail_health():
        raise RuntimeError("candidate is unhealthy")

    monkeypatch.setattr(runtime, "healthcheck", fail_health)
    runtime.launch("update")
    assert wait_job(runtime)["state"] == "failed"
    assert (runtime.directory / "current").resolve() == previous
    assert started == [candidate, previous]
    runtime.process = None


@pytest.mark.parametrize("line,state", [("success", "succeeded"), ("downloading", "failed")])
def test_pull_requires_success(runtime, monkeypatch, line, state):
    def respond(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        assert request.url.path == "/api/pull"
        assert json.loads(request.content)["model"] == runtime.config["vision_model"]
        return httpx.Response(200, text=json.dumps({"status": line}) + "\n")

    client = httpx.Client
    monkeypatch.setattr(
        "local_vision_mcp.runtime.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    runtime.launch("ensure")
    assert wait_job(runtime)["state"] == state


def test_owned_process_starts_and_stops(runtime, tmp_path, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = tmp_path / "ollama"
    script.write_text(
        f"#!{sys.executable}\n"
        "import http.server, os, urllib.parse\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  self.send_response(200); self.end_headers(); "
        'self.wfile.write(b\'{"version":"test"}\')\n'
        "url = urllib.parse.urlparse(os.environ['OLLAMA_HOST'])\n"
        "http.server.HTTPServer((url.hostname, url.port), Handler).serve_forever()\n"
    )
    script.chmod(0o755)
    runtime.url = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(Runtime, "binary", property(lambda self: str(script)))
    runtime.directory.mkdir()
    runtime.job = {"state": "running"}
    runtime.start()
    process = runtime.process
    assert runtime.status()["owns_server"]
    recovered = processes.recover(runtime.directory / "ollama-owner.json")
    assert recovered is not None and recovered.pid == process.pid
    runtime.close()
    assert process.poll() is not None


def test_management_tool_roundtrip(tmp_path):
    pulls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version":"test"}')

        def do_POST(self):
            if self.path == "/api/show":
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"capabilities":["vision"]}')
                return
            pulls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"success"}\n')

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ollama_url": f"http://127.0.0.1:{backend.server_port}"}))

    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "local_vision_mcp"],
            env={
                "LOCAL_VISION_RUNTIME_DIR": str(tmp_path / "runtime"),
                "LOCAL_VISION_CONFIG": str(config),
            },
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            invalid = await client.call_tool("local_media_manage", {"action": "shell"})
            assert invalid.isError
            status = await client.call_tool("local_media_status", {})
            data = json.loads(status.content[0].text)
            assert data["runtime"]["managed_directory"] == str(tmp_path / "runtime")
            assert data["runtime"]["job"] is None
            assert not list((tmp_path / "runtime").glob("release-*"))
            result = await client.call_tool("local_media_manage", {"action": "ensure"})
            assert not result.isError
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = await client.call_tool("local_media_status", {})
                job = json.loads(result.content[0].text)["runtime"]["job"]
                if job["state"] != "running":
                    assert job["state"] == "succeeded"
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("MCP management job timed out")
            assert len(pulls) == 1 and "model" in pulls[0]

    try:
        asyncio.run(check())
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join(3)


@pytest.mark.parametrize("arch", ["arm64", "x86_64"])
def test_macos_universal_install_keeps_runner_and_metal_layout(runtime, monkeypatch, arch):
    import gzip

    runtime.directory.mkdir()
    runtime.job = {"state": "running"}
    monkeypatch.setattr("local_vision_mcp.runtime.platform.system", lambda: "Darwin")
    monkeypatch.setattr("local_vision_mcp.runtime.platform.machine", lambda: arch)
    monkeypatch.setattr("local_vision_mcp.runtime.platform.mac_ver", lambda: ("14.0", (), ""))
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name in ("ollama", "llama-server", "mlx_metal_v3/libmlx.dylib"):
            member = tarfile.TarInfo(name)
            data = b"#!/bin/sh\nexit 0\n"
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(200, content=gzip.compress(payload.getvalue()))

    client = httpx.Client
    monkeypatch.setattr(
        "local_vision_mcp.runtime.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    release = runtime.install()
    assert requests == [
        "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz"
    ]
    assert runtime.binary == str(runtime.directory / "current/ollama")
    assert (release / "llama-server").is_file()
    assert (
        "mlx_metal_v3/libmlx.dylib" in json.loads((release / "manifest.json").read_text())["files"]
    )
    assert (release / "ollama").stat().st_mode & 0o111


def test_macos_unsupported_version_fails_before_download(runtime, monkeypatch):
    monkeypatch.setattr("local_vision_mcp.runtime.platform.system", lambda: "Darwin")
    monkeypatch.setattr("local_vision_mcp.runtime.platform.machine", lambda: "arm64")
    monkeypatch.setattr("local_vision_mcp.runtime.platform.mac_ver", lambda: ("13.7", (), ""))
    with pytest.raises(RuntimeError, match="macOS 14"):
        runtime.install()
    assert not runtime.directory.exists()
