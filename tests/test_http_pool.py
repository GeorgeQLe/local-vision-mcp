import asyncio
import json
import socket
import threading
import time
from datetime import timedelta

import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

from local_vision_mcp.server import Service, create_server
from local_vision_mcp.setup import recommend


def test_two_http_clients_share_pool_and_disconnect_does_not_close_it(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(path)
    config = recommend({"ram_gb": 32, "cuda_vram_gb": 0})
    config["workers"] = 2
    service = Service(config, (tmp_path,))
    lock = threading.Lock()
    both_started, release = threading.Event(), threading.Event()
    started = 0

    def infer(*args):
        nonlocal started
        with lock:
            started += 1
            if started >= 2:
                both_started.set()
        assert release.wait(10)
        return "visible"

    monkeypatch.setattr(service.vision, "analyze", infer)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = create_server(service, port=port).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started

    async def exercise():
        url = f"http://127.0.0.1:{port}/mcp"
        async with (
            streamable_http_client(url) as (read_b, write_b, _),
            ClientSession(read_b, write_b, read_timeout_seconds=timedelta(seconds=15)) as b,
        ):
            await b.initialize()
            async with (
                streamable_http_client(url) as (read_a, write_a, _),
                ClientSession(read_a, write_a, read_timeout_seconds=timedelta(seconds=15)) as a,
            ):
                await a.initialize()
                jobs = [
                    asyncio.create_task(client.call_tool("analyze_image", {"path": str(path)}))
                    for client in (a, b, a)
                ]
                assert await asyncio.to_thread(both_started.wait, 3)
                until = time.monotonic() + 3
                while service.pool.status()["queued"] != 1 and time.monotonic() < until:
                    await asyncio.sleep(0.01)
                # This succeeds while both inference workers are blocked.
                status = await b.call_tool("local_media_status", {})
                pool = json.loads(status.content[0].text)["pool"]
                assert pool["active"] == 2 and pool["queued"] == 1
                release.set()
                assert all(not result.isError for result in await asyncio.gather(*jobs))
            # Client A's session shutdown must not destroy B's shared service.
            result = await b.call_tool("analyze_image", {"path": str(path)})
            assert not result.isError
            assert not service.pool.status()["closed"]

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        server.should_exit = True
        thread.join(10)
        sock.close()
        service.close()
    assert not thread.is_alive()
