import asyncio
import threading

import pytest
from PIL import Image

from local_vision_mcp.pool import QueueFullError, QueueTimeoutError, WorkerPool
from local_vision_mcp.server import Service
from local_vision_mcp.setup import recommend


def test_fifo_capacity_cancellation_and_failure_recovery():
    pool = WorkerPool(workers=1, capacity=2)
    entered, release = threading.Event(), threading.Event()
    order = []

    def blocking():
        entered.set()
        assert release.wait(3)
        return {"value": "first"}

    try:
        active = pool.submit(blocking)
        assert entered.wait(1)
        cancelled = pool.submit(lambda: order.append("cancelled") or {})
        second = pool.submit(lambda: order.append("second") or {})
        with pytest.raises(QueueFullError):
            pool.submit(dict)
        assert cancelled.cancel()
        third = pool.submit(lambda: order.append("third") or {})
        release.set()
        assert active.result(2)["value"] == "first"
        second.result(2)
        third.result(2)
        assert order == ["second", "third"]

        def fail():
            raise ValueError("deliberate failure")

        with pytest.raises(ValueError):
            pool.submit(fail).result(2)
        assert pool.submit(lambda: {"ok": True}).result(2)["ok"]
        assert pool.status()["cancelled"] == 1
        assert pool.status()["rejected"] == 1
        assert pool.status()["failed"] == 1
    finally:
        release.set()
        pool.close()
    assert all(not t.is_alive() for t in pool.threads)
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit(dict)


def test_pending_deadline_expires_while_worker_is_busy():
    pool = WorkerPool(workers=1, capacity=2, queue_timeout=0.1)
    entered, release, should_not_run = threading.Event(), threading.Event(), threading.Event()
    try:
        active = pool.submit(lambda: entered.set() or release.wait(3) and {})
        assert entered.wait(1)
        pending = pool.submit(lambda: should_not_run.set() or {})
        with pytest.raises(QueueTimeoutError):
            pending.result(1)
        assert not should_not_run.is_set()
        assert pool.status()["queued"] == 0
        release.set()
        active.result(2)
    finally:
        release.set()
        pool.close()


def test_pool_concurrency_is_bounded():
    pool = WorkerPool(workers=2, capacity=8)
    barrier = threading.Barrier(3)
    release = threading.Event()

    def hold():
        barrier.wait(timeout=2)
        assert release.wait(2)
        return {}

    try:
        running = [pool.submit(hold) for _ in range(2)]
        barrier.wait(timeout=2)
        pending = [pool.submit(dict) for _ in range(4)]
        assert pool.status()["active"] == 2
        assert pool.status()["queued"] == 4
        release.set()
        for result in running + pending:
            assert result.result(2)["elapsed_seconds"] >= result.result()["service_seconds"]
        assert pool.status()["peak_active"] == 2
    finally:
        release.set()
        pool.close()


def test_shutdown_cancels_pending_but_retains_active_slot():
    pool = WorkerPool()
    entered, release = threading.Event(), threading.Event()
    try:
        active = pool.submit(lambda: entered.set() or release.wait(3) and {})
        assert entered.wait(1)
        pending = pool.submit(dict)
        pool.close(wait=False)
        assert pending.cancelled() and pool.status()["active"] == 1
        release.set()
        active.result(2)
    finally:
        release.set()
        pool.close()


def test_cancelled_async_request_is_removed_without_model_call(tmp_path, monkeypatch):
    image = tmp_path / "test.png"
    Image.new("RGB", (32, 32)).save(image)
    cfg = recommend({"ram_gb": 32, "cuda_vram_gb": 0})
    service = Service(cfg, (tmp_path,))
    entered, release = threading.Event(), threading.Event()
    calls = []

    def infer(*args):
        calls.append(args)
        entered.set()
        assert release.wait(3)
        return "ok"

    monkeypatch.setattr(service.vision, "analyze", infer)

    async def exercise():
        running = asyncio.create_task(service.analyze_async(str(image), "Read", "image"))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(service.analyze_async(str(image), "Read", "image"))
        await asyncio.sleep(0)
        assert service.pool.status()["queued"] == 1
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert service.pool.status()["queued"] == 0
        release.set()
        await running
        assert len(calls) == 1

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        service.close()
