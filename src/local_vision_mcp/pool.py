"""Bounded FIFO admission, cancellation and queue deadlines for expensive local jobs."""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass


class QueueFullError(RuntimeError):
    """The pending queue is full; the caller may retry with backoff."""


class QueueTimeoutError(TimeoutError):
    """The job expired before execution and was never sent to a model."""


@dataclass
class Job:
    run: Callable[[], dict]
    future: Future
    submitted: float
    deadline: float


class WorkerPool:
    def __init__(self, workers: int = 1, capacity: int = 16, queue_timeout: float = 120):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workers must be an integer between 1 and 32")
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("queue_capacity must be an integer between 1 and 4096")
        if not math.isfinite(queue_timeout) or queue_timeout <= 0:
            raise ValueError("queue_timeout must be finite and positive")
        self.workers, self.capacity, self.queue_timeout = workers, capacity, queue_timeout
        self.condition = threading.Condition()
        self.pending: deque[Job] = deque()
        self.closed = False
        self.counters = {
            "submitted": 0,
            "active": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "expired": 0,
            "cancelled": 0,
            "peak_active": 0,
            "peak_queued": 0,
        }
        self.threads = [
            threading.Thread(target=self._work, daemon=True, name=f"media-worker-{i}")
            for i in range(workers)
        ]
        self.threads.append(
            threading.Thread(target=self._expire, daemon=True, name="media-queue-deadlines")
        )
        for thread in self.threads:
            thread.start()

    def submit(self, run: Callable[[], dict]) -> Future:
        with self.condition:
            if self.closed:
                raise RuntimeError("Worker pool is closed")
            if len(self.pending) >= self.capacity:
                self.counters["rejected"] += 1
                raise QueueFullError("Media queue is full; retry with backoff")
            now = time.monotonic()
            job = Job(run, Future(), now, now + self.queue_timeout)
            self.pending.append(job)
            self.counters["submitted"] += 1
            self.counters["peak_queued"] = max(self.counters["peak_queued"], len(self.pending))
            job.future.add_done_callback(
                lambda future: self._cancel(job) if future.cancelled() else None
            )
            self.condition.notify_all()
            return job.future

    def _cancel(self, job: Job):
        with self.condition:
            if job in self.pending:
                self.pending.remove(job)
            self.counters["cancelled"] += 1
            self.condition.notify_all()

    def _expire(self):
        with self.condition:
            while not self.closed:
                now = time.monotonic()
                while self.pending and self.pending[0].deadline <= now:
                    job = self.pending.popleft()
                    if job.future.set_running_or_notify_cancel():
                        self.counters["expired"] += 1
                        job.future.set_exception(QueueTimeoutError("Media job expired in queue"))
                delay = max(0.001, self.pending[0].deadline - now) if self.pending else None
                self.condition.wait(delay)

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closed)
                if not self.pending:
                    return
                job = self.pending.popleft()
                if not job.future.set_running_or_notify_cancel():
                    continue
                started = time.monotonic()
                if started >= job.deadline:
                    self.counters["expired"] += 1
                    job.future.set_exception(QueueTimeoutError("Media job expired in queue"))
                    continue
                self.counters["active"] += 1
                self.counters["peak_active"] = max(
                    self.counters["peak_active"], self.counters["active"]
                )
            try:
                result = job.run()
                finished = time.monotonic()
                result = {
                    **result,
                    "queue_seconds": round(started - job.submitted, 6),
                    "service_seconds": round(finished - started, 6),
                    "elapsed_seconds": round(finished - job.submitted, 6),
                }
            except BaseException as exc:  # noqa: BLE001 - propagate failures through the Future
                with self.condition:
                    self.counters["failed"] += 1
                    self.counters["active"] -= 1
                job.future.set_exception(exc)
            else:
                with self.condition:
                    self.counters["completed"] += 1
                    self.counters["active"] -= 1
                job.future.set_result(result)

    def status(self) -> dict:
        with self.condition:
            return {
                **self.counters,
                "queued": len(self.pending),
                "workers": self.workers,
                "capacity": self.capacity,
                "closed": self.closed,
                "queue_timeout_seconds": self.queue_timeout,
            }

    def close(self, wait: bool = True):
        with self.condition:
            self.closed = True
            for job in list(self.pending):
                job.future.cancel()
            self.condition.notify_all()
        if wait:
            for thread in self.threads:
                thread.join()
