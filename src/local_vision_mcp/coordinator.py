"""Local-vision adapter for the reusable durable coordinator contract."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

from agentic_mcp_foundation import LifecycleError
from agentic_mcp_foundation.jobs import TERMINAL, JobEngine, JobStore, canonical
from agentic_mcp_foundation.transport import SharedClient, serve
from pydantic import BaseModel, ConfigDict, Field

from . import media


class MediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    path: str
    question: str = Field(min_length=1, max_length=8000)
    kind: Literal["image", "video", "audio"]
    max_tokens: int = Field(default=384, ge=64, le=2048)
    start: float = Field(default=0, ge=0)
    duration: float = Field(default=30, gt=0, le=120)
    frames: int = Field(default=6, ge=1, le=12)
    include_audio: bool = False
    crop: list[int] | None = None


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def build_identity():
    import agentic_mcp_foundation

    digest = hashlib.sha256()
    for root in (Path(__file__).parent, Path(agentic_mcp_foundation.__file__).parent):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return {
        "version": importlib.metadata.version("local-vision-mcp"),
        "build": digest.hexdigest(),
        "contract_major": 1,
    }


def scope(config, roots):
    return hashlib.sha256(canonical([config, [str(p) for p in roots]]).encode()).hexdigest()[:24]


class Coordinator:
    def __init__(self, config, roots, directory):
        from .server import Service

        self.service = Service(config, roots)
        self.directory = directory
        self.store = JobStore(directory / "jobs.sqlite3")
        self.store.recover()
        self.engine = JobEngine(
            self.store,
            {
                "analyze": self.analyze,
                "ensure": self.manage,
                "update": self.manage,
                "verify": self.verify,
            },
            workers=config.get("workers", 1),
            capacity=config.get("queue_capacity", 16) + config.get("workers", 1),
            exclusive=("ensure", "update", "verify"),
            queue_timeout=config.get("queue_timeout", 120),
        )

    def analyze(self, request, context):
        context.progress("checking media identity")
        request = dict(request)
        expected = request.pop("media_sha256")
        source = media.local_file(request["path"], self.service.roots)
        if digest_file(source) != expected:
            raise LifecycleError(
                "MEDIA_CHANGED", "Media changed after submission; submit a new key"
            )
        context.progress("analyzing")
        result = self.service.analyze(**request, expected_sha256=expected)
        return {**result, "media_sha256": expected, "producer": build_identity()}

    def manage(self, request, context):
        runtime = self.service.runtime
        runtime.job = {"action": request["action"], "state": "running", "error": None}
        runtime.progress_hook = context.progress
        try:
            if request["action"] == "ensure" and self.service.config["vision_backend"] == "omni":
                runtime.download_omni()
                runtime.job["state"] = "succeeded"
            else:
                runtime._run(request["action"], request.get("with_omni", False))
            context.check()
            if runtime.job["state"] != "succeeded":
                if runtime.job.get("failure"):
                    error = runtime.job["failure"]
                    raise LifecycleError(error["code"], error["message"], error["retryable"])
                raise RuntimeError(runtime.job["error"])
            return {"runtime": runtime.status(), "capabilities": runtime.capabilities()}
        finally:
            runtime.progress_hook = None

    def verify(self, request, context):
        from tempfile import TemporaryDirectory

        from agentic_mcp_foundation.transport import file_lock
        from PIL import Image

        context.progress("running real vision probe")
        with TemporaryDirectory(prefix="local-vision-probe-") as tmp:
            path = Path(tmp) / "red.png"
            Image.new("RGB", (96, 96), "red").save(path)
            raw, _ = media.image_bytes(path, None)
            with file_lock(self.service.runtime.usage_lock, blocking=False, shared=True):
                backend = (
                    self.service.omni
                    if self.service.config["vision_backend"] == "omni"
                    else self.service.vision
                )
                answer = backend.analyze("Name the dominant color in one word.", [raw], 64)
        passed = answer.strip().strip(". !'\"`").casefold() == "red"
        report = {
            "passed": passed,
            "answer": answer[:256],
            "model": self.service.config["omni_model"]
            if self.service.config["vision_backend"] == "omni"
            else self.service.vision.model,
            "producer": build_identity(),
            "timestamp": time.time(),
            "scope": "single color probe; not an accuracy qualification",
        }
        if not passed:
            raise LifecycleError("PROBE_FAILED", "Vision probe did not identify the red image")
        return report

    def dispatch(self, operation, payload):
        if operation == "status":
            self.engine.expire_queued()
            result = self.service.status()
            recent = self.store.recent()
            management = next(
                (job for job in recent if job["operation"] in ("ensure", "update")), None
            )
            result["runtime"]["job"] = management
            return {
                **result,
                "producer": build_identity(),
                "jobs": recent,
                "coordinator_directory": str(self.directory),
                "capabilities": self.service.runtime.capabilities(),
            }
        if operation == "submit":
            kind = payload["operation"]
            request = payload["request"]
            if kind == "analyze":
                request = MediaRequest.model_validate(request).model_dump()
                source = media.local_file(request["path"], self.service.roots)
                request["path"] = str(source)
                request["media_sha256"] = digest_file(source)
            elif kind in ("ensure", "update"):
                if (
                    set(request) - {"with_omni"}
                    or type(request.get("with_omni", False)) is not bool
                ):
                    raise ValueError("Only boolean with_omni is accepted")
                request = {"action": kind, "with_omni": request.get("with_omni", False)}
            elif kind == "verify":
                if request:
                    raise ValueError("verify takes no request arguments")
            else:
                raise LifecycleError("UNKNOWN_OPERATION", "Operation is not registered")
            return self.engine.submit(kind, request, payload["idempotency_key"])
        if operation == "job":
            return self.engine.get(payload["id"])
        if operation == "cancel":
            return self.engine.cancel(payload["id"])
        if operation == "retire":
            if payload.get("stop_backend", False):
                from agentic_mcp_foundation.transport import file_lock

                with file_lock(self.service.runtime.usage_lock, blocking=False):
                    self.engine.drain()
                    self.service.runtime.stop()
            else:
                self.engine.drain()
            return {"retired": True}
        raise LifecycleError("UNKNOWN_OPERATION", "RPC operation is not registered")


class RemoteService(SharedClient):
    """MCP façade: disconnecting this object does not shut down the shared coordinator."""

    def __init__(self, config, roots):
        from .runtime import Runtime

        self.config, self.roots = config, roots
        runtime = Runtime(
            config, config.get("ollama_url", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        )
        self.directory = runtime.directory / "coordinators" / scope(config, roots)
        super().__init__(
            self.directory,
            build_identity(),
            [sys.executable, "-m", "local_vision_mcp.coordinator", str(self.directory)],
            {"config": config, "roots": [str(p) for p in roots]},
        )

    def connect(self):
        home = os.getenv("AGENTIC_MCP_HOME")
        if home:
            active = json.loads((Path(home) / "active.json").read_text())
            if active["acceptance"]["producer"]["build"] != self.identity["build"]:
                raise LifecycleError(
                    "GENERATION_CHANGED",
                    "Refresh the MCP connection to load "
                    "the active installed payload; old work remains recoverable",
                )
        super().connect()

    def submit_job(self, request, idempotency_key):
        return self.call(
            "submit",
            {"operation": "analyze", "request": request, "idempotency_key": idempotency_key},
        )

    def manage(self, action, with_omni=False, idempotency_key=None):
        return self.call(
            "submit",
            {
                "operation": action,
                "request": {"with_omni": with_omni},
                "idempotency_key": idempotency_key or uuid.uuid4().hex,
            },
        )

    def get_job(self, job_id):
        return self.call("job", {"id": job_id})

    def cancel_job(self, job_id):
        return self.call("cancel", {"id": job_id})

    def status(self):
        return self.call("status")

    async def analyze_async(
        self,
        path,
        question,
        kind,
        max_tokens=384,
        start=0,
        duration=30,
        frames=6,
        include_audio=False,
        crop=None,
    ):
        import asyncio

        request = {
            "path": path,
            "question": question,
            "kind": kind,
            "max_tokens": max_tokens,
            "start": start,
            "duration": duration,
            "frames": frames,
            "include_audio": include_audio,
            "crop": crop,
        }
        job = await asyncio.to_thread(self.submit_job, request, uuid.uuid4().hex)
        while job["state"] not in TERMINAL:
            await asyncio.sleep(0.1)
            job = await asyncio.to_thread(self.get_job, job["id"])
        if job["state"] != "succeeded":
            error = job["error"] or {"code": "CANCELLED", "message": "Job cancelled"}
            raise LifecycleError(error["code"], error["message"])
        return job["result"]

    def close(self, wait=False):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    settings = json.loads((args.directory / "settings.json").read_text())
    # Job recovery must occur after obtaining exclusive process ownership.
    coordinator = None
    initialization = threading.Lock()

    def dispatch(operation, payload):
        nonlocal coordinator
        with initialization:
            if coordinator is None:
                coordinator = Coordinator(
                    settings["config"], tuple(Path(p) for p in settings["roots"]), args.directory
                )
        return coordinator.dispatch(operation, payload)

    serve(
        args.directory,
        build_identity(),
        dispatch,
        lambda: coordinator.service.close(stop_runtime=False) if coordinator else None,
    )


if __name__ == "__main__":
    main()
