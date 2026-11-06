#!/usr/bin/env python3
"""Opt-in installed runtime acceptance. Downloads Ollama/models; never enables a host plugin."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True, help="Dedicated test installation")
    parser.add_argument("--reuse-models", type=Path)
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--provision", choices=["update", "ensure"], default="update")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    home = args.home.resolve()
    if not (home / "active.json").is_file():
        raise ValueError("Install an isolated runtime first")
    models = home / "acceptance-models"
    models.mkdir(exist_ok=True)
    (models / "blobs").mkdir(exist_ok=True)
    if args.reuse_models:
        source = args.reuse_models.resolve()
        for blob in (source / "blobs").iterdir():
            target = models / "blobs" / blob.name
            if not target.exists():
                target.symlink_to(blob)
        if not (models / "manifests").exists():
            shutil.copytree(source / "manifests", models / "manifests")
    env = {**os.environ, "OLLAMA_MODELS": str(models), "LOCAL_VISION_HOME": str(home)}
    env.pop("LOCAL_VISION_RUNTIME_DIR", None)
    env.pop("LOCAL_VISION_CONFIG", None)
    env.pop("LOCAL_VISION_ROOTS", None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings_path = home / "settings.json"
    original = settings_path.read_bytes()
    settings = json.loads(original)
    settings["config"].update(
        ollama_url=f"http://127.0.0.1:{port}", vision_model=args.model, omni_enabled=False
    )
    settings_path.write_text(json.dumps(settings))
    launcher = home / "control/launcher.py"

    def command(*arguments):
        result = subprocess.run(
            [sys.executable, str(launcher), "--cli", *arguments],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(result.stdout)

    def wait(job):
        deadline = time.monotonic() + 1800
        phase = None
        while job["state"] in ("queued", "running"):
            if job["phase"] != phase:
                phase = job["phase"]
                print(phase, file=sys.stderr, flush=True)
            if time.monotonic() > deadline:
                command("cancel", job["id"])
                raise TimeoutError(
                    "Live acceptance timed out; inspect retained job before retrying"
                )
            time.sleep(1)
            job = command("job", job["id"])
        if job["state"] != "succeeded":
            raise RuntimeError(json.dumps(job["error"]))
        return job

    report = {"live": True, "model": args.model, "passed": False}
    try:
        report["provision"] = wait(
            command(args.provision, "--key", "live-provision-" + str(time.time_ns()))
        )
        report["probe"] = wait(command("verify", "--key", "live-probe-" + str(time.time_ns())))
        status = command("doctor")
        report.update(
            passed=True,
            producer=status["producer"],
            capabilities=status["capabilities"],
            ollama_version=status["runtime"]["ollama_version"],
        )
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        try:
            command("stop", "--backend")
        finally:
            settings_path.write_bytes(original)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "ollama_version": report["ollama_version"],
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
