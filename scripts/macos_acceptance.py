#!/usr/bin/env python3
"""Run clean-checkout MacBook acceptance, with opt-in real model provisioning."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home", type=Path, help="Dedicated test installation; defaults to a new temp directory"
    )
    parser.add_argument(
        "--root", type=Path, action="append", help="Allowed media root; repeat as needed"
    )
    parser.add_argument(
        "--live", action="store_true", help="Download Ollama/models and run actual vision inference"
    )
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--reuse-models", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Run this acceptance on the MacBook; this host is not macOS")
    if int(platform.mac_ver()[0].split(".")[0]) < 14:
        parser.error("macOS 14 or newer is required")
    for command in ("uv", "git", "ffmpeg", "ffprobe"):
        if not shutil.which(command):
            parser.error(f"Missing {command}; see docs/macos-validation.md")
    source = Path(__file__).resolve().parents[1]
    home = (
        (args.home or Path(tempfile.mkdtemp(prefix="local-vision-macbook-"))).expanduser().resolve()
    )
    if (home / "active.json").exists():
        parser.error("Use a fresh dedicated --home to avoid replacing another installation")
    home.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "home": str(home),
        "passed": False,
        "backend_inference_tested": False,
        "checks": [],
    }

    def check(name, command, timeout=1800):
        print(f"Running {name}", flush=True)
        started = time.monotonic()
        result = subprocess.run(command, check=False, cwd=source, timeout=timeout)
        report["checks"].append(
            {
                "name": name,
                "exit_code": result.returncode,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
        result.check_returncode()

    try:
        check(
            "locked dependencies", ["uv", "sync", "--locked", "--extra", "dev", "--extra", "bench"]
        )
        check("lint", ["uv", "run", "ruff", "check", "src", "tests", "scripts"])
        check("format", ["uv", "run", "ruff", "format", "--check", "src", "tests", "scripts"])
        check("tests", ["uv", "run", "pytest", "-q"], timeout=300)
        command = ["uv", "run", "local-vision", "--home", str(home), "install"]
        for root in args.root or [source]:
            command += ["--root", str(root.expanduser().resolve(strict=True))]
        check("installed plugin", command)
        check(
            "offline launcher doctor",
            [sys.executable, str(home / "control/launcher.py"), "--doctor"],
        )
        if args.live:
            command = [
                sys.executable,
                str(source / "scripts/live_acceptance.py"),
                "--home",
                str(home),
                "--model",
                args.model,
                "--output",
                str(home / "live-report.json"),
            ]
            if args.reuse_models:
                command += ["--reuse-models", str(args.reuse_models.expanduser().resolve())]
            check("real vision probe", command, timeout=2400)
            report["backend_inference_tested"] = True
            report["live"] = json.loads((home / "live-report.json").read_text())
        report["passed"] = True
    finally:
        path = home / "macbook-acceptance.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Acceptance report: {path}", flush=True)


if __name__ == "__main__":
    main()
