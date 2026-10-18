"""Generate fixtures and run repeatable model × worker × client benchmark matrices."""

import argparse
import datetime
import json
import math
import os
import subprocess
from pathlib import Path

from ..setup import hardware, recommend
from .fixtures import generate
from .runner import (
    environment,
    load_manifest,
    recommendations,
    run_scenario,
    workload_key,
    write_report,
)

PROFILES = {
    "native": {},
    "cpu-2-threads": {"ollama_options": {"num_gpu": 0, "num_thread": 2, "num_ctx": 4096}},
    "cpu-4-threads": {"ollama_options": {"num_gpu": 0, "num_thread": 4, "num_ctx": 4096}},
    "small-images": {
        "image_max_edge": 768,
        "video_max_edge": 512,
        "ollama_options": {"num_ctx": 4096},
    },
}


def counts(value, maximum):
    values = list(dict.fromkeys(int(v) for v in value.split(",")))
    if not values or any(not 1 <= v <= maximum for v in values):
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers in [1, {maximum}]")
    return values


def container_info(name):
    result = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=True, timeout=15
    )
    value = json.loads(result.stdout)[0]
    config = value["HostConfig"]
    return {
        "name": name,
        "image_id": value["Image"],
        "running": value["State"]["Running"],
        "oom_killed": value["State"]["OOMKilled"],
        "limits": {
            k: config.get(k)
            for k in (
                "Memory",
                "MemorySwap",
                "NanoCpus",
                "CpuQuota",
                "CpuPeriod",
                "CpusetCpus",
                "DeviceRequests",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixtures = sub.add_parser("fixtures", help="Create deterministic synthetic test media")
    fixtures.add_argument("--output", type=Path, required=True)
    fixtures.add_argument("--seed", type=int, default=42)
    fixtures.add_argument("--variants", type=int, default=3)
    fixtures.add_argument("--images-only", action="store_true")
    run = sub.add_parser("run", help="Benchmark already-installed models; never auto-download")
    run.add_argument("--manifest", type=Path, required=True)
    models = run.add_mutually_exclusive_group(required=True)
    models.add_argument("--models", nargs="+", help="Installed Ollama vision model tags")
    models.add_argument(
        "--matrix", type=Path, help="JSON model definitions, including optional Omni"
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--agents", type=lambda v: counts(v, 128), default=[1, 4])
    run.add_argument("--workers", type=lambda v: counts(v, 32), default=[1, 2])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--profiles", default="native", help="Comma-separated built-in profiles")
    run.add_argument(
        "--profile-file", type=Path, help="JSON object mapping profile names to service overrides"
    )
    run.add_argument("--ollama-url", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    run.add_argument("--queue-capacity", type=int, default=16)
    run.add_argument("--queue-timeout", type=float, default=120)
    run.add_argument("--inference-timeout", type=float, default=300)
    run.add_argument("--min-accuracy", type=float, default=0.9)
    run.add_argument("--max-p95", type=float, default=30)
    run.add_argument("--min-samples", type=int, default=10)
    run.add_argument("--kind", action="append", choices=["image", "video", "audio"])
    run.add_argument("--category", action="append")
    run.add_argument("--limit", type=int, help="Limit cases per model for a quick smoke test")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--container", help="Record actual Docker limits/image of the dedicated inference container"
    )
    args = parser.parse_args()
    if args.command == "fixtures":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Fixture output must be empty to preserve existing datasets")
        print(generate(args.output, args.seed, args.variants, not args.images_only))
        return
    if (
        args.min_samples < 1
        or not 1 <= args.repeats <= 1000
        or not 0 <= args.warmup <= 10
        or not 0 <= args.min_accuracy <= 1
        or not math.isfinite(args.max_p95)
        or args.max_p95 <= 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Invalid repeats, warmup, accuracy, latency or case limit")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory to preserve prior benchmark results")
    document, all_cases = load_manifest(args.manifest)
    candidates = (
        json.loads(args.matrix.read_text())["models"]
        if args.matrix
        else [
            {"name": model, "backend": "ollama", "kinds": ["image", "video"]}
            for model in args.models
        ]
    )
    profiles = (
        json.loads(args.profile_file.read_text())
        if args.profile_file
        else {name: PROFILES[name] for name in args.profiles.split(",")}
    )
    if not candidates or not profiles:
        parser.error("Model matrix and profile list must not be empty")
    for model in candidates:
        if model.get("backend", "ollama") not in ("ollama", "omni"):
            parser.error("Supported backends are ollama and omni")
        if model.get("backend", "ollama") == "ollama" and "audio" in model.get("kinds", []):
            parser.error("Ollama audio is not implemented; use Omni for audio cases")
    hw = hardware()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "version": 1,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": environment(),
        "hardware": hw,
        "manifest": document,
        "seed": args.seed,
        "load_mode": "closed-loop clients, no automatic retries",
        "resource_constraints": container_info(args.container)
        if args.container
        else {
            "enforcement": "external server limits unknown; profiles only request runtime settings"
        },
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "max_p95_seconds": args.max_p95,
            "min_samples": args.min_samples,
        },
        "trials": [],
        "recommendations": [],
    }
    with (args.output / "requests.jsonl").open("w") as log:
        for candidate in candidates:
            backend = candidate.get("backend", "ollama")
            kinds = candidate.get(
                "kinds", ["image", "video"] if backend == "ollama" else ["image", "video", "audio"]
            )
            cases = [
                c
                for c in all_cases
                if c["kind"] in kinds
                and (not args.kind or c["kind"] in args.kind)
                and (not args.category or c.get("category") in args.category)
                and (backend != "ollama" or not c.get("request", {}).get("include_audio"))
            ]
            if args.limit:
                cases = cases[: args.limit]
            if not cases:
                print(f"Skipping {candidate['name']}: no matching cases", flush=True)
                continue
            if len(cases) * args.repeats > 10000:
                parser.error("At most 10000 requests per trial are allowed")
            for profile_name, overrides in profiles.items():
                for workers in args.workers:
                    for agents in args.agents:
                        config = {
                            **recommend(hw),
                            **candidate.get("config", {}),
                            **overrides,
                            "vision_backend": backend,
                            "workers": workers,
                            "queue_capacity": args.queue_capacity,
                            "queue_timeout": args.queue_timeout,
                            "inference_timeout": args.inference_timeout,
                            "ollama_url": args.ollama_url,
                            "omni_enabled": backend == "omni",
                        }
                        config["omni_model" if backend == "omni" else "vision_model"] = candidate[
                            "name"
                        ]
                        config["ollama_options"] = {
                            **candidate.get("config", {}).get("ollama_options", {}),
                            **overrides.get("ollama_options", {}),
                        }
                        trial = {
                            "id": len(report["trials"]),
                            "model": candidate["name"],
                            "profile": profile_name,
                            "agents": agents,
                            "workers": workers,
                            "config": config,
                            "workload": workload_key(cases),
                            "case_ids": [c["id"] for c in cases],
                            "repeats": args.repeats,
                        }
                        print(
                            f"Trial {trial['id']}: {candidate['name']}, {profile_name}, "
                            f"{agents} agents / {workers} workers, {len(cases) * args.repeats} requests",
                            flush=True,
                        )

                        def on_row(row, trial_id=trial["id"]):
                            log.write(json.dumps({"trial_id": trial_id, **row}) + "\n")
                            log.flush()
                            print(
                                f"  {row['case_id']}: {'pass' if row['grade']['passed'] else 'FAIL'} "
                                f"({row['latency_seconds']:.2f}s)",
                                flush=True,
                            )

                        try:
                            trial.update(
                                run_scenario(
                                    config,
                                    cases,
                                    agents,
                                    args.repeats,
                                    args.warmup,
                                    args.seed,
                                    args.manifest.parent,
                                    on_row=on_row,
                                )
                            )
                        except Exception as exc:  # noqa: BLE001 - retain per-model failures in results
                            trial.update(
                                status="failed",
                                error={"type": type(exc).__name__, "message": str(exc)},
                            )
                            print(f"  Failed: {exc}", flush=True)
                        report["trials"].append(trial)
                        report["recommendations"] = recommendations(
                            report["trials"], args.min_accuracy, args.max_p95, args.min_samples
                        )
                        write_report(report, args.output)
    if args.container:
        report["resource_constraints_after"] = container_info(args.container)
    write_report(report, args.output)
    print(f"Report: {args.output / 'report.md'}")
    if not report["trials"] or all(t["status"] == "failed" for t in report["trials"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
