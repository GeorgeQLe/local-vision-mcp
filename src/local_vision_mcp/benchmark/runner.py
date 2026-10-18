"""Run quality/load matrices through the same Service and queue used by MCP."""

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from ..server import Service
from .scoring import score


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    document = json.loads(path.read_text())
    if document.get("version") != 1 or not document.get("cases"):
        raise ValueError("Expected a version 1 manifest with nonempty cases")
    ids, cases = set(), []
    for original in document["cases"]:
        case = dict(original)
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        if case["kind"] not in ("image", "video", "audio"):
            raise ValueError(f"Invalid kind: {case['kind']}")
        if (
            not isinstance(case.get("rubric", {}).get("expected"), dict)
            or not case["rubric"]["expected"]
        ):
            raise ValueError("Each case needs a nonempty rubric.expected JSON object")
        request_keys = {"max_tokens", "start", "duration", "frames", "include_audio", "crop"}
        if set(case.get("request", {})) - request_keys:
            raise ValueError("Unknown case request option")
        source = (path.parent / case["path"]).resolve(strict=True)
        if not source.is_relative_to(path.parent.resolve()) or not source.is_file():
            raise ValueError("Benchmark media must be regular files inside the manifest directory")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if case.get("sha256") and case["sha256"] != digest:
            raise ValueError(f"Media hash mismatch: {case['id']}")
        case.update(path=str(source), sha256=digest)
        cases.append(case)
    return document, cases


def ollama_metadata(service: Service) -> dict:
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.post(
            service.vision.url + "/api/show", json={"model": service.vision.model}
        )
        if response.is_error:
            raise RuntimeError(
                f"Ollama preflight failed ({response.status_code}): {response.text[:1000]}"
            )
        show = response.json()
        if show.get("remote_model") or show.get("remote_host"):
            raise ValueError("Benchmark requires local weights, not a cloud-backed Ollama model")
        if "vision" not in show.get("capabilities", []):
            raise ValueError("Ollama model does not advertise vision capability")
        tags = client.get(service.vision.url + "/api/tags")
        tags.raise_for_status()
        model = next(
            (
                m
                for m in tags.json()["models"]
                if m["name"] in (service.vision.model, service.vision.model + ":latest")
            ),
            {},
        )
        version = client.get(service.vision.url + "/api/version")
        version.raise_for_status()
        return {
            "version": version.json(),
            "model": model,
            "capabilities": show.get("capabilities"),
            "details": show.get("details"),
            "parameters": show.get("parameters"),
        }


class Monitor:
    """Separate wrapper RSS from Ollama-reported allocation; neither is GPU peak telemetry."""

    def __init__(self, url: str | None):
        self.url = url
        self.stop = threading.Event()
        self.data = {
            "wrapper_tree_peak_rss_bytes": None,
            "ollama_peak_allocated_bytes": None,
            "ollama_peak_allocated_vram_bytes": None,
            "samples": 0,
            "sample_errors": 0,
        }
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            process = None
        with httpx.Client(timeout=1, trust_env=False) as client:
            while not self.stop.is_set():
                self.data["samples"] += 1
                if process:
                    try:
                        rss = sum(
                            p.memory_info().rss
                            for p in [process, *process.children(recursive=True)]
                        )
                        self.data["wrapper_tree_peak_rss_bytes"] = max(
                            self.data["wrapper_tree_peak_rss_bytes"] or 0, rss
                        )
                    except (psutil.Error, OSError):
                        self.data["sample_errors"] += 1
                if self.url:
                    try:
                        result = client.get(self.url + "/api/ps")
                        result.raise_for_status()
                        models = result.json()["models"]
                        for field, key in [
                            ("size", "ollama_peak_allocated_bytes"),
                            ("size_vram", "ollama_peak_allocated_vram_bytes"),
                        ]:
                            value = sum(m.get(field, 0) for m in models)
                            self.data[key] = max(self.data[key] or 0, value)
                    except (httpx.HTTPError, KeyError, ValueError):
                        self.data["sample_errors"] += 1
                self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()


def percentile(values: list[float], quantile: float):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def summarize(rows: list[dict], seconds: float) -> dict:
    completed = [row for row in rows if not row.get("error")]
    passed = sum(row["grade"]["passed"] for row in rows)
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "count": len(subset),
            "pass_rate": sum(row["grade"]["passed"] for row in subset) / len(subset),
        }
    return {
        "requests": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "pass_rate": passed / len(rows),
        "mean_field_score": statistics.mean(row["grade"]["score"] for row in rows),
        "p50_latency_seconds": percentile([r["latency_seconds"] for r in rows], 0.5),
        "p95_latency_seconds": percentile([r["latency_seconds"] for r in rows], 0.95),
        "p95_success_latency_seconds": percentile([r["latency_seconds"] for r in completed], 0.95),
        "p95_queue_seconds": percentile([r["result"]["queue_seconds"] for r in completed], 0.95),
        "completed_per_second": len(completed) / seconds,
        "correct_per_second": passed / seconds,
        "wall_seconds": seconds,
        "mean_report_tokens_estimate": statistics.mean(
            r["report_tokens_estimate"] for r in completed
        )
        if completed
        else None,
        "by_category": categories,
    }


def run_scenario(
    config: dict,
    cases: list[dict],
    agents: int,
    repeats: int,
    warmup: int,
    seed: int,
    root: Path,
    on_row=None,
    preflight=True,
    service_factory=Service,
) -> dict:
    service = service_factory(config, (root.resolve(),))
    rows, warmups = [], []
    try:
        backend = (
            ollama_metadata(service) if preflight and config["vision_backend"] == "ollama" else {}
        )
        for i in range(warmup):
            case = cases[i % len(cases)]
            started = time.monotonic()
            result = service.analyze(
                case["path"], case["question"], case["kind"], **case.get("request", {})
            )
            warmups.append(
                {
                    "case_id": case["id"],
                    "seconds": time.monotonic() - started,
                    "inference": result.get("inference", {}),
                }
            )
        jobs = [(case, repeat) for repeat in range(repeats) for case in cases]
        random.Random(seed).shuffle(jobs)

        def call(job):
            case, repeat = job
            begun = time.monotonic()
            row = {
                "case_id": case["id"],
                "category": case.get("category", case["kind"]),
                "kind": case["kind"],
                "repeat": repeat,
            }
            try:
                result = service.analyze(
                    case["path"], case["question"], case["kind"], **case.get("request", {})
                )
                grade = score(result["answer"], case["rubric"])
                if result["truncated"]:
                    grade.update(score=0.0, passed=False, reason="truncated")
                row.update(
                    result=result,
                    grade=grade,
                    report_tokens_estimate=math.ceil(len(json.dumps(result)) / 4),
                )
            except Exception as exc:  # noqa: BLE001 - failed requests are benchmark observations
                row.update(
                    error={"type": type(exc).__name__, "message": str(exc)},
                    grade={"score": 0.0, "passed": False, "reason": "request_error"},
                )
            row["latency_seconds"] = time.monotonic() - begun
            return row

        url = service.vision.url if config["vision_backend"] == "ollama" else None
        with Monitor(url) as monitor:
            begun = time.monotonic()
            with ThreadPoolExecutor(max_workers=agents) as callers:
                futures = [callers.submit(call, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    if on_row:
                        on_row(row)
            elapsed = time.monotonic() - begun
        return {
            "status": "completed",
            "summary": summarize(rows, elapsed),
            "resources": monitor.data,
            "pool": service.pool.status(),
            "warmup": warmups,
            "backend": backend,
            "rows": rows,
        }
    finally:
        service.close()


def workload_key(cases: list[dict]) -> str:
    # Content, question, grading and request settings all affect comparability.
    normalized = [{k: v for k, v in case.items() if k != "path"} for case in cases]
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def recommendations(
    trials: list[dict], min_accuracy: float, max_p95: float, min_samples: int = 10
) -> list[dict]:
    """Never rank disjoint modalities/case subsets or failed runs as equivalent."""
    groups = {}
    for trial in trials:
        if trial["status"] != "completed":
            continue
        summary = trial["summary"]
        if (
            summary["requests"] < min_samples
            or summary["errors"]
            or summary["pass_rate"] < min_accuracy
            or summary["p95_latency_seconds"] > max_p95
        ):
            continue
        key = (trial["workload"], trial["profile"], trial["agents"])
        groups.setdefault(key, []).append(trial)
    return [
        {
            "workload": key[0],
            "profile": key[1],
            "agents": key[2],
            "trial_id": max(
                group, key=lambda t: (t["summary"]["correct_per_second"], t["summary"]["pass_rate"])
            )["id"],
        }
        for key, group in groups.items()
    ]


def environment() -> dict:
    versions = {}
    for package in (
        "local-vision-mcp",
        "mcp",
        "httpx",
        "pillow",
        "torch",
        "transformers",
        "psutil",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
    }


def write_report(report: dict, output: Path):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "report.json")
    lines = [
        "# Local media benchmark",
        "",
        "Synthetic diagnostics are not a general model ranking. Compare matching workload, profile and agent count.",
        "",
        "| Trial | Model | Profile | Agents / workers | Pass | Errors | p95 seconds | Correct/sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial in report["trials"]:
        model = trial["model"].replace("|", "/")
        if trial["status"] != "completed":
            lines.append(
                f"| {trial['id']} | {model} | {trial['profile']} | {trial['agents']} / {trial['workers']} | failed | — | — | — |"
            )
            continue
        s = trial["summary"]
        lines.append(
            f"| {trial['id']} | {model} | {trial['profile']} | {trial['agents']} / {trial['workers']} | "
            f"{s['pass_rate']:.1%} | {s['errors']} | {s['p95_latency_seconds']:.3f} | {s['correct_per_second']:.3f} |"
        )
    lines += [
        "",
        "## Qualified configurations",
        "",
        (
            f"Minimum pass rate: {report['thresholds']['min_accuracy']:.0%}; "
            f"maximum p95 latency: {report['thresholds']['max_p95_seconds']} seconds; minimum samples: {report['thresholds'].get('min_samples', 10)}; zero request errors required."
        ),
        "",
    ]
    if not report["recommendations"]:
        lines.append("No configuration met the thresholds. There is no recommended winner.")
    for item in report["recommendations"]:
        trial = next(t for t in report["trials"] if t["id"] == item["trial_id"])
        (output / f"config-trial-{trial['id']}.json").write_text(
            json.dumps(trial["config"], indent=2) + "\n"
        )
        lines.append(
            f"- Trial {item['trial_id']}: profile {item['profile']}, {item['agents']} agents, workload {item['workload'][:12]}."
        )
    lines += [
        "",
        "Report tokens are a character-based estimate, not API billing. Local model token counts describe local inference only.",
        "Ollama allocation samples cover all loaded models on that endpoint, not exclusive per-request or physical peak VRAM.",
        "See report.json for raw answers, scoring, runtime identity, settings, warmups and failures.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
