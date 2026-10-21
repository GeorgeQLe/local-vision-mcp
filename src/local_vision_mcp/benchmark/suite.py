"""Comparable local/native-CLI evaluations with durable results and a live dashboard."""

import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..structured import validate_answer
from .dashboard import Dashboard, aggregate
from .providers import make_provider, token_cost
from .runner import load_manifest
from .scoring import score


def add_commands(sub):
    compare = sub.add_parser(
        "compare", help="Compare local models and native Codex/Claude CLI vision"
    )
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument(
        "--config", type=Path, required=True, help="JSON models, pricing and token estimates"
    )
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument(
        "--parallel", type=int, default=1, help="Concurrent cases; default avoids GPU contention"
    )
    compare.add_argument("--timeout", type=float, default=300)
    compare.add_argument("--limit", type=int)
    compare.add_argument("--kind", action="append", choices=["image", "video", "audio"])
    compare.add_argument("--category", action="append")
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument(
        "--max-cost", type=float, help="Soft USD dispatch guard; requires per-model token forecasts"
    )
    compare.add_argument(
        "--min-accuracy", type=float, default=0.9, help="Exit 1 if any model falls below this gate"
    )
    demo = sub.add_parser(
        "demo", help="Show the live dashboard using simulated results; no model calls"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--delay", type=float, default=0.15)
    for parser in (compare, demo):
        tui = parser.add_mutually_exclusive_group()
        tui.add_argument(
            "--tui", dest="tui", action="store_const", const="on", help="Force Rich live dashboard"
        )
        tui.add_argument(
            "--no-tui",
            dest="tui",
            action="store_const",
            const="off",
            help="Plain progress lines for CI/logs",
        )
        parser.set_defaults(tui="auto")


def load_models(path):
    document = json.loads(path.read_text())
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError("Config needs a nonempty models array")
    names = set()
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Invalid model definition: expected a JSON object")  # noqa: TRY004
        if model.get("backend") not in ("ollama", "omni", "codex", "claude"):
            raise ValueError("Supported backends: ollama, omni, codex, claude")
        if not all(
            isinstance(model.get(key), str) and model[key].strip() for key in ("name", "model")
        ):
            raise ValueError("Each model needs a nonempty name and model identifier")
        if model["name"] in names:
            raise ValueError("Model names must be unique")
        names.add(model["name"])
        if model.get("perception", "native") not in ("native", "mcp", "shadow"):
            raise ValueError("perception must be native, mcp or shadow")
        if model.get("local_strategy", "single") not in ("single", "consensus"):
            raise ValueError("local_strategy must be single or consensus")
        if model.get("local_strategy") == "consensus" and model.get("perception") != "mcp":
            raise ValueError("Consensus requires MCP perception")
        if model.get("perception") == "mcp" and (
            model["backend"] not in ("codex", "claude")
            or not isinstance(model.get("local_model"), str)
            or not model["local_model"].strip()
        ):
            raise ValueError("MCP perception requires a CLI backend and local_model")
        if model.get("perception") == "shadow" and (
            model["backend"] not in ("codex", "claude")
            or not isinstance(model.get("local_model"), str)
            or not isinstance(model.get("routing_policy"), str)
        ):
            raise ValueError("Shadow requires a native CLI, local_model and routing_policy")
        if "structured_output" in model and type(model["structured_output"]) is not bool:
            raise ValueError("structured_output must be boolean")
        for mapping in (model.get("pricing", {}), model.get("estimated_tokens", {})):
            if not isinstance(mapping, dict):
                raise ValueError("Invalid pricing/estimated_tokens: expected JSON objects")  # noqa: TRY004
            for value in mapping.values():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(
                        "Token forecasts and prices must be finite nonnegative numbers"
                    )
        pricing_keys = {"input", "cached_input", "cache_creation_input", "output"}
        if set(model.get("pricing", {})) - pricing_keys:
            raise ValueError("Unknown pricing key (rates must be USD per million tokens)")
        if set(model.get("estimated_tokens", {})) - {
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
        }:
            raise ValueError("Unknown estimated_tokens key")
        estimates = model.get("estimated_tokens", {})
        if estimates and estimates.get("input_tokens", 0) < (
            estimates.get("cached_input_tokens", 0)
            + estimates.get("cache_creation_input_tokens", 0)
        ):
            raise ValueError("Cached tokens cannot exceed total input tokens")
        if model.get("effort") and model["backend"] not in ("codex", "claude"):
            raise ValueError("effort applies only to native CLI backends")
    return models


def forecast(model):
    if model["backend"] in ("ollama", "omni"):
        return 0.0
    return token_cost(model.get("estimated_tokens"), model.get("pricing"))


def select_cases(cases, models, kinds=None, categories=None, limit=None):
    # Every model sees the same cases. No denominator silently changes by provider.
    audio_capable = all(model["backend"] == "omni" for model in models)
    if kinds and "audio" in kinds and not audio_capable:
        raise ValueError("Audio comparison currently requires all models to use Omni")
    selected = [
        case
        for case in cases
        if (not kinds or case["kind"] in kinds)
        and (not categories or case.get("category") in categories)
        and (
            audio_capable
            or (case["kind"] != "audio" and not case.get("request", {}).get("include_audio"))
        )
    ]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No mutually supported cases match the filters")
    return selected


def checkpoint(output, report):
    temp = output / "report.json.tmp"
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temp.replace(output / "report.json")
    lines = [
        "# Perception comparison",
        "",
        f"Status: {report['status']}",
        "",
        "**SIMULATED DEMO — not model measurements.**"
        if report.get("demo")
        else "Native CLI harnesses and local Service; this is not a raw API comparison.",
        "",
        "| Model | Completed | Pass | Field score | Errors | Estimated USD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["summary"].items():
        accuracy, fields = stats["accuracy"], stats["field_score"]
        lines.append(
            f"| {name.replace('|', '/')} | {stats['completed']} | "
            f"{'—' if accuracy is None else f'{accuracy:.1%}'} | "
            f"{'—' if fields is None else f'{fields:.1%}'} | {stats['errors']} | "
            f"${stats['known_cost_usd']:.4f} + {stats['unknown_cost_requests']} unknown |"
        )
    lines += [
        "",
        (
            "Costs are API-equivalent or CLI-reported estimates, not subscription invoices. "
            "Local hardware/electricity excluded. Unknown costs remain unknown. "
            "Forecasts and --max-cost are soft estimates, not provider spending caps."
        ),
        "",
        (
            "Errors count as failed cases. Correctness uses the manifest's deterministic rubric; "
            "synthetic fixtures do not establish broad perception parity."
        ),
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def evaluate(provider, model, case, repeat):
    started = time.monotonic()
    row = {
        "model_name": model["name"],
        "requested_model": model["model"],
        "backend": model["backend"],
        "case_id": case["id"],
        "category": case.get("category", "other"),
        "kind": case["kind"],
        "repeat": repeat,
        "media_sha256": case["sha256"],
        "request": case.get("request", {}),
    }
    # Rubric and fixture generator metadata never cross the provider boundary.
    public_case = {
        key: case[key]
        for key in ("path", "sha256", "question", "kind", "request", "routing_task")
        if key in case
    }
    try:
        result = provider.analyze(public_case)
        schema = case.get("request", {}).get("response_schema")
        if schema is not None:
            result["output_validation"] = validate_answer(
                result["answer"], schema, truncated=result.get("truncated", False)
            )
        row.update(result)
        row["grade"] = score(result["answer"], case["rubric"])
        if result.get("error"):
            row["grade"].update(passed=False, score=0.0, reason="provider_error")
        if result.get("output_validation", {}).get("valid") is False:
            row["grade"].update(passed=False, reason="output_contract_rejected")
        if result.get("truncated"):
            row["grade"].update(passed=False, reason="truncated_answer")
    except Exception as exc:  # noqa: BLE001 — isolate model failures and preserve partial results
        row.update(
            error=f"{type(exc).__name__}: {str(exc)[:500]}",
            estimated_cost_usd=None,
            grade={"passed": False, "score": 0.0, "reason": "provider_error", "fields": {}},
        )
    row["seconds"] = time.monotonic() - started
    return row


def run_suite(
    models,
    cases,
    output,
    *,
    repeats=1,
    parallel=1,
    timeout=300,
    seed=42,
    max_cost=None,
    min_accuracy=0.9,
    tui="auto",
    provider_factory=make_provider,
    demo=False,
):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory; results are never overwritten")
    forecasts = [forecast(model) for model in models]
    if max_cost is not None and any(value is None for value in forecasts):
        raise ValueError("--max-cost requires pricing and estimated_tokens for every hosted model")
    planned = None if None in forecasts else sum(forecasts) * len(cases) * repeats
    jobs = [
        (index, case, repeat)
        for repeat in range(repeats)
        for case in cases
        for index in range(len(models))
    ]
    random.Random(seed).shuffle(jobs)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "rows": [],
        "models": models,
        "total": len(jobs),
        "per_model": len(cases) * repeats,
        "started": time.monotonic(),
        "active": {},
        "projected_cost": planned,
        "status": "Preparing providers",
        "demo": demo,
    }
    report = {
        "version": 1,
        "demo": demo,
        "status": state["status"],
        "models": models,
        "total": len(jobs),
        "projected_cost_usd": planned,
        "seed": seed,
        "max_cost_usd": max_cost,
        "min_accuracy": min_accuracy,
        "manifest_fingerprint": hashlib.sha256(
            json.dumps(
                [{k: v for k, v in case.items() if k != "path"} for case in cases], sort_keys=True
            ).encode()
        ).hexdigest(),
        "cases": cases,
        "selected_case_ids": [case["id"] for case in cases],
        "providers": {},
        "rows": state["rows"],
        "summary": {},
    }
    providers, pending, stop = {}, {}, threading.Event()

    def save():
        report["status"] = state["status"]
        report["elapsed_seconds"] = time.monotonic() - state["started"]
        report["summary"] = {
            model["name"]: aggregate(
                [row for row in state["rows"] if row["model_name"] == model["name"]]
            )
            for model in models
        }
        report["categories"] = {
            model["name"]: {
                category: aggregate(
                    [
                        row
                        for row in state["rows"]
                        if row["model_name"] == model["name"] and row["category"] == category
                    ]
                )
                for category in sorted({case.get("category", "other") for case in cases})
            }
            for model in models
        }
        checkpoint(output, report)

    save()
    with Dashboard(state, tui) as dashboard, (output / "requests.jsonl").open("w") as log:
        executor = ThreadPoolExecutor(max_workers=parallel)
        position, interrupted, budget_stopped = 0, False, False
        try:
            for index, model in enumerate(models):
                dashboard.update()
                providers[index] = provider_factory(
                    model, Path(cases[0]["path"]).parent.resolve(), timeout, stop
                )
                report["providers"][model["name"]] = providers[index].metadata
            state["status"] = "Running"
            while pending or position < len(jobs):
                try:
                    while (
                        not stop.is_set()
                        and not budget_stopped
                        and len(pending) < parallel
                        and position < len(jobs)
                    ):
                        index, case, repeat = jobs[position]
                        if max_cost is not None:
                            # Unknown/error requests reserve their forecast; never count them as free.
                            spent = sum(
                                row.get("estimated_cost_usd")
                                if row.get("estimated_cost_usd") is not None
                                else forecasts[
                                    next(
                                        i
                                        for i, m in enumerate(models)
                                        if m["name"] == row["model_name"]
                                    )
                                ]
                                for row in state["rows"]
                            )
                            reserved = sum(forecasts[item[0]] for item in pending.values())
                            if spent + reserved + forecasts[index] > max_cost:
                                budget_stopped = True
                                break
                        position += 1
                        record = {
                            "event": "started",
                            "model_name": models[index]["name"],
                            "case_id": case["id"],
                            "repeat": repeat,
                        }
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        future = executor.submit(
                            evaluate, providers[index], models[index], case, repeat
                        )
                        pending[future] = (index, case, repeat)
                        state["active"][id(future)] = {
                            "model": models[index]["name"],
                            "label": case["id"],
                        }
                    if not pending:
                        break
                    done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        state["active"].pop(id(future))
                        row = future.result()
                        state["rows"].append(row)
                        log.write(
                            json.dumps({"event": "completed", **row}, ensure_ascii=False) + "\n"
                        )
                        log.flush()
                        save()
                        dashboard.update(row)
                    dashboard.update()
                    if stop.is_set() and not pending:
                        break
                except KeyboardInterrupt:
                    interrupted = True
                    stop.set()
                    state["status"] = "Stopping — waiting for active local inference"
                    save()
            state["status"] = (
                "Interrupted" if interrupted else "Budget stopped" if budget_stopped else "Complete"
            )
        except BaseException:
            stop.set()
            state["status"] = "Setup/execution failed"
            raise
        finally:
            executor.shutdown(wait=True)
            for provider in providers.values():
                provider.close()
            save()
            dashboard.update()
    print(
        f"{state['status']}: {len(state['rows'])}/{len(jobs)} results. Report: {output / 'report.md'}"
    )
    if interrupted:
        return 130
    if budget_stopped:
        return 2
    return int(
        any(
            stats["accuracy"] is None or stats["accuracy"] < min_accuracy or stats["errors"]
            for stats in report["summary"].values()
        )
    )


class DemoProvider:
    """Intentionally synthetic. Never receives or reads fixture answers."""

    delay = 0.15

    def __init__(self, model, root, timeout, stop):
        self.model, self.stop = model, stop
        self.metadata = {"simulated": True}

    def analyze(self, case):
        self.stop.wait(self.delay)
        number = int(Path(case["path"]).stem)
        correct = (number + len(self.model["name"])) % 5 != 0
        usage = {"input_tokens": 1500 + number * 10, "output_tokens": 100}
        return {
            "answer": json.dumps({"value": number if correct else -1}),
            "usage": usage,
            "estimated_cost_usd": 0.0 if self.model["backend"] == "ollama" else 0.004,
            "cost_source": "simulated",
        }

    def close(self):
        pass


def dispatch(args):
    if args.command == "demo":
        if not math.isfinite(args.delay) or not 0 <= args.delay <= 10:
            raise ValueError("Demo delay must be in [0, 10] seconds")
        DemoProvider.delay = args.delay
        models = [
            {"name": name, "model": "SIMULATED", "backend": backend}
            for name, backend in (
                ("Local Qwen", "ollama"),
                ("Native Codex", "codex"),
                ("Native Claude", "claude"),
            )
        ]
        cases = [
            {
                "id": f"scene-{n:02}",
                "path": str(Path("/tmp") / f"{n}.png"),
                "sha256": "simulated",
                "question": "Return the value as JSON.",
                "kind": "image",
                "category": "OCR" if n % 2 else "spatial",
                "rubric": {"expected": {"value": n}},
            }
            for n in range(12)
        ]
        run_suite(
            models,
            cases,
            args.output,
            parallel=3,
            tui=args.tui,
            provider_factory=DemoProvider,
            demo=True,
            min_accuracy=0,
        )
        return
    if (
        not 1 <= args.repeats <= 1000
        or not 1 <= args.parallel <= 32
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not 0 <= args.min_accuracy <= 1
        or (args.limit is not None and args.limit < 1)
        or (args.max_cost is not None and (not math.isfinite(args.max_cost) or args.max_cost < 0))
    ):
        raise ValueError("Invalid repeat, parallel, timeout, accuracy, limit or cost option")
    models = load_models(args.config)
    _, cases = load_manifest(args.manifest)
    cases = select_cases(cases, models, args.kind, args.category, args.limit)
    # Root must cover nested media directories, not only the first case's subdirectory.
    root = args.manifest.parent.resolve()

    def factory(model, _root, timeout, stop):
        return make_provider(model, root, timeout, stop)

    code = run_suite(
        models,
        cases,
        args.output,
        repeats=args.repeats,
        parallel=args.parallel,
        timeout=args.timeout,
        seed=args.seed,
        max_cost=args.max_cost,
        min_accuracy=args.min_accuracy,
        tui=args.tui,
        provider_factory=factory,
    )
    raise SystemExit(code)
