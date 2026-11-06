"""Describe task-level capability and overhead; never qualify a production router.

Usage: python -m local_vision_mcp.benchmark.task_gap REPORT... --output DIRECTORY
Reports remain separate: model settings and workloads may differ between runs.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .efficiency import paired_summary, totals
from .scoring import score


def observations(value):
    """Read runtime observations from tool results, never tool arguments."""
    if isinstance(value, dict):
        if "answer" in value and isinstance(value.get("inference"), dict):
            yield value
        else:
            for child in value.values():
                yield from observations(child)
    elif isinstance(value, list):
        for child in value:
            yield from observations(child)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        if isinstance(decoded, (dict, list)):
            yield from observations(decoded)


def first_local_observation(row, case, model):
    """Match the original question, excluding host follow-ups from local accuracy."""
    if model["backend"] in ("ollama", "omni"):
        return row if "answer" in row else None
    if model.get("local_strategy") == "consensus":
        for audit in row.get("local_consensus_audit", []):
            if audit["question"] == case["question"]:
                return {"answer": audit["decision"]["answer"] or ""}
        return None
    results = {r.get("tool_use_id"): r for r in row.get("mcp_results", [])}
    for call in row.get("mcp_calls", []):
        arguments = call.get("arguments", call.get("input", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
        if not isinstance(arguments, dict) or arguments.get("question") != case["question"]:
            continue
        result = call.get("result", results.get(call.get("id")))
        observation = next(observations(result), None)
        if observation is not None:
            return observation
    return None


def route(model):
    if model["backend"] in ("ollama", "omni"):
        return "local direct"
    if model.get("perception") == "mcp":
        return "MCP local " + model.get("local_strategy", "single")
    return "native"


def summarize(report):
    cases = {c["id"]: c for c in report["cases"]}
    selected = set(report.get("selected_case_ids", cases))
    categories = sorted({cases[c].get("category", "other") for c in selected})
    groups, comparisons = [], []
    for category in categories:
        ids = {c for c in selected if cases[c].get("category", "other") == category}
        rows = [r for r in report["rows"] if r["case_id"] in ids]
        for model in report["models"]:
            arm = [r for r in rows if r["model_name"] == model["name"]]
            local_grades, missing, failures, recovered = [], [], [], []
            for row in arm:
                observation = first_local_observation(row, cases[row["case_id"]], model)
                if observation is None:
                    missing.append(row["case_id"])
                    continue
                answer = observation.get("answer")
                if observation.get("error"):
                    reason = "provider_error"
                elif observation.get("truncated") or (
                    observation.get("inference", {}).get("done_reason") == "length"
                ):
                    reason = "truncated"
                elif not isinstance(answer, str):
                    reason = "missing_answer"
                else:
                    grade = score(answer, cases[row["case_id"]]["rubric"])
                    reason = None if grade["passed"] else grade["reason"]
                    if reason == "scored":
                        reason = "field_mismatch"
                local_grades.append(reason is None)
                if reason:
                    failures.append({"case_id": row["case_id"], "reason": reason})
                    if row["grade"]["passed"]:
                        recovered.append(row["case_id"])
            metrics = totals(arm)
            local_direct = model["backend"] in ("ollama", "omni")
            if local_direct:
                # Direct-local Provider.usage uses the local tokenizer, not hosted tokens.
                metrics["local_tokens"] = {
                    "calls_with_usage": metrics["usage_known_requests"],
                    "input_tokens": metrics["host_tokens"]["input_tokens"],
                    "output_tokens": metrics["host_tokens"]["output_tokens"],
                }
                metrics["host_tokens"] = None
                metrics["host_total_tokens"] = None
                metrics["host_uncached_input_tokens"] = None
            groups.append(
                {
                    "category": category,
                    "model": model,
                    "route": route(model),
                    **metrics,
                    "observed_media_hashes": sorted({cases[r["case_id"]]["sha256"] for r in arm}),
                    "host_usage_complete": not local_direct
                    and bool(arm)
                    and metrics["usage_known_requests"] == len(arm),
                    "first_local": {
                        "graded": len(local_grades),
                        "passed": sum(bool(g) for g in local_grades),
                        "missing_case_ids": missing,
                        "failures": failures,
                        "failure_reasons": dict(Counter(f["reason"] for f in failures)),
                        "host_recovered_case_ids": recovered,
                    }
                    if route(model) != "native"
                    else None,
                    "failed_case_ids": [r["case_id"] for r in arm if not r["grade"]["passed"]],
                }
            )
        # Preserve pairing/completeness checks and suppress unknown usage claims.
        if any(m.get("perception") == "mcp" for m in report["models"]):
            subset = {
                **report,
                "rows": rows,
                "total": report["total"] * len(ids) // len(selected),
                "selected_case_ids": sorted(ids),
            }
            comparisons.extend(
                {"category": category, **pair} for pair in paired_summary(subset)["comparisons"]
            )
    return {
        "status": report["status"],
        "groups": groups,
        "paired_native_mcp": comparisons,
        "qualified_for_routing": False,
        "scope": (
            "Descriptive diagnostics only. Hosts, repeats and related scenes are not independent "
            "samples. No router was executed. First local grading excludes changed-question "
            "follow-ups; missing observations are reported, not counted as local successes."
        ),
    }


def write_reports(paths, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    lines = ["# Vision task gap", "", "Each run retains its own workload and model settings.", ""]
    for path in paths:
        raw = path.read_bytes()
        run = {"source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest()}
        run.update(summarize(json.loads(raw)))
        runs.append(run)
        lines += [
            f"## {path.parent.name}",
            "",
            f"Status: {run['status']}",
            "",
            "| Task | Model / host | Route | Final correct | First local correct | Unique media | Hosted tokens | Mean seconds |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for group in run["groups"]:
            local = group["first_local"]
            first = f"{local['passed']}/{local['graded']}" if local else "—"
            tokens = f"{group['host_total_tokens']:,}" if group["host_usage_complete"] else "—"
            seconds = group["mean_seconds"]
            seconds = f"{seconds:.2f}" if seconds is not None else "—"
            lines.append(
                f"| {group['category']} | {group['model']['name']} | {group['route']} | "
                f"{group['passed']}/{group['requests']} | {first} | "
                f"{len(group['observed_media_hashes'])} | {tokens} | {seconds} |"
            )
        lines += ["", run["scope"], ""]
    lines += [
        (
            "No task category is automatically qualified by this report. Evaluate a frozen router "
            "on new held-out scenes before promotion. Ground truth is used only for offline scoring, "
            "never as a runtime routing signal. Hosted totals include cache usage; local tokens "
            "remain separate in the JSON. No efficiency claim follows from accuracy alone."
        ),
        "",
    ]
    (output / "task-gap.json").write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    (output / "task-gap.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_reports(args.reports, args.output)
    print(args.output / "task-gap.md")
