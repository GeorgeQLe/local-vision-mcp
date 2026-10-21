"""Separate sampled-video perception, still controls and known sampling gaps."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .efficiency import totals


def summarize(report):
    cases = {c["id"]: c for c in report["cases"]}
    result = []
    for model in report["models"]:
        rows = [r for r in report["rows"] if r["model_name"] == model["name"]]
        groups = defaultdict(list)
        tasks = defaultdict(dict)
        for row in rows:
            case = cases[row["case_id"]]
            count = case.get("request", {}).get("frames", 1)
            groups[(case["kind"], count)].append(row)
            tasks[(case.get("task_id", case["id"]), row["repeat"])][case["kind"]] = row
        static_pairs = [t for t in tasks.values() if "image" in t and "video" in t]
        measurements = []
        for (kind, count), selected in sorted(groups.items()):
            metrics = totals(selected)
            if model["backend"] in ("ollama", "omni"):
                metrics["local_tokens"] = {
                    "calls_with_usage": metrics["usage_known_requests"],
                    "input_tokens": metrics["host_tokens"]["input_tokens"],
                    "output_tokens": metrics["host_tokens"]["output_tokens"],
                }
                metrics["host_tokens"] = None
                metrics["host_total_tokens"] = None
                metrics["host_uncached_input_tokens"] = None
            measurements.append({"kind": kind, "frames": count, **metrics})
        result.append(
            {
                "model": model,
                "measurements": measurements,
                "static_control_pairs": len(static_pairs),
                "still_pass_video_fail_case_ids": [
                    t["video"]["case_id"]
                    for t in static_pairs
                    if t["image"]["grade"]["passed"] and not t["video"]["grade"]["passed"]
                ],
                "still_fail_video_pass_case_ids": [
                    t["video"]["case_id"]
                    for t in static_pairs
                    if not t["image"]["grade"]["passed"] and t["video"]["grade"]["passed"]
                ],
                "failed_cases": [
                    {
                        "id": r["case_id"],
                        "task_id": cases[r["case_id"]].get("task_id"),
                        "frames": cases[r["case_id"]].get("request", {}).get("frames", 1),
                        "kind": cases[r["case_id"]]["kind"],
                        "reason": r["grade"]["reason"],
                        "answer": r.get("answer"),
                    }
                    for r in rows
                    if not r["grade"]["passed"]
                ],
            }
        )
    known_gaps = [
        {
            "case_id": c["id"],
            "task_id": c.get("task_id"),
            "frames": c.get("request", {}).get("frames", 1),
            "sampled_expected": c["rubric"]["expected"],
            "reference_expected": c["full_trace_reference"],
        }
        for c in cases.values()
        if c.get("sample_covers_reference") is False
    ]
    return {
        "status": report["status"],
        "models": result,
        "known_input_sampling_gaps": known_gaps,
        "scope": (
            "Scores concern the supplied frames, not exhaustive video coverage. Known input "
            "gaps are determined from frozen source annotations, not provider answers. Native "
            "agents see the same frames; recovering unsampled events requires resampling. "
            "Direct-local usage is local tokenizer usage, not hosted token savings."
        ),
    }


def write_report(path):
    summary = summarize(json.loads(path.read_text()))
    path.with_name("video.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Real-video pilot",
        "",
        f"Status: {summary['status']}",
        "",
        "| Model | Input | Frames | Correct | Errors | Mean seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in summary["models"]:
        for group in model["measurements"]:
            lines.append(
                f"| {model['model']['name']} | {group['kind']} | {group['frames']} | "
                f"{group['passed']}/{group['requests']} | {group['errors']} | "
                f"{group['mean_seconds']:.2f} |"
            )
    lines += ["", summary["scope"], ""]
    path.with_name("video.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(write_report(parser.parse_args().report), indent=2))
