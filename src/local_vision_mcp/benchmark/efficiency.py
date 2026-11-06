"""Summarize paired CLI native/MCP runs without mixing local and hosted tokens.

Usage: python -m local_vision_mcp.benchmark.efficiency path/to/report.json
Writes efficiency.json and efficiency.md beside the original report.
"""

import argparse
import json
from pathlib import Path


def local_reports(row):
    """Extract inference metrics only from recorded MCP protocol results."""
    reports = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("inference"), dict):
                reports.append(value["inference"])
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                visit(json.loads(value))
            except ValueError:
                pass

    for call in row.get("mcp_calls", []):
        visit(call.get("result"))
    visit(row.get("mcp_results", []))
    return reports


def local_usage(row):
    """Keep local tokenizer counts separate from hosted usage."""
    reports = local_reports(row)
    return {
        "calls_with_usage": sum(r.get("calls", 1) for r in reports),
        "input_tokens": sum(r.get("prompt_eval_count", 0) or 0 for r in reports),
        "output_tokens": sum(r.get("eval_count", 0) or 0 for r in reports),
    }


def totals(rows):
    known = [
        r["usage"]
        for r in rows
        if all(
            isinstance((r.get("usage") or {}).get(k), int)
            for k in ("input_tokens", "output_tokens")
        )
    ]
    tokens = {
        key: sum(u.get(key, 0) or 0 for u in known)
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
        )
    }
    local = [local_usage(row) for row in rows]
    return {
        "requests": len(rows),
        "passed": sum(bool(r["grade"]["passed"]) for r in rows),
        "errors": sum(bool(r.get("error")) for r in rows),
        "usage_known_requests": len(known),
        "host_tokens": tokens,
        "host_total_tokens": tokens["input_tokens"] + tokens["output_tokens"],
        "host_uncached_input_tokens": tokens["input_tokens"] - tokens["cached_input_tokens"],
        "local_tokens": {
            key: sum(r[key] for r in local)
            for key in ("calls_with_usage", "input_tokens", "output_tokens")
        },
        "mean_seconds": sum(r["seconds"] for r in rows) / len(rows) if rows else None,
    }


def paired_summary(report):
    comparisons = []
    for assisted in report["models"]:
        if assisted.get("perception") != "mcp":
            continue
        baselines = [
            m
            for m in report["models"]
            if (m["backend"], m["model"], m.get("effort"), m.get("perception", "native"))
            == (assisted["backend"], assisted["model"], assisted.get("effort"), "native")
        ]
        if len(baselines) != 1:
            raise ValueError("Each MCP arm needs exactly one matching native baseline")
        baseline = baselines[0]
        native = {
            (r["case_id"], r["repeat"]): r
            for r in report["rows"]
            if r["model_name"] == baseline["name"]
        }
        mcp = {
            (r["case_id"], r["repeat"]): r
            for r in report["rows"]
            if r["model_name"] == assisted["name"]
        }
        keys = sorted(native.keys() & mcp.keys())
        direct, indirect = [native[k] for k in keys], [mcp[k] for k in keys]
        first, second = totals(direct), totals(indirect)
        expected = report["total"] // len(report["models"])
        complete = (
            len(keys) == expected
            and len(native) == len(mcp) == len(keys)
            and report["status"] == "Complete"
        )
        known = bool(keys) and first["usage_known_requests"] == second[
            "usage_known_requests"
        ] == len(keys)
        regressions = [
            k[0] for k in keys if native[k]["grade"]["passed"] and not mcp[k]["grade"]["passed"]
        ]
        improvements = [
            k[0] for k in keys if not native[k]["grade"]["passed"] and mcp[k]["grade"]["passed"]
        ]
        comparisons.append(
            {
                "backend": assisted["backend"],
                "model": assisted["model"],
                "native_name": baseline["name"],
                "mcp_name": assisted["name"],
                "local_model": assisted["local_model"],
                "local_strategy": assisted.get("local_strategy", "single"),
                "complete": complete,
                "paired_requests": len(keys),
                "native": first,
                "mcp": second,
                "regression_case_ids": regressions,
                "improvement_case_ids": improvements,
                "host_token_reduction_fraction": (
                    1 - second["host_total_tokens"] / first["host_total_tokens"]
                    if complete and known and first["host_total_tokens"]
                    else None
                ),
            }
        )
    return {
        "status": report["status"],
        "comparisons": comparisons,
        "scope": "Descriptive paired measurements; synthetic cases do not establish general parity.",
    }


def write_summary(path):
    report = json.loads(path.read_text())
    summary = paired_summary(report)
    path.with_name("efficiency.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# CLI agent token efficiency",
        "",
        f"Status: {summary['status']}",
        "",
        "| CLI | Paired tasks | Native correct | MCP correct | Native host tokens | MCP host tokens | Token reduction | Mean seconds native / MCP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["comparisons"]:
        a, b = item["native"], item["mcp"]
        change = item["host_token_reduction_fraction"]
        reduction = "unknown/incomplete" if change is None else f"{change:.1%}"
        latency = (
            f"{a['mean_seconds']:.2f} / {b['mean_seconds']:.2f}"
            if item["paired_requests"]
            else "unknown"
        )
        lines.append(
            f"| {item['backend']} ({item['local_strategy']}) | {item['paired_requests']} | "
            f"{a['passed']} | {b['passed']} | {a['host_total_tokens']:,} | "
            f"{b['host_total_tokens']:,} | {reduction} | {latency} |"
        )
    lines += [
        "",
        "Negative token reduction means MCP consumed more hosted tokens.",
        "",
        (
            "Host tokens include reported input (including cache reads/writes) and output. "
            "Reasoning already included in output is not added twice. Local inference tokens "
            "are recorded separately in efficiency.json; different tokenizers are not combined. "
            "Unknown usage prevents a token reduction claim. These are subscription CLI usage "
            "measurements, not a monetary savings estimate."
        ),
        "",
        (
            "Each task starts a fresh CLI session. The MCP arm uses the real stdio tool surface "
            "and an isolated in-process server; its local model remains warm in Ollama. "
            "It is explicitly instructed to use MCP, with no native fallback. This measures "
            "one-question perception quality and total reported host usage, including tool "
            "overhead. It does not measure autonomous adoption, longer coding workflows, "
            "shared coordinator performance, or general audio/video parity."
        ),
        "",
        (
            "Results are descriptive. Related synthetic scenes and repeated questions are "
            "not independent evidence of population-level equivalence. Representative held-out "
            "real tasks and a predefined quality margin are needed for a parity claim."
        ),
        "",
    ]
    for item in summary["comparisons"]:
        lines += [
            f"{item['backend']} regressions: "
            + (", ".join(item["regression_case_ids"]) or "none")
            + ".",
            "",
        ]
    path.with_name("efficiency.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    write_summary(args.report)
    print(args.report.with_name("efficiency.md"))
