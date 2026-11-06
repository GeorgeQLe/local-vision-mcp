"""Describe structured-output screening and conservative hybrid measurements without promotion."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from ..structured import validate_answer
from .scoring import score


def lower_bound(successes, total, alpha=0.05):
    """One-sided exact binomial bound; independence is a separate sampling requirement."""
    if not total or not successes:
        return 0.0
    if successes == total:
        return alpha ** (1 / total)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(
            math.comb(total, k) * p**k * (1 - p) ** (total - k) for k in range(successes, total + 1)
        )
        if tail < alpha:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def summarize(report):
    cases = {c["id"]: c for c in report["cases"]}
    groups = {}
    for name in (m["name"] for m in report["models"]):
        rows = [r for r in report["rows"] if r["model_name"] == name]
        by_category = {}
        for category in sorted({r["category"] for r in rows}):
            subset = [r for r in rows if r["category"] == category]
            passed, rejected, semantic = 0, 0, 0
            for row in subset:
                case = cases[row["case_id"]]
                contract = validate_answer(
                    row.get("answer"),
                    case["request"]["response_schema"],
                    truncated=row.get("truncated", False),
                )
                correct = row["grade"]["passed"] and contract["valid"] and not row.get("error")
                passed += bool(correct)
                rejected += not contract["valid"] or bool(row.get("error"))
                semantic += contract["valid"] and not correct and not row.get("error")
            by_category[category] = {
                "completed": len(subset),
                "passed": passed,
                "contract_or_provider_rejections": rejected,
                "valid_but_wrong": semantic,
                "descriptive_95pct_lower_bound": lower_bound(passed, len(subset)),
            }
        local_rows = [
            r["local_observation"] for r in rows if r.get("local_observation") is not None
        ]
        timings = sorted(r["seconds"] for r in rows)
        hosted = [r for r in rows if r["backend"] in ("codex", "claude")]
        local = local_rows or ([r for r in rows] if rows and rows[0]["backend"] == "ollama" else [])

        def tokens(items):
            values = [
                r.get("usage", {}).get(k) for r in items for k in ("input_tokens", "output_tokens")
            ]
            return sum(values) if values and all(type(v) is int for v in values) else None

        shadow_grades = []
        for row in rows:
            observation = row.get("local_observation")
            if observation is not None:
                case = cases[row["case_id"]]
                shadow_grades.append(
                    not observation.get("error")
                    and not observation.get("truncated")
                    and validate_answer(
                        observation.get("answer"), case["request"]["response_schema"]
                    )["valid"]
                    and score(observation.get("answer", ""), case["rubric"])["passed"]
                )
        groups[name] = {
            "completed": len(rows),
            "passed": sum(v["passed"] for v in by_category.values()),
            "errors": sum(bool(r.get("error")) for r in rows),
            "categories": by_category,
            "hosted_tokens": tokens(hosted),
            "local_tokens": tokens(local),
            "mean_seconds": statistics.mean(timings) if timings else None,
            "median_seconds": statistics.median(timings) if timings else None,
            "p95_seconds": timings[max(0, math.ceil(0.95 * len(timings)) - 1)] if timings else None,
            "local_shadow_calls": len(local_rows),
            "local_shadow_passed": sum(shadow_grades),
            "native_calls": len(hosted),
            "accepted_local": sum(bool(r.get("accepted_local")) for r in rows),
            "local_shadow_errors": sum(bool(r.get("error")) for r in local_rows),
        }
    failures = [
        {
            k: row.get(k)
            for k in ("model_name", "case_id", "answer", "grade", "output_validation", "error")
        }
        for row in report["rows"]
        if not row["grade"]["passed"]
    ]
    return {
        "qualified_for_routing": False,
        "status": report["status"],
        "models": groups,
        "failures": failures,
        "limitations": [
            "Synthetic related generator families are not independent real-source qualification.",
            "Direct local tokens and hosted tokens use different tokenizers.",
            "Shadow mode retains native answers and cannot avoid native uploads or calls.",
            "A binomial bound is descriptive only when the observations are correlated.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = summarize(json.loads(raw))
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Structured output screening",
        "",
        "No local route is promoted.",
        "",
        "| Model | Correct | Contract/provider rejected | Valid but wrong | Mean s | Hosted tokens | Local tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in result["models"].items():
        categories = stats["categories"].values()
        rejected = sum(c["contract_or_provider_rejections"] for c in categories)
        semantic = sum(c["valid_but_wrong"] for c in categories)
        lines.append(
            f"| {name} | {stats['passed']}/{stats['completed']} | {rejected} | {semantic} | "
            f"{stats['mean_seconds']:.2f} | {stats['hosted_tokens']} | {stats['local_tokens']} |"
        )
    lines += ["", *result["limitations"]]
    (args.output / "qualification.md").write_text("\n".join(lines) + "\n")
    print(args.output / "qualification.md")


if __name__ == "__main__":
    main()
