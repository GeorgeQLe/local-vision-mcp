"""Report hosted usage, local work and correlated errors for consensus experiments."""

import argparse
import json
from pathlib import Path

from .efficiency import local_reports, totals, write_summary
from .scoring import score


def summarize(report):
    cases = {case["id"]: case for case in report["cases"]}
    arms = []
    for model in report["models"]:
        rows = [r for r in report["rows"] if r["model_name"] == model["name"]]
        audits = [(r, a) for r in rows for a in r.get("local_consensus_audit", [])]
        first_local_grades = []
        for row in rows:
            original = next(
                (
                    a
                    for a in row.get("local_consensus_audit", [])
                    if a["question"] == cases[row["case_id"]]["question"]
                ),
                None,
            )
            if original is not None:
                first_local_grades.append(
                    score(original["decision"]["answer"] or "", cases[row["case_id"]]["rubric"])
                )
        wrong_unanimous = []
        unresolved = 0
        local_seconds = 0
        overlap_seconds = 0
        invalid_passes = 0
        unscored_unanimous = 0
        for row, audit in audits:
            decision = audit["decision"]
            if decision["status"] == "unanimous":
                if audit["question"] != cases[row["case_id"]]["question"]:
                    unscored_unanimous += 1
                elif not score(decision["answer"], cases[row["case_id"]]["rubric"])["passed"]:
                    wrong_unanimous.append(row["case_id"])
            unresolved += decision["status"] == "unresolved"
            invalid_passes += sum(p.get("canonical") is None for p in audit["passes"])
            local_seconds += sum(p["ended"] - p["started"] for p in audit["passes"])
            overlap_seconds += audit["request_overlap_seconds"]
        arms.append(
            {
                "name": model["name"],
                "backend": model["backend"],
                "route": "native"
                if model.get("perception") != "mcp"
                else model.get("local_strategy", "single"),
                **totals(rows),
                "audited_calls": len(audits),
                "first_local_consensus": {
                    "graded_requests": len(first_local_grades),
                    "passed": sum(g["passed"] for g in first_local_grades),
                    "expected_requests": len(rows),
                }
                if model.get("local_strategy") == "consensus"
                else None,
                "wrong_unanimous_case_ids": wrong_unanimous,
                "unresolved_calls": unresolved,
                "invalid_local_passes": invalid_passes,
                "unscored_followup_unanimous_calls": unscored_unanimous,
                "summed_local_inference_seconds": sum(
                    r.get("total_duration", 0) / 1e9 for row in rows for r in local_reports(row)
                ),
                "summed_local_eval_seconds": sum(
                    r.get("eval_duration", 0) / 1e9 for row in rows for r in local_reports(row)
                ),
                "summed_local_request_seconds": local_seconds,
                "summed_first_two_request_overlap_seconds": overlap_seconds,
            }
        )
    return {"status": report["status"], "arms": arms}


def write_report(path):
    report = json.loads(path.read_text())
    write_summary(path)
    summary = summarize(report)
    path.with_name("consensus.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Local consensus experiment",
        "",
        f"Status: {summary['status']}",
        "",
        "| Individual host | Perception route | Host answer correct | First local vote correct | Hosted tokens | Mean seconds | Local calls | Local input / output tokens | Wrong unanimous local calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        local = arm["local_tokens"]
        vote = arm["first_local_consensus"]
        local_correct = f"{vote['passed']}/{vote['graded_requests']}" if vote else "—"
        route = {
            "native": "native",
            "single": "MCP: single local pass",
            "consensus": "MCP: local model consensus",
        }[arm["route"]]
        lines.append(
            f"| {arm['backend']} | {route} | {arm['passed']}/{arm['requests']} | {local_correct} | "
            f"{arm['host_total_tokens']:,} | {arm['mean_seconds']:.2f} | "
            f"{local['calls_with_usage']} | {local['input_tokens']:,} / {local['output_tokens']:,} | "
            f"{len(arm['wrong_unanimous_case_ids'])} |"
        )
    lines += [
        "",
        (
            "Only local Qwen passes vote. Each Claude or Codex host is one agent receiving "
            "one reduced report per MCP call. First local vote accuracy is graded before "
            "host interpretation or follow-up calls; missing original-question audits are "
            "visible through expected_requests versus graded_requests in consensus.json."
        ),
        "",
        (
            "Hosted tokens include cache reads and writes; they are not a billing estimate. "
            "Local tokenizers and hosted tokenizers are reported separately. Unknown usage "
            "is recorded in consensus.json. Local calls are inference passes, including retries."
        ),
        "",
        (
            "Wrong unanimous calls are scored against hidden ground truth before the host "
            "answers. The verifier sees the first two candidates, so votes are correlated. "
            "Request overlap shows concurrent client requests, not simultaneous GPU kernels. "
            "Separate serving replicas are required for Qwen 3.5 on the tested Ollama version."
        ),
        "",
        (
            "This tests a package of enlarged views, stricter local prompts and voting. "
            "It does not isolate voting's causal contribution or establish general parity. "
            "The synthetic scenes share layouts and are not independent real-world samples."
        ),
        "",
    ]
    path.with_name("consensus.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print(json.dumps(write_report(parser.parse_args().report), indent=2))
