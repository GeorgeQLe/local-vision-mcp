import json

from local_vision_mcp.benchmark.task_gap import first_local_observation, summarize, write_reports


def document():
    case = {
        "id": "absent",
        "category": "abstention",
        "sha256": "same-scene",
        "question": "Read SHA or null",
        "rubric": {"expected": {"sha": None}},
    }
    native = {"name": "native", "backend": "claude", "model": "host"}
    local = {
        **native,
        "name": "mcp",
        "perception": "mcp",
        "local_model": "small",
    }
    observation = {
        "answer": '{"sha":"build-123"}',
        "inference": {"prompt_eval_count": 20, "eval_count": 5},
    }
    row = {
        "case_id": "absent",
        "repeat": 0,
        "seconds": 2,
        "grade": {"passed": True},
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    return {
        "cases": [case],
        "models": [native, local],
        "rows": [
            {**row, "model_name": "native"},
            {
                **row,
                "model_name": "mcp",
                "mcp_calls": [
                    {"id": "one", "input": {"question": case["question"]}},
                    {"id": "two", "input": {"question": "Transcribe all text"}},
                ],
                "mcp_results": [
                    {"tool_use_id": "one", "content": json.dumps(observation)},
                    {
                        "tool_use_id": "two",
                        "content": json.dumps({**observation, "answer": '{"sha":null}'}),
                    },
                ],
            },
        ],
        "status": "Complete",
        "total": 2,
    }


def test_host_recovery_does_not_qualify_wrong_first_local_answer(tmp_path):
    report = document()
    group = summarize(report)["groups"][1]
    assert group["passed"] == 1
    local = group["first_local"]
    assert local["passed"] == 0 and local["graded"] == 1
    assert local["missing_case_ids"] == []
    assert local["failure_reasons"] == {"field_mismatch": 1}
    assert local["host_recovered_case_ids"] == ["absent"]
    assert group["observed_media_hashes"] == ["same-scene"]
    assert not summarize(report)["qualified_for_routing"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    write_reports([path], tmp_path / "analysis")
    assert (tmp_path / "analysis/task-gap.json").is_file()


def test_codex_result_matching_truncation_and_missing_original():
    report = document()
    case, model, row = report["cases"][0], report["models"][1], report["rows"][1]
    observation = {"answer": '{"sha":null}', "inference": {}, "truncated": True}
    row["mcp_calls"] = [
        {"arguments": json.dumps({"question": case["question"]}), "result": observation}
    ]
    row["mcp_results"] = []
    assert first_local_observation(row, case, model) == observation
    assert summarize(report)["groups"][1]["first_local"]["passed"] == 0
    row["mcp_calls"][0]["arguments"] = {"question": "different question"}
    local = summarize(report)["groups"][1]["first_local"]
    assert local["graded"] == 0 and local["passed"] == 0
    assert local["missing_case_ids"] == ["absent"]


def test_unknown_usage_suppresses_savings_and_local_tokens_are_not_hosted():
    report = document()
    report["rows"][1].pop("usage")
    result = summarize(report)
    assert not result["groups"][1]["host_usage_complete"]
    assert result["paired_native_mcp"][0]["host_token_reduction_fraction"] is None
    report = document()
    report["models"] = [{"name": "small", "backend": "ollama", "model": "small"}]
    report["rows"] = [{**report["rows"][0], "model_name": "small", "answer": '{"sha":null}'}]
    report["total"] = 1
    group = summarize(report)["groups"][0]
    assert group["host_total_tokens"] is None
    assert group["local_tokens"]["input_tokens"] == 100
    assert group["first_local"]["passed"] == 1


def test_consensus_scores_first_original_vote_and_partial_runs_stay_partial():
    report = document()
    report["models"][1]["local_strategy"] = "consensus"
    row = report["rows"][1]
    row["local_consensus_audit"] = [
        {"question": "different", "decision": {"answer": '{"sha":null}'}},
        {"question": report["cases"][0]["question"], "decision": {"answer": None}},
    ]
    assert summarize(report)["groups"][1]["first_local"]["passed"] == 0
    report["status"] = "Running"
    report["rows"] = report["rows"][:1]
    result = summarize(report)
    assert result["groups"][1]["requests"] == 0
    assert not result["paired_native_mcp"][0]["complete"]
    assert result["paired_native_mcp"][0]["host_token_reduction_fraction"] is None
