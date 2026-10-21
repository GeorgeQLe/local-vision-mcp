import copy

from local_vision_mcp.benchmark.efficiency import local_usage, paired_summary


def report():
    models = [
        {"name": "direct", "backend": "codex", "model": "same", "effort": "medium"},
        {
            "name": "assisted",
            "backend": "codex",
            "model": "same",
            "effort": "medium",
            "perception": "mcp",
            "local_model": "local",
        },
    ]
    rows = []
    for name, inputs in [("direct", 100), ("assisted", 200)]:
        for repeat in range(2):
            rows.append(
                {
                    "model_name": name,
                    "case_id": "one",
                    "repeat": repeat,
                    "grade": {"passed": True},
                    "seconds": 1,
                    "usage": {
                        "input_tokens": inputs,
                        "output_tokens": 10,
                        "cached_input_tokens": 50,
                    },
                }
            )
    return {
        "models": models,
        "rows": rows,
        "status": "Complete",
        "total": 4,
        "selected_case_ids": ["one"],
    }


def test_paired_host_tokens_include_cache_and_repeated_calls():
    document = report()
    item = paired_summary(document)["comparisons"][0]
    assert item["complete"]
    assert item["paired_requests"] == 2
    assert item["native"]["host_total_tokens"] == 220
    assert item["mcp"]["host_total_tokens"] == 420
    assert item["native"]["host_uncached_input_tokens"] == 100
    assert item["host_token_reduction_fraction"] < 0
    assert item["regression_case_ids"] == []
    document["rows"][-1]["grade"]["passed"] = False
    assert paired_summary(document)["comparisons"][0]["regression_case_ids"] == ["one"]


def test_partial_or_unknown_usage_cannot_claim_token_savings():
    document = report()
    document["rows"][-1].pop("usage")
    item = paired_summary(document)["comparisons"][0]
    assert item["host_token_reduction_fraction"] is None
    assert item["mcp"]["usage_known_requests"] == 1
    document = report()
    document["rows"].pop()
    item = paired_summary(document)["comparisons"][0]
    assert not item["complete"]
    assert item["host_token_reduction_fraction"] is None


def test_local_metrics_remain_separate_for_both_protocols():
    payload = '{"inference":{"prompt_eval_count":1000,"eval_count":20}}'
    codex = {"mcp_calls": [{"result": {"content": [{"type": "text", "text": payload}]}}]}
    claude = {"mcp_results": [{"type": "tool_result", "content": payload}]}
    expected = {"calls_with_usage": 1, "input_tokens": 1000, "output_tokens": 20}
    assert local_usage(codex) == local_usage(claude) == expected
    codex["mcp_calls"][0]["arguments"] = {"question": payload}
    assert local_usage(codex) == expected  # User text is not trusted runtime usage.
    document = copy.deepcopy(report())
    document["rows"][-1].update(codex)
    item = paired_summary(document)["comparisons"][0]
    assert item["mcp"]["local_tokens"] == expected
    assert item["mcp"]["host_total_tokens"] == 420
