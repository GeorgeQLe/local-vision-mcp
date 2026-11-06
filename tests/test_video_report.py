from local_vision_mcp.benchmark.video_report import summarize


def test_sampled_success_does_not_hide_known_source_transition_gap():
    cases = [
        {
            "id": "one-frame",
            "kind": "video",
            "task_id": "camera:states",
            "request": {"frames": 1},
            "rubric": {"expected": {"states": ["person"]}},
            "full_trace_reference": {"states": ["black", "person"]},
            "sample_covers_reference": False,
        }
    ]
    row = {
        "model_name": "small",
        "case_id": "one-frame",
        "repeat": 0,
        "grade": {"passed": True},
        "seconds": 1,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    report = {
        "status": "Complete",
        "cases": cases,
        "models": [{"name": "small", "backend": "ollama"}],
        "rows": [row],
    }
    result = summarize(report)
    group = result["models"][0]["measurements"][0]
    assert group["passed"] == 1
    assert group["host_total_tokens"] is None
    assert group["local_tokens"]["input_tokens"] == 100
    assert result["known_input_sampling_gaps"][0]["case_id"] == "one-frame"


def test_still_video_pairs_are_by_task_and_repeat_not_just_source():
    cases = [
        {"id": "image", "kind": "image", "task_id": "same:ocr"},
        {"id": "video", "kind": "video", "task_id": "same:ocr", "request": {"frames": 3}},
        {"id": "other", "kind": "image", "task_id": "same:count"},
    ]
    rows = [
        {
            "model_name": "native",
            "case_id": c["id"],
            "repeat": 0,
            "grade": {"passed": c["kind"] == "image", "reason": "scored"},
            "seconds": 1,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        for c in cases
    ]
    result = summarize(
        {
            "status": "Complete",
            "cases": cases,
            "models": [{"name": "native", "backend": "codex"}],
            "rows": rows,
        }
    )["models"][0]
    assert result["static_control_pairs"] == 1
    assert result["still_pass_video_fail_case_ids"] == ["video"]
    assert result["measurements"][0]["host_total_tokens"] == 220
