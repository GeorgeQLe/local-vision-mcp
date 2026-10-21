import json
from pathlib import Path

import pytest

from local_vision_mcp.benchmark.routing import recommend, reject_observation


def test_restrictions_are_model_scoped_and_shadow_never_replaces_native():
    policy = json.loads((Path(__file__).parents[1] / "benchmarks/routing.policy.json").read_text())
    request = {
        "task": "clear_large_text_ocr",
        "kind": "image",
        "model_digest": policy["reference_model_digest"],
    }
    assert recommend(policy, **request)["primary"] == "native"
    assert not recommend(policy, **request)["local_shadow"]
    assert recommend(policy, **request, allow_shadow=True)["local_shadow"]
    for changed, reason in (
        ({"kind": "video"}, "unqualified_modality"),
        ({"model_digest": "future-model"}, "unqualified_model"),
        ({"task": "optional_field_presence_or_absence"}, "known_local_failure_mode"),
        ({"task": "dense_table_exact_row_extraction"}, "known_local_failure_mode"),
        ({"task": "unseen"}, "outside_tested_task_envelope"),
    ):
        decision = recommend(policy, **{**request, **changed}, allow_shadow=True)
        assert decision == {"primary": "native", "local_shadow": False, "reason": reason}
    with pytest.raises(ValueError):
        recommend({**policy, "qualified_local_tasks": ["ocr"]}, **request)


def test_valid_json_does_not_override_truncation_or_required_fields():
    good = {"answer": '{"code":"ERR-123"}', "inference": {"done_reason": "stop"}}
    assert reject_observation(good, ["code"]) is None
    assert reject_observation({**good, "inference": {"done_reason": "length"}}, ["code"]) == (
        "truncated"
    )
    assert reject_observation(good, ["sha"]) == "missing_required_fields"
    assert reject_observation({"answer": "not JSON"}, ["code"]) == "invalid_json"
    assert reject_observation({"answer": '{"code":NaN}'}, ["code"]) == "invalid_json"
