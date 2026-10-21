"""Conservative, offline routing recommendations; not installed-server dispatch.

Task labels must come from the production request, never the evaluator's rubric.
Local shadow observations never replace the primary native answer.
"""

import json


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def recommend(policy, *, task, kind, model_digest, allow_shadow=False):
    if policy.get("version") != 1 or policy.get("qualified_local_tasks"):
        raise ValueError("This conservative policy supports no promoted local tasks")
    decision = {"primary": "native", "local_shadow": False}
    if model_digest != policy["reference_model_digest"]:
        return {**decision, "reason": "unqualified_model"}
    if kind not in policy["shadow_modalities"]:
        return {**decision, "reason": "unqualified_modality"}
    if task in policy["native_required_tasks"]:
        return {**decision, "reason": "known_local_failure_mode"}
    if task not in policy["shadow_candidates"]:
        return {**decision, "reason": "outside_tested_task_envelope"}
    return {
        **decision,
        "local_shadow": bool(allow_shadow),
        "reason": "candidate_requires_qualification",
    }


def reject_observation(observation, required_fields):
    """Return a rejection reason; valid structure is not proof of correct pixels."""
    if observation.get("error"):
        return "provider_error"
    if observation.get("truncated") or (
        observation.get("inference", {}).get("done_reason") == "length"
    ):
        return "truncated"
    if observation.get("output_validation", {}).get("valid") is False:
        return "output_contract_rejected"
    answer = observation.get("answer")
    if not isinstance(answer, str):
        return "missing_answer"
    text = answer.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        return "invalid_json"
    if not isinstance(value, dict) or any(key not in value for key in required_fields):
        return "missing_required_fields"
    return None
