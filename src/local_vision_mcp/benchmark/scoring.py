"""Deterministic task checks; no paid model judge or answer-leaking prompts."""

import json
import math
import re


def normalize(value):
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value


def equivalent(actual, expected, tolerance=0):
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and abs(actual - expected) <= tolerance
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(equivalent(a, e, tolerance) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and equivalent(actual[k], e, tolerance) for k, e in expected.items()
        )
    return isinstance(actual, str) and normalize(actual) == normalize(expected)


def score(answer: str, rubric: dict) -> dict:
    """Score expected JSON fields; extra fields are allowed, missing fields fail."""
    text = answer.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)[:-3].strip()
    try:
        actual = json.loads(text)
    except (ValueError, TypeError):
        return {"score": 0.0, "passed": False, "reason": "invalid_json", "fields": {}}
    expected = rubric["expected"]
    fields = {
        key: isinstance(actual, dict)
        and key in actual
        and equivalent(actual[key], value, rubric.get("tolerance", {}).get(key, 0))
        for key, value in expected.items()
    }
    forbidden = any(
        re.search(pattern, answer, re.IGNORECASE) for pattern in rubric.get("forbidden_regex", [])
    )
    value = sum(fields.values()) / len(fields) if fields else 0.0
    return {
        "score": 0.0 if forbidden else value,
        "passed": bool(fields) and all(fields.values()) and not forbidden,
        "fields": fields,
        "reason": "forbidden_content" if forbidden else "scored",
    }
