"""Caller-supplied output contracts, independent of any benchmark answer key."""

import json
import math

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def prepare_schema(schema):
    """Snapshot and validate a bounded, self-contained JSON object schema."""
    if schema is None:
        return None
    try:
        encoded = json.dumps(schema, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("response_schema must contain finite JSON values") from exc
    if len(encoded.encode()) > 32768:
        raise ValueError("response_schema must be at most 32 KiB")
    schema = json.loads(encoded)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("response_schema must have type object")

    def check(node, depth=0):
        if depth > 16:
            raise ValueError("response_schema nesting exceeds 16 levels")
        if isinstance(node, dict):
            if any(k in node for k in ("$ref", "$dynamicRef", "$recursiveRef")):
                raise ValueError("response_schema references are unsupported; inline definitions")
            for value in node.values():
                check(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                check(value, depth + 1)

    check(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError("Invalid response_schema") from exc
    return schema


def validate_answer(answer, schema, *, truncated=False):
    """Validate structure only. A valid result may still misread the media."""
    if truncated:
        return {"valid": False, "reason": "truncated"}

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        return value

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            answer, parse_constant=constant, parse_float=finite_float, object_pairs_hook=pairs
        )
    except (ValueError, TypeError):
        return {"valid": False, "reason": "invalid_json"}
    if not Draft202012Validator(schema).is_valid(value):
        return {"valid": False, "reason": "schema_mismatch"}
    return {"valid": True, "reason": "schema_valid"}
