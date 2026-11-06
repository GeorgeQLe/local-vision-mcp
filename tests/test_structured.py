import json

import httpx
import pytest

from local_vision_mcp.backends import Ollama
from local_vision_mcp.structured import prepare_schema, validate_answer

SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": ["integer", "null"]}},
    "required": ["count"],
    "additionalProperties": False,
}


def test_schema_contract_rejects_wrong_keys_types_duplicates_and_truncation():
    schema = prepare_schema(SCHEMA)
    assert validate_answer('{"count":2}', schema)["valid"]
    assert validate_answer('{"count":null}', schema)["valid"]
    for answer in (
        '{"count":true}',
        '{"count":"2"}',
        '{"count":2,"extra":0}',
        '{"total":2}',
        '{"count":NaN}',
        '{"count":1,"count":2}',
    ):
        assert not validate_answer(answer, schema)["valid"]
    assert validate_answer('{"count":2}', schema, truncated=True)["reason"] == "truncated"
    schema["required"].clear()
    assert SCHEMA["required"] == ["count"]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"a": {"$ref": "https://example.com/schema"}}},
        {"type": "array"},
        {"type": "object", "required": "wrong"},
        {"type": "object", "description": "x" * 32768},
    ],
)
def test_invalid_or_remote_schemas_are_rejected(schema):
    with pytest.raises(ValueError):
        prepare_schema(schema)


def test_ollama_uses_schema_constraint_and_grounding_prompt(monkeypatch):
    def respond(request):
        data = json.loads(request.content)
        assert data["format"] == SCHEMA
        assert "output schema" in data["messages"][1]["content"]
        assert data["options"]["num_predict"] == 64
        return httpx.Response(200, json={"message": {"content": '{"count":2}'}})

    client_class = httpx.Client
    monkeypatch.setattr(
        "local_vision_mcp.backends.httpx.Client",
        lambda **kw: client_class(transport=httpx.MockTransport(respond), **kw),
    )
    assert (
        Ollama("test", "http://localhost:11434").analyze("Count", [b"pixels"], 64, SCHEMA)
        == '{"count":2}'
    )
