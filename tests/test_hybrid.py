import json
import threading

import pytest

from local_vision_mcp.benchmark import hybrid


def make_provider(tmp_path, monkeypatch, *, fail_local=False, task="candidate", digest="tested"):
    policy = {
        "version": 1,
        "reference_model_digest": "tested",
        "qualified_local_tasks": [],
        "shadow_modalities": ["image"],
        "shadow_candidates": ["candidate"],
        "native_required_tasks": ["tiny"],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    calls = []

    class Fake:
        def __init__(self, model, *args):
            self.local = model["backend"] == "ollama"
            self.metadata = {"runtime": {"model": {"digest": digest}}}

        def analyze(self, case):
            assert "rubric" not in case
            calls.append("local" if self.local else "native")
            if self.local and fail_local:
                raise TimeoutError("local timeout")
            return {
                "answer": '{"count":999}' if self.local else '{"count":2}',
                "usage": {"input_tokens": 10, "output_tokens": 3},
            }

        def close(self):
            pass

    monkeypatch.setattr(hybrid, "Provider", Fake)
    provider = hybrid.ShadowProvider(
        {
            "backend": "codex",
            "model": "native",
            "local_model": "local",
            "routing_policy": str(path),
        },
        tmp_path,
        1,
        threading.Event(),
    )
    case = {"path": "x", "kind": "image", "question": "Count", "routing_task": task, "request": {}}
    return provider, case, calls


def test_shadow_never_accepts_wrong_local_value_or_mixes_tokenizers(tmp_path, monkeypatch):
    provider, case, calls = make_provider(tmp_path, monkeypatch)
    result = provider.analyze(case)
    assert calls == ["local", "native"]
    assert result["answer"] == '{"count":2}'
    assert result["local_observation"]["answer"] == '{"count":999}'
    assert result["usage"]["input_tokens"] == 10
    assert result["accepted_local"] is False
    assert result["native_called"] is True


def test_local_failure_still_runs_native_and_retains_consumed_work(tmp_path, monkeypatch):
    provider, case, calls = make_provider(tmp_path, monkeypatch, fail_local=True)
    result = provider.analyze(case)
    assert calls == ["local", "native"]
    assert result["local_rejection"] == "provider_error"
    assert result["answer"] == '{"count":2}'
    assert result["local_seconds"] >= 0


@pytest.mark.parametrize(
    "task,digest", [("tiny", "tested"), ("unknown", "tested"), ("candidate", "new")]
)
def test_failure_tasks_unknown_requests_and_models_skip_local(tmp_path, monkeypatch, task, digest):
    provider, case, calls = make_provider(tmp_path, monkeypatch, task=task, digest=digest)
    result = provider.analyze(case)
    assert calls == ["native"]
    assert result["local_observation"] is None
