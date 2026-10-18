import json

import pytest

from local_vision_mcp.benchmark import runner
from local_vision_mcp.benchmark.fixtures import generate
from local_vision_mcp.benchmark.scoring import score
from local_vision_mcp.server import Service
from local_vision_mcp.setup import recommend


def test_grading_handles_wrong_types_missing_fields_and_injection():
    rubric = {"expected": {"code": "ERR-1234", "count": 3, "speech": False, "missing": None}}
    assert score(
        '```json\n{"code":"err-1234","count":3,"speech":false,"missing":null}\n```', rubric
    )["passed"]
    assert not score('{"code":"ERR-1234","count":true,"speech":0,"missing":null}', rubric)["passed"]
    assert not score("The answer is ERR-1234", rubric)["passed"]
    assert not score('{"code":"ERR-1234"}', rubric)["passed"]
    assert score('{"value":4.1}', {"expected": {"value": 4}, "tolerance": {"value": 0.2}})["passed"]
    assert not score(
        '{"x":"ok","extra":"OVERRIDE"}', {"expected": {"x": "ok"}, "forbidden_regex": ["override"]}
    )["passed"]


def test_fixtures_reproducible_and_media_hashes_enforced(tmp_path):
    first = generate(tmp_path / "a", seed=1, variants=1, audiovisual=False)
    second = generate(tmp_path / "b", seed=1, variants=1, audiovisual=False)
    _, a = runner.load_manifest(first)
    _, b = runner.load_manifest(second)
    assert len(a) == 8
    assert runner.workload_key(a) == runner.workload_key(b)
    from pathlib import Path

    Path(a[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_manifest(first)


def test_benchmark_records_errors_and_scores_without_leaking_answers(tmp_path, monkeypatch):
    manifest = generate(tmp_path / "data", variants=1, audiovisual=False)
    _, cases = runner.load_manifest(manifest)
    config = recommend({"ram_gb": 32, "cuda_vram_gb": 0})
    config["workers"] = 2
    calls = []

    def factory(config, roots):
        service = Service(config, roots)

        def answer(prompt, images, tokens):
            calls.append(prompt)
            assert cases[0]["rubric"]["expected"]["code"] not in prompt
            if "greatest" in prompt:
                raise RuntimeError("simulated backend failure")
            return json.dumps(cases[0]["rubric"]["expected"])

        service.vision.analyze = answer
        return service

    monitor_class = runner.Monitor
    monkeypatch.setattr(runner, "Monitor", lambda url: monitor_class(None))
    result = runner.run_scenario(
        config,
        cases[:2],
        agents=3,
        repeats=2,
        warmup=1,
        seed=1,
        root=manifest.parent,
        preflight=False,
        service_factory=factory,
    )
    assert result["summary"]["requests"] == 4
    assert result["summary"]["errors"] == 2
    assert result["summary"]["pass_rate"] == 0.5
    assert len(result["warmup"]) == 1
    assert all("queue_seconds" in r["result"] for r in result["rows"] if "result" in r)
    assert len(calls) == 5


def test_recommendations_require_accuracy_latency_and_equal_workloads():
    def trial(i, workload, accuracy, latency, errors=0):
        return {
            "id": i,
            "workload": workload,
            "profile": "native",
            "agents": 4,
            "status": "completed",
            "summary": {
                "requests": 20,
                "errors": errors,
                "pass_rate": accuracy,
                "p95_latency_seconds": latency,
                "correct_per_second": 1 / latency,
            },
        }

    trials = [
        trial(0, "image", 0.5, 1),
        trial(1, "image", 0.95, 2),
        trial(2, "image", 1, 100),
        trial(3, "audio", 1, 1),
        trial(4, "image", 1, 1, 1),
    ]
    assert {r["trial_id"] for r in runner.recommendations(trials, 0.9, 10)} == {1, 3}
    assert not runner.recommendations(trials[:1], 0.9, 10)
