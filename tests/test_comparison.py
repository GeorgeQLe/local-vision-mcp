import base64
import hashlib
import io
import json
import sys
import threading
import time
from typing import ClassVar

import pytest
from PIL import Image
from rich.console import Console

from local_vision_mcp.benchmark import suite
from local_vision_mcp.benchmark.dashboard import aggregate, render
from local_vision_mcp.benchmark.providers import (
    Provider,
    cli_request,
    mcp_evidence,
    parse_events,
    run_process,
    token_cost,
)


@pytest.fixture
def case(tmp_path):
    path = tmp_path / "0.png"
    Image.new("RGB", (50, 40), "red").save(path)
    return {
        "id": "red",
        "path": str(path),
        "kind": "image",
        "category": "color",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "question": 'Return JSON with the visible "color".',
        "request": {"crop": [0, 0, 20, 20], "max_tokens": 64},
        "rubric": {"expected": {"color": "GROUND_TRUTH_SECRET"}},
    }


def model(backend="codex"):
    return {
        "name": backend,
        "backend": backend,
        "model": "test-model",
        "pricing": {"input": 1, "cached_input": 0.1, "output": 10},
        "estimated_tokens": {"input_tokens": 1000, "output_tokens": 100},
    }


def test_cost_cache_accounting_and_unknowns():
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "output_tokens": 100,
        "reasoning_output_tokens": 90,
    }
    assert token_cost(usage, model()["pricing"]) == pytest.approx(0.00164)
    assert token_cost(usage, {"input": 1, "output": 10}) is None
    assert token_cost({"input_tokens": 0, "output_tokens": 0}, {"input": 0}) == 0
    assert token_cost({"input_tokens": 2}, model()["pricing"]) is None
    assert token_cost(None, model()["pricing"]) is None
    assert token_cost({**usage, "cached_input_tokens": 1001}, model()["pricing"]) is None


def test_native_event_parsers():
    output = "\n".join(
        json.dumps(item)
        for item in [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"color":"red"}'},
            },
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 2}},
        ]
    )
    assert parse_events("codex", output)["usage"]["input_tokens"] == 100
    assert parse_events("codex", output)["resolved_model"] is None
    with pytest.raises(RuntimeError, match="successful turn"):
        parse_events("codex", output + '\n{"type":"turn.failed"}')
    claude = {
        "type": "result",
        "subtype": "success",
        "result": '{"color":"red"}',
        "usage": {
            "input_tokens": 20,
            "cache_read_input_tokens": 60,
            "cache_creation_input_tokens": 10,
            "output_tokens": 5,
        },
        "modelUsage": {"resolved-name": {}},
        "total_cost_usd": 0.03,
    }
    result = parse_events("claude", json.dumps(claude))
    assert result["usage"]["input_tokens"] == 90
    assert result["resolved_model"] == ["resolved-name"]
    assert result["reported_cost_usd"] == 0.03
    assert token_cost(result["usage"], model()["pricing"]) is None  # cache-write price missing
    with pytest.raises(RuntimeError, match="successful result"):
        parse_events("claude", json.dumps({**claude, "is_error": True}))
    with pytest.raises(RuntimeError):
        parse_events("codex", "not a protocol response")


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_cli_payload_contains_pixels_without_ground_truth(case, tmp_path, backend):
    argv, stdin, provenance = cli_request(model(backend), case, tmp_path)
    assert "GROUND_TRUTH_SECRET" not in stdin
    assert "rubric" not in stdin
    assert provenance["coverage"]["sent_size"] == (20, 20)
    if backend == "claude":
        blocks = json.loads(stdin)["message"]["content"]
        image = base64.b64decode(blocks[1]["source"]["data"])
        assert argv[argv.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in argv
    else:
        image = (tmp_path / "frame-0.png").read_bytes()
        assert argv[argv.index("--image") + 1] == str(tmp_path / "frame-0.png")
        assert "--ignore-user-config" in argv
        assert "features.shell_tool=false" in argv
        assert "tools.view_image=false" in argv
        assert argv[-1] == "-"
    assert Image.open(io.BytesIO(image)).size == (20, 20)
    assert provenance["image_sha256"] == [hashlib.sha256(image).hexdigest()]
    (tmp_path / "0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Media changed"):
        cli_request(model(backend), case, tmp_path)


def test_subprocess_deadline_cancel_and_redaction(tmp_path):
    stop = threading.Event()
    assert (
        run_process([sys.executable, "-c", "print(input())"], "hello\n", tmp_path, 3, stop).strip()
        == "hello"
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], "", tmp_path, 0.15, stop)
    assert time.monotonic() - started < 3
    stop.set()
    with pytest.raises(InterruptedError):
        run_process([sys.executable, "-c", "import time; time.sleep(10)"], "", tmp_path, 10, stop)
    stop.clear()
    with pytest.raises(RuntimeError) as error:
        run_process(
            [sys.executable, "-c", "import sys; sys.stderr.write('SECRET'); sys.exit(2)"],
            "",
            tmp_path,
            3,
            stop,
        )
    assert "SECRET" not in str(error.value)


def test_large_image_input_survives_a_delayed_reader(tmp_path):
    payload = "base64-image-data" * 131072
    argv = [
        sys.executable,
        "-c",
        (
            "import hashlib,sys,time; time.sleep(.25); data=sys.stdin.buffer.read(); "
            "print(len(data),hashlib.sha256(data).hexdigest())"
        ),
    ]
    result = run_process(argv, payload, tmp_path, 3, threading.Event()).strip()
    assert result == f"{len(payload.encode())} {hashlib.sha256(payload.encode()).hexdigest()}"


def test_local_provider_applies_requested_inference_options(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "local_vision_mcp.benchmark.runner.ollama_metadata", lambda service: {"local": True}
    )
    candidate = {
        "name": "local",
        "backend": "ollama",
        "model": "test",
        "ollama_options": {"num_ctx": 4096, "temperature": 0.25},
    }
    provider = Provider(candidate, tmp_path, 15, threading.Event())
    try:
        assert provider.service.vision.options["num_ctx"] == 4096
        assert provider.service.vision.options["temperature"] == 0.25
    finally:
        provider.close()


@pytest.mark.parametrize("backend", ["codex", "claude"])
def test_mcp_arm_has_no_pixel_attachment_and_only_preapproves_analysis(case, tmp_path, backend):
    candidate = {**model(backend), "perception": "mcp", "local_model": "qwen3.5:4b"}
    argv, stdin, provenance = cli_request(candidate, case, tmp_path)
    assert "GROUND_TRUTH_SECRET" not in stdin
    assert provenance["perception"] == "mcp"
    if backend == "codex":
        assert "--image" not in argv
        assert 'mcp_servers.local-vision.tools.analyze_image.approval_mode="approve"' in argv
        assert "mcp_servers.local-vision.required=true" in argv
        prompt = stdin
    else:
        content = json.loads(stdin)["message"]["content"]
        assert [block["type"] for block in content] == ["text"]
        assert argv[argv.index("--allowedTools") + 1] == "mcp__local-vision__analyze_image"
        server = json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]["local-vision"]
        assert "--in-process" in server["args"]
        prompt = content[0]["text"]
    assert json.dumps(case["request"]["crop"]) in prompt
    assert case["path"] in prompt


def test_mcp_evidence_rejects_advertised_failed_and_unrelated_calls():
    call = {
        "type": "mcp_tool_call",
        "server": "local-vision",
        "tool": "analyze_image",
        "status": "completed",
        "result": {"content": []},
        "error": None,
    }
    event = lambda item: json.dumps({"type": "item.completed", "item": item})
    assert mcp_evidence("codex", event(call), "image")["mcp_successful_calls"] == 1
    for output in (
        "",
        event({**call, "tool": "local_media_status"}),
        event({**call, "status": "failed", "error": {"message": "denied"}}),
        event({**call, "result": {"isError": True}}),
    ):
        with pytest.raises(RuntimeError, match="No successful"):
            mcp_evidence("codex", output, "image")
    use = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "id": "call-1", "name": "mcp__local-vision__analyze_image"}
            ]
        },
    }
    result = {
        "type": "user",
        "message": {
            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "report"}]
        },
    }
    assert (
        mcp_evidence("claude", json.dumps(use) + "\n" + json.dumps(result), "image")[
            "mcp_successful_calls"
        ]
        == 1
    )
    result["message"]["content"][0]["is_error"] = True
    with pytest.raises(RuntimeError, match="No successful"):
        mcp_evidence("claude", json.dumps(use) + "\n" + json.dumps(result), "image")


class FakeProvider:
    seen: ClassVar[list] = []
    closed: ClassVar[list] = []

    def __init__(self, candidate, root, timeout, stop):
        self.model, self.stop = candidate, stop
        self.metadata = {"fake": True}

    def analyze(self, public):
        assert "rubric" not in public
        assert "expected" not in json.dumps(public)
        self.seen.append(public)
        return {"answer": '{"color":"red"}', "estimated_cost_usd": 0.002}

    def close(self):
        self.closed.append(self.model["name"])


def test_runner_grades_persists_failures_and_never_leaks(case, tmp_path):
    output = tmp_path / "results"
    code = suite.run_suite([model()], [case], output, tui="off", provider_factory=FakeProvider)
    assert code == 1  # red differs from hidden expected string
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "Complete"
    assert report["summary"]["codex"]["accuracy"] == 0
    assert report["categories"]["codex"]["color"]["completed"] == 1
    events = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "completed"]
    with pytest.raises(ValueError, match="never overwritten"):
        suite.run_suite([model()], [case], output, tui="off", provider_factory=FakeProvider)


def test_soft_budget_reserves_active_requests(case, tmp_path):
    output = tmp_path / "budget"
    code = suite.run_suite(
        [model()],
        [case],
        output,
        repeats=4,
        parallel=3,
        max_cost=0.003,
        tui="off",
        provider_factory=FakeProvider,
    )
    report = json.loads((output / "report.json").read_text())
    assert code == 2
    assert report["status"] == "Budget stopped"
    assert len(report["rows"]) == 1
    with pytest.raises(ValueError, match="requires pricing"):
        suite.run_suite(
            [{**model(), "pricing": {}}],
            [case],
            tmp_path / "unknown",
            max_cost=1,
            tui="off",
            provider_factory=FakeProvider,
        )


def test_errors_are_failed_cases_and_cost_is_unknown(case, tmp_path):
    class Broken(FakeProvider):
        def analyze(self, public):
            raise TimeoutError("stalled")

    output = tmp_path / "broken"
    code = suite.run_suite([model()], [case], output, tui="off", provider_factory=Broken)
    stats = json.loads((output / "report.json").read_text())["summary"]["codex"]
    assert code == 1
    assert stats["errors"] == stats["unknown_cost_requests"] == 1
    assert stats["accuracy"] == 0


def test_failed_mcp_validation_preserves_consumed_usage(case):
    class MissingCall:
        def analyze(self, public):
            return {
                "answer": '{"color":"red"}',
                "error": "No successful local-vision perception tool call recorded",
                "usage": {"input_tokens": 123, "output_tokens": 10},
            }

    case["rubric"]["expected"]["color"] = "red"
    row = suite.evaluate(MissingCall(), model(), case, 0)
    assert not row["grade"]["passed"]
    assert row["grade"]["score"] == 0
    assert row["usage"]["input_tokens"] == 123


def test_interrupt_drains_and_preserves_partial_results(case, tmp_path, monkeypatch):
    real_wait = suite.wait
    first = True

    def interrupt_once(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise KeyboardInterrupt
        return real_wait(*args, **kwargs)

    class WaitForStop(FakeProvider):
        def analyze(self, public):
            assert self.stop.wait(3)
            raise InterruptedError("cancelled")

    monkeypatch.setattr(suite, "wait", interrupt_once)
    output = tmp_path / "interrupted"
    code = suite.run_suite(
        [model()], [case], output, repeats=5, tui="off", provider_factory=WaitForStop
    )
    report = json.loads((output / "report.json").read_text())
    assert code == 130
    assert report["status"] == "Interrupted"
    assert len(report["rows"]) == 1
    assert report["rows"][0]["error"].startswith("InterruptedError")


def test_setup_failure_leaves_report_and_closes_ready_providers(case, tmp_path):
    class FailSecond(FakeProvider):
        def __init__(self, candidate, *args):
            if candidate["name"] == "bad":
                raise RuntimeError("missing CLI")
            super().__init__(candidate, *args)

    FakeProvider.closed = []
    output = tmp_path / "setup-failed"
    with pytest.raises(RuntimeError, match="missing CLI"):
        suite.run_suite(
            [model(), {**model(), "name": "bad"}],
            [case],
            output,
            tui="off",
            provider_factory=FailSecond,
        )
    assert FakeProvider.closed == ["codex"]
    assert json.loads((output / "report.json").read_text())["status"] == "Setup/execution failed"


def test_shared_case_selection_and_config_validation(case, tmp_path):
    audio = {**case, "id": "audio", "kind": "audio"}
    assert suite.select_cases([case, audio], [model()]) == [case]
    with pytest.raises(ValueError, match="requires all models"):
        suite.select_cases([case, audio], [model()], kinds=["audio"])
    path = tmp_path / "config.json"
    for value in (
        {},
        {"models": []},
        {"models": [model(), model()]},
        {"models": [{**model(), "pricing": {"input": -1}}]},
        {"models": [{**model(), "pricing": {"typo": 1}}]},
        {"models": [{**model(), "estimated_tokens": {"typo": 1}}]},
        {"models": [{**model(), "perception": "unknown"}]},
        {"models": [{**model(), "perception": "mcp"}]},
        {"models": [{**model("ollama"), "perception": "mcp", "local_model": "local"}]},
    ):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            suite.load_models(path)


def test_dashboard_renders_real_stats_with_unknown_cost():
    row = {
        "model_name": "codex",
        "case_id": "red",
        "grade": {"passed": True, "score": 1},
        "seconds": 2,
        "estimated_cost_usd": None,
    }
    state = {
        "rows": [row],
        "models": [model()],
        "total": 2,
        "per_model": 2,
        "started": time.monotonic() - 2,
        "active": {},
        "projected_cost": None,
        "status": "Running",
    }
    console = Console(file=io.StringIO(), width=160, color_system=None)
    console.print(render(state))
    text = console.file.getvalue()
    assert "100.0%" in text
    assert "1 unknown" in text
    assert "1/2 completed" in text
    assert "planned total: unknown" in text
    assert aggregate([row])["unknown_cost_requests"] == 1
