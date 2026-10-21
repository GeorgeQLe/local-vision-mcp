import json
import threading

import pytest
from PIL import Image

from local_vision_mcp.benchmark.consensus import ConsensusService, parse_answer, reduce_votes
from local_vision_mcp.benchmark.consensus_report import summarize
from local_vision_mcp.benchmark.efficiency import local_usage
from local_vision_mcp.benchmark.providers import cli_request


def test_exact_votes_and_invalid_answers():
    a = parse_answer('```json\n{"build":"build-123"}\n```')
    assert reduce_votes([{"canonical": a}] * 3)["status"] == "unanimous"
    majority = reduce_votes([{"canonical": a}, {"canonical": a}, {"canonical": None}])
    assert majority["status"] == "majority"
    assert reduce_votes([{"canonical": a}, {"canonical": '{"build":"123"}'}])["answer"] is None
    for bad in ("[]", "{}", '{"x":NaN}', "uncertain"):
        with pytest.raises(ValueError):
            parse_answer(bad)


def test_parallel_views_then_verifier_and_private_audit(tmp_path, monkeypatch):
    barrier = threading.Barrier(2, timeout=5)
    calls, lock = [], threading.Lock()

    class Vision:
        def analyze(self, prompt, images, tokens):
            if "Candidate strings" not in prompt:
                barrier.wait()
            else:
                assert len(calls) == 2
            with lock:
                calls.append(prompt)
            return '{"build":"build-123"}'

        def last_metrics(self):
            return {"prompt_eval_count": 100, "eval_count": 10, "done_reason": "stop"}

    service = object.__new__(ConsensusService)
    service.vision, service.roots = Vision(), (tmp_path,)
    service.image_edge, service.config = 1536, {"vision_model": "fake"}
    service.audit_lock = threading.Lock()
    path = tmp_path / "image.png"
    Image.new("RGB", (200, 100), "red").save(path)
    audit_path = tmp_path / "private.jsonl"
    monkeypatch.setenv("LOCAL_VISION_CONSENSUS_AUDIT", str(audit_path))
    result = service._analyze_media(
        str(path), "Read build as JSON.", "image", 128, 0, 30, 6, False, None, None, None
    )
    assert result["consensus"]["status"] == "unanimous"
    assert "passes" not in result and "Candidate strings" not in json.dumps(result)
    assert result["inference"]["prompt_eval_count"] == 300
    assert local_usage({"mcp_calls": [{"result": result}]})["calls_with_usage"] == 3
    audit = json.loads(audit_path.read_text())
    assert len(audit["passes"]) == 3
    assert audit["request_overlap_seconds"] > 0
    assert audit["passes"][0]["image_sha256"] != audit["passes"][1]["image_sha256"]


def test_truncated_pass_does_not_vote():
    class Vision:
        def analyze(self, *args):
            return '{"x":1}'

        def last_metrics(self):
            return {"done_reason": "length", "eval_count": 128}

    service = object.__new__(ConsensusService)
    service.vision = Vision()
    result = service._pass("test", "question", b"fake", 128)
    assert result["canonical"] is None
    assert result["inference"]["eval_count"] == 128


def test_consensus_mcp_payload_keeps_rubric_private(tmp_path):
    import hashlib

    path = tmp_path / "image.png"
    Image.new("RGB", (50, 40), "red").save(path)
    model = {
        "backend": "codex",
        "model": "test",
        "perception": "mcp",
        "local_model": "local",
        "local_strategy": "consensus",
    }
    case = {
        "path": str(path),
        "kind": "image",
        "question": "Read build as JSON.",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rubric": {"expected": {"secret": "NEVER_SEND_THIS"}},
    }
    argv, stdin, _ = cli_request(model, case, tmp_path)
    assert "local_vision_mcp.benchmark.consensus" in " ".join(argv)
    assert "NEVER_SEND_THIS" not in stdin + " ".join(argv)
    assert "--image" not in argv


def test_report_counts_wrong_agreement_and_separates_followups():
    case = {"id": "test", "question": "original", "rubric": {"expected": {"code": "right"}}}
    wrong = {
        "question": "original",
        "decision": {"status": "unanimous", "answer": '{"code":"wrong"}'},
        "passes": [{"started": 1, "ended": 2, "canonical": "wrong"}] * 3,
        "request_overlap_seconds": 1,
    }
    row = {
        "model_name": "consensus",
        "case_id": "test",
        "repeat": 0,
        "seconds": 2,
        "grade": {"passed": False},
        "local_consensus_audit": [wrong, {**wrong, "question": "other request"}],
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    report = {
        "status": "Complete",
        "cases": [case],
        "rows": [row],
        "models": [
            {
                "name": "consensus",
                "backend": "codex",
                "perception": "mcp",
                "local_strategy": "consensus",
            }
        ],
    }
    arm = summarize(report)["arms"][0]
    assert arm["wrong_unanimous_case_ids"] == ["test"]
    assert arm["unscored_followup_unanimous_calls"] == 1
    assert arm["summed_first_two_request_overlap_seconds"] == 2
    assert arm["host_total_tokens"] == 110
    assert arm["first_local_consensus"] == {
        "graded_requests": 1,
        "passed": 0,
        "expected_requests": 1,
    }
    row["grade"]["passed"] = True  # A host can correct a wrong local vote.
    arm = summarize(report)["arms"][0]
    assert arm["passed"] == 1
    assert arm["first_local_consensus"]["passed"] == 0


def test_consensus_rejects_unimplemented_schema_before_inference():
    service = object.__new__(ConsensusService)
    with pytest.raises(ValueError, match="does not support response_schema"):
        service._analyze_media(
            "x", "q", "image", 128, 0, 30, 6, False, None, response_schema={"type": "object"}
        )
