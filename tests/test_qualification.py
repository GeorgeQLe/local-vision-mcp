import hashlib
import json

import pytest

from local_vision_mcp.benchmark.qualification_fixtures import generate
from local_vision_mcp.benchmark.qualification_report import lower_bound, summarize
from local_vision_mcp.benchmark.runner import load_manifest
from local_vision_mcp.benchmark.suite import evaluate


def test_fresh_fixtures_have_no_answers_in_schema_and_preserve_prior_sets(tmp_path):
    manifest = generate(tmp_path / "one", seed=12, per_task=2)
    _document, cases = load_manifest(manifest)
    assert len(cases) == 8
    assert len({c["sha256"] for c in cases}) == 8
    assert len({c["category"] for c in cases}) == 4
    for case in cases:
        schema = case["request"]["response_schema"]
        assert "const" not in json.dumps(schema) and "enum" not in json.dumps(schema)
        assert not {"rubric", "source_group", "category"} & set(schema)
    other = generate(tmp_path / "two", seed=13, per_task=2)
    assert (
        hashlib.sha256(manifest.read_bytes()).digest()
        != hashlib.sha256(other.read_bytes()).digest()
    )
    with pytest.raises(ValueError, match="empty"):
        generate(tmp_path / "one", seed=12)


def test_all_arms_share_validation_and_schema_valid_wrong_values_remain_wrong(tmp_path):
    manifest = generate(tmp_path / "data", seed=14, per_task=1)
    _, cases = load_manifest(manifest)
    case = cases[0]

    class Fake:
        def analyze(self, public):
            assert "rubric" not in public and "category" not in public
            assert public["routing_task"] == case["routing_task"]
            return {"answer": '{"text":"wrong"}'}

    model = {"name": "native", "model": "test", "backend": "codex"}
    row = evaluate(Fake(), model, case, 0)
    assert row["output_validation"]["valid"]
    assert not row["grade"]["passed"]
    report = summarize({"models": [model], "cases": cases, "rows": [row], "status": "Complete"})
    stats = report["models"]["native"]["categories"][case["category"]]
    assert stats["valid_but_wrong"] == 1
    assert stats["contract_or_provider_rejections"] == 0
    assert not report["qualified_for_routing"]


def test_bounds_do_not_confuse_small_perfect_screening_with_qualification():
    assert lower_bound(32, 32) == pytest.approx(0.05 ** (1 / 32))
    assert lower_bound(300, 300) >= 0.99
    assert lower_bound(0, 0) == 0
    assert 0 < lower_bound(9, 10) < 0.9


def test_counting_scene_has_neutral_decorations_and_explicit_scope(tmp_path):
    from PIL import Image

    manifest = generate(tmp_path / "counts", seed=17, per_task=8)
    _, cases = load_manifest(manifest)
    for case in cases:
        if case["category"] == "small_separated_object_counting":
            assert "Exclude the title bar and page background" in case["question"]
            with Image.open(case["path"]) as image:
                title = image.getpixel((0, 0))
                background = image.getpixel((image.width - 1, image.height - 1))
                assert title[0] == title[1] == title[2]
                assert background == (255, 255, 255)
