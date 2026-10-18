import asyncio
import json
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from local_vision_mcp import media
from local_vision_mcp.backends import Ollama
from local_vision_mcp.server import Service
from local_vision_mcp.setup import recommend


@pytest.fixture
def service(tmp_path):
    cfg = recommend({"ram_gb": 32, "cuda_vram_gb": 0})
    service = Service(cfg, (tmp_path.resolve(),))
    yield service
    service.close()


@pytest.fixture
def screenshot(tmp_path):
    path = tmp_path / "screen.png"
    Image.new("RGB", (2000, 1000), "white").save(path)
    return path


def test_image_crop_and_text_only(service, screenshot, monkeypatch):
    seen = {}

    def infer(prompt, images, tokens):
        seen.update(prompt=prompt, images=images)
        return "x" * 10000

    monkeypatch.setattr(service.vision, "analyze", infer)
    result = service.analyze(str(screenshot), "Read the button", "image", 64, crop=[0, 0, 500, 500])
    assert result["coverage"]["sent_size"] == (500, 500)
    assert len(result["answer"]) == 512 and result["truncated"]
    assert "untrusted_media" in result
    assert "images" not in result
    assert "Read the button" in seen["prompt"]


def test_path_symlink_escape(service, tmp_path):
    link = tmp_path / "escape"
    link.symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="outside"):
        service.analyze(str(link), "Read", "image")


def test_invalid_bounds(service, screenshot):
    with pytest.raises(ValueError, match="max_tokens"):
        service.analyze(str(screenshot), "Read", "image", 9000)
    with pytest.raises(ValueError, match="crop"):
        service.analyze(str(screenshot), "Read", "image", crop=[0, 0, 9000, 1])
    assert service.pool.status()["active"] == 0
    for start, duration in [(float("nan"), 2), (0, float("inf")), (-1, 1), (0, 121)]:
        with pytest.raises(ValueError):
            media.interval({"format": {"duration": 10}}, start, duration)


def test_disabled_audio(service, screenshot):
    service.config["omni_enabled"] = False
    with pytest.raises(ValueError, match="disabled"):
        service.analyze(str(screenshot), "Listen", "audio")


def test_video_and_audio_real_ffmpeg(service, tmp_path, monkeypatch):
    video = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )

    def infer(prompt, images, tokens, audio):
        assert len(images) == 2
        assert audio.exists()
        assert audio.stat().st_size > 1000
        assert "frame_timestamps" in prompt
        return "Red frames and a steady tone."

    monkeypatch.setattr(service.omni, "analyze", infer)
    result = service.analyze(str(video), "Describe", "video", frames=2, include_audio=True)
    assert result["coverage"]["frame_timestamps_seconds"] == [0.5, 1.5]
    assert result["coverage"]["duration_seconds"] == 2
    assert result["coverage"]["audio_analyzed"]


def test_hardware_profiles():
    cpu = recommend({"ram_gb": 32, "cuda_vram_gb": 0})
    assert cpu["vision_model"] == "qwen3.5:2b" and cpu["omni_device"] == "cpu"
    gpu = recommend({"ram_gb": 32, "cuda_vram_gb": 24})
    assert gpu["vision_model"] == "qwen3.5:4b" and gpu["omni_device"] == "cuda"
    assert not recommend({"ram_gb": 8, "cuda_vram_gb": 0})["omni_enabled"]


def test_ollama_local_only():
    with pytest.raises(ValueError, match="local"):
        Ollama("test", "https://example.com")


def test_stdio_mcp_roundtrip(tmp_path):
    async def check():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "local_vision_mcp"],
            env={"LOCAL_VISION_ROOTS": str(tmp_path)},
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as client,
        ):
            await client.initialize()
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == {
                "analyze_image",
                "analyze_video",
                "analyze_audio",
                "local_media_status",
            }
            result = await client.call_tool("local_media_status", {})
            assert not result.isError
            assert json.loads(result.content[0].text)["allowed_roots"] == [str(tmp_path)]
            bad = await client.call_tool("analyze_image", {"path": "/etc/passwd"})
            assert bad.isError

    asyncio.run(check())


def test_ollama_request_contract(monkeypatch):
    import base64

    import httpx

    def respond(request):
        payload = json.loads(request.content)
        assert payload["stream"] is False and payload["think"] is False
        assert payload["options"]["num_predict"] == 128
        assert payload["messages"][1]["images"] == [base64.b64encode(b"image").decode()]
        return httpx.Response(200, json={"message": {"content": "A red rectangle."}})

    client_class = httpx.Client
    monkeypatch.setattr(
        "local_vision_mcp.backends.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    assert Ollama("test", "http://localhost:11434").analyze("Describe", [b"image"], 128) == (
        "A red rectangle."
    )
