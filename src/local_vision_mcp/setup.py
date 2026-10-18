"""Hardware-aware configuration and explicit model provisioning."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def hardware() -> dict:
    ram = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) / 1024**2
    except OSError:
        try:
            ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024**3
        except (OSError, subprocess.SubprocessError):
            pass
    vram = 0.0
    try:
        values = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode()
            .splitlines()
        )
        vram = max(float(v) for v in values) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "ram_gb": round(ram, 1),
        "cuda_vram_gb": round(vram, 1),
        "ollama": bool(shutil.which("ollama")),
        "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
    }


def recommend(hw: dict) -> dict:
    # Conservative starting points, not speed or memory guarantees. qwen3.5 beat qwen3-vl on
    # the diagnostic benchmark (RTX 2080 Super); larger GPUs are untested, so stay at 4B.
    vram, ram = hw["cuda_vram_gb"], hw["ram_gb"]
    vision = "qwen3.5:4b" if vram >= 8 else "qwen3.5:2b"
    return {
        "vision_model": vision,
        "vision_backend": "ollama",
        "omni_model": "Qwen/Qwen2.5-Omni-3B",
        "omni_device": "cuda" if vram >= 10 else "cpu",
        "omni_enabled": vram >= 10 or ram >= 16,
        "workers": 1,
        "queue_capacity": 16,
        "queue_timeout": 120,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".local-vision.json"))
    parser.add_argument(
        "--download", action="store_true", help="Pull vision and download Omni weights"
    )
    parser.add_argument("--vision-model")
    parser.add_argument("--omni-model")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    hw = hardware()
    config = recommend(hw)
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    config["workers"] = args.workers
    if args.vision_model:
        config["vision_model"] = args.vision_model
    if args.omni_model:
        config["omni_model"] = args.omni_model
        config["omni_enabled"] = True
    if args.vision_only:
        config["omni_enabled"] = False
    print(json.dumps({"hardware": hw, "selected": config}, indent=2))
    if args.download:
        if not hw["ollama"]:
            parser.error("Install Ollama first: https://ollama.com/download")
        if config["omni_enabled"]:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                parser.error("Install the omni extra first: uv sync --extra omni")
        subprocess.run(["ollama", "pull", config["vision_model"]], check=True)
        if config["omni_enabled"]:
            snapshot_download(config["omni_model"])
    args.config.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Configuration written to {args.config.resolve()}")
    if not hw["ffmpeg"]:
        print("Install FFmpeg (including ffprobe) to analyze video and audio.")
    if config["omni_enabled"] and config["omni_device"] == "cpu":
        print(
            "Omni will use CPU; first load and inference may be slow. Benchmark before relying on it."
        )
