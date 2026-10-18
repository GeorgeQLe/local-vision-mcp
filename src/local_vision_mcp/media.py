"""Bounded local media decoding; URLs are intentionally not accepted."""

import io
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageOps


def local_file(value: str, roots: tuple[Path, ...]) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Media path is outside LOCAL_VISION_ROOTS")
    if not path.is_file() or path.stat().st_size > 2 * 1024**3:
        raise ValueError("Expected a regular file of at most 2 GiB")
    return path


def run(args: list[str]) -> bytes:
    try:
        return subprocess.run(args, check=True, capture_output=True, timeout=60).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("Install FFmpeg and ffprobe for audio/video support") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Media decoding failed: {exc.stderr.decode(errors='replace')[-500:]}"
        ) from exc


def probe(path: Path) -> dict:
    return json.loads(
        run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    )


def interval(info: dict, start: float, duration: float) -> tuple[float, float]:
    total = float(info["format"]["duration"])
    if not all(math.isfinite(v) for v in (start, duration, total)):
        raise ValueError("Times must be finite")
    if start < 0 or start >= total or not 0 < duration <= 120:
        raise ValueError("start must be within media; duration must be in (0, 120] seconds")
    return start, min(duration, total - start)


def image_bytes(source, crop: list[int] | None = None, max_edge: int = 1536) -> tuple[bytes, dict]:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original = image.size
        if crop:
            if len(crop) != 4:
                raise ValueError("crop must be [left, top, right, bottom] in source pixels")
            left, top, right, bottom = crop
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("crop is outside image bounds")
            image = image.crop(tuple(crop))
        image.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), {"original_size": original, "sent_size": image.size, "crop": crop}


def frames(
    path: Path, start: float, duration: float, count: int, max_edge: int = 1024
) -> tuple[list[bytes], list[float]]:
    if not 1 <= count <= 12:
        raise ValueError("frames must be between 1 and 12")
    times = [start + duration * (i + 0.5) / count for i in range(count)]
    images = []
    for timestamp in times:
        raw = run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(timestamp),
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
        )
        if not raw:
            raise ValueError(f"No video frame at {timestamp:.3f}s")
        images.append(raw)
    return images, times


def audio_clip(path: Path, output: Path, start: float, duration: float):
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
    )
