"""Fresh structured-output screening scenes. Synthetic families are not real-source qualification."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TASKS = (
    "clear_large_text_ocr",
    "small_separated_object_counting",
    "explicit_visible_labeled_fields",
    "simple_table_comparison_with_schema_validation",
)


def output_schema(fields):
    return {
        "type": "object",
        "properties": {name: {"type": [kind, "null"]} for name, kind in fields.items()},
        "required": list(fields),
        "additionalProperties": False,
    }


SCHEMAS = dict(
    zip(
        TASKS,
        (
            output_schema({"text": "string"}),
            output_schema({"count": "integer"}),
            output_schema({"value": "string"}),
            output_schema({"service": "string", "requests": "integer"}),
        ),
        strict=True,
    )
)


def truetype(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def generate(output, *, seed, per_task=32, split="screening"):
    if not 1 <= per_task <= 100:
        raise ValueError("per_task must be between 1 and 100")
    if split not in ("development", "screening"):
        raise ValueError("Unknown split")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    rng, cases = random.Random(seed), []
    backgrounds = ("#ffffff", "#f1f5f9", "#fef3c7", "#e0f2fe")
    fonts = ("DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSerif.ttf")
    for task in TASKS:
        for i in range(per_task):
            family = i % 8
            width, height = rng.choice((1000, 1200, 1400)), rng.choice((700, 850, 950))
            background = "#ffffff" if task == TASKS[1] else backgrounds[family % 4]
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            font = truetype(fonts[family % 3], rng.choice((28, 32, 36)))
            small = truetype(fonts[family % 3], 22)
            x = rng.randrange(45, 160) if family % 2 else width // 3
            y = rng.randrange(100, 210)
            draw.rectangle((0, 0, width, 65), fill="#303030" if task == TASKS[1] else "#183153")
            draw.text((30, 15), f"Workstation {rng.randrange(10, 99)}", font=small, fill="white")
            draw.text(
                (30, height - 48),
                f"Updated {rng.randrange(10, 99)} minutes ago",
                font=small,
                fill="#475569",
            )
            if task == TASKS[0]:
                text = (
                    rng.choice(("LOAD", "FAULT", "QUEUE", "READY"))
                    + "-"
                    + str(rng.randrange(1000, 9999))
                )
                draw.text((x, y), "Status message", font=small, fill="#475569")
                draw.text((x, y + 65), text, font=font, fill="#111827")
                question, expected = (
                    "Transcribe the large status message exactly. Return field text (string).",
                    {"text": text},
                )
            elif task == TASKS[1]:
                count, others = rng.randrange(1, 8), rng.randrange(0, 4)
                colors = ["blue"] * count + ["red"] * others
                rng.shuffle(colors)
                # Non-overlapping cells with varied centers and shape; every object is fully visible.
                cells = [
                    (50 + col * (width - 100) // 4, 125 + row * 160)
                    for row in range(3)
                    for col in range(4)
                ]
                rng.shuffle(cells)
                for color, (cx, cy) in zip(colors, cells, strict=False):
                    cx, cy = cx + rng.randrange(0, 20), cy + rng.randrange(0, 15)
                    box = (cx, cy, cx + 70, cy + 65)
                    if family % 3 == 0:
                        draw.ellipse(box, fill=color)
                    else:
                        draw.rectangle(box, fill=color)
                question, expected = (
                    "Count only the separate bright-blue objects in the main content area. Exclude the title bar and page background. Return field count (integer).",
                    {"count": count},
                )
            elif task == TASKS[2]:
                label = rng.choice(("Batch ID", "Coupon code", "Serial number", "Ticket ID"))
                value = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
                draw.text((x, y), label + ":", font=small, fill="#475569")
                draw.text((x, y + 60), value, font=font, fill="#111827")
                draw.text(
                    (x, y + 180),
                    f"Warehouse: {rng.randrange(100, 999)}",
                    font=small,
                    fill="#475569",
                )
                question, expected = (
                    f"Read the explicitly visible {label}. Preserve all characters. Return field value (string).",
                    {"value": value},
                )
            else:
                names = rng.sample(
                    ("Birch", "Atlas", "Pine", "Vega", "Lake", "Delta", "Opal"), rng.randrange(3, 6)
                )
                values = rng.sample(range(100, 950), len(names))
                draw.text((x, y), "Service", font=small, fill="#475569")
                draw.text((x + 280, y), "Requests", font=small, fill="#475569")
                for j, (name, value) in enumerate(zip(names, values, strict=True)):
                    draw.text((x, y + 65 + j * 75), name, font=font, fill="#111827")
                    draw.text((x + 280, y + 65 + j * 75), str(value), font=font, fill="#111827")
                maximum = max(values)
                question = "Which service has the highest Requests value? Return fields service (string) and requests (integer)."
                expected = {"service": names[values.index(maximum)], "requests": maximum}
            path = output / f"{len(cases):04}.png"
            image.save(path)
            cases.append(
                {
                    "id": f"{split}-{len(cases):04}",
                    "kind": "image",
                    "category": task,
                    "routing_task": task,
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "question": question
                    + " Use null for any unreadable field. Return only the requested JSON object.",
                    "request": {"max_tokens": 128, "response_schema": SCHEMAS[task]},
                    "rubric": {"expected": expected},
                    "source_group": f"synthetic-family-{family}",
                }
            )
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "synthetic": True,
                "seed": seed,
                "split": split,
                "description": "New scenes with related generator families; not independent real-source qualification.",
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-task", type=int, default=32)
    parser.add_argument("--split", choices=("development", "screening"), default="screening")
    args = parser.parse_args()
    print(generate(args.output, seed=args.seed, per_task=args.per_task, split=args.split))


if __name__ == "__main__":
    main()
