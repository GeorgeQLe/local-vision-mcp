#!/usr/bin/env python3
"""Checkout bootstrap: host agents can run this before the MCP server exists."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    source = Path(__file__).resolve().parents[1]
    raise SystemExit(
        subprocess.call(
            [
                "uv",
                "run",
                "--project",
                str(source),
                "local-vision",
                *sys.argv[1:],
            ]
        )
    )
