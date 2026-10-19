"""Host-agent bootstrap and diagnosis, independent of an MCP connection."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_mcp_foundation import launcher
from agentic_mcp_foundation.publication import publish, recover
from agentic_mcp_foundation.transport import atomic_json, file_lock

from .coordinator import RemoteService
from .setup import hardware, recommend


def home_directory():
    return Path(
        os.getenv(
            "LOCAL_VISION_HOME",
            os.getenv(
                "AGENTIC_MCP_HOME",
                str(
                    Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
                    / "local-vision-mcp"
                ),
            ),
        )
    )


def settings(home=None):
    home = home or home_directory()
    config = recommend(hardware())
    roots = [str(Path.cwd())]
    path = home / "settings.json"
    if path.exists():
        document = json.loads(path.read_text())
        config.update(document["config"])
        roots = document["roots"]
    override = (
        Path(os.environ["LOCAL_VISION_CONFIG"])
        if os.getenv("LOCAL_VISION_CONFIG")
        else (None if path.exists() else Path(".local-vision.json"))
    )
    if override is not None and override.exists():
        config.update(json.loads(override.read_text()))
    if os.getenv("LOCAL_VISION_ROOTS"):
        roots = os.environ["LOCAL_VISION_ROOTS"].split(os.pathsep)
    resolved = tuple(Path(p).expanduser().resolve(strict=True) for p in roots if p)
    if not resolved or any(not p.is_dir() for p in resolved):
        raise ValueError("Configure at least one existing media root directory")
    return config, resolved


async def smoke(python, config, roots, runtime, launch=None, home=None):
    """Probe the actual installed stdio command, not the checkout's Python imports."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=launch["command"] if launch else str(python),
        args=launch["args"] if launch else ["-m", "local_vision_mcp"],
        cwd=str(roots),
        env={
            "LOCAL_VISION_CONFIG": str(config),
            "LOCAL_VISION_ROOTS": str(roots),
            "LOCAL_VISION_RUNTIME_DIR": str(runtime),
            **({"LOCAL_VISION_HOME": str(home)} if home is not None else {}),
        },
    )
    try:
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
            initialized = await client.initialize()
            tools = await client.list_tools()
            required = {
                "analyze_image",
                "analyze_video",
                "analyze_audio",
                "local_media_manage",
                "local_media_status",
                "submit_media_job",
                "get_media_job",
                "cancel_media_job",
                "verify_local_media",
            }
            if {t.name for t in tools.tools} != required:
                raise RuntimeError("Installed tool catalog differs from the released contract")
            status = await client.call_tool("local_media_status", {})
            if status.isError:
                raise RuntimeError("Installed server failed status check")
            body = json.loads(status.content[0].text)
            if initialized.serverInfo.version != body["producer"]["version"]:
                raise RuntimeError("MCP handshake version differs from producer identity")
            catalog = json.dumps([t.model_dump(mode="json") for t in tools.tools], sort_keys=True)
            return {
                "server": initialized.serverInfo.model_dump(),
                "producer": body["producer"],
                "tool_catalog_sha256": hashlib.sha256(catalog.encode()).hexdigest(),
                "tools": sorted(required),
                "backend_ready": body["capabilities"]["image"]["ready"],
            }
    finally:
        from agentic_mcp_foundation.transport import rpc

        for endpoint in runtime.rglob("endpoint.json"):
            rpc(json.loads(endpoint.read_text()), "retire")


def install(args):
    home = args.home.expanduser().resolve()
    source = Path(__file__).resolve().parents[2]
    if not (source / "uv.lock").is_file():
        raise ValueError("Install/update from a source distribution containing uv.lock")
    roots = tuple(Path(p).expanduser().resolve(strict=True) for p in args.root)
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError("--root must identify existing media directories")
    config = recommend(hardware())
    config["omni_enabled"] = args.omni and config["omni_enabled"]
    release = importlib.metadata.version("local-vision-mcp") + "-" + uuid.uuid4().hex[:12]
    payload = home / "payloads" / release
    with file_lock(home / "install.lock"):
        recover(home)
        previous = (home / "active.json").read_bytes() if (home / "active.json").exists() else None
        payload.mkdir(parents=True)
        try:
            with TemporaryDirectory(prefix="local-vision-build-") as temporary:
                temporary = Path(temporary)
                subprocess.run(
                    ["uv", "build", "--wheel", "--out-dir", str(temporary)],
                    cwd=source,
                    check=True,
                    stdout=sys.stderr,
                )
                command = [
                    "uv",
                    "export",
                    "--frozen",
                    "--no-dev",
                    "--no-emit-project",
                    "--no-emit-package",
                    "agentic-mcp-foundation",
                    "--output-file",
                    str(temporary / "requirements.txt"),
                ]
                if args.omni:
                    command.extend(["--extra", "omni"])
                subprocess.run(command, cwd=source, check=True, stdout=subprocess.DEVNULL)
                subprocess.run(
                    ["uv", "venv", "--python", sys.executable, str(payload / "venv")],
                    check=True,
                    stdout=sys.stderr,
                )
                python = payload / "venv/bin/python"
                subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--require-hashes",
                        "-r",
                        str(temporary / "requirements.txt"),
                    ],
                    check=True,
                    stdout=sys.stderr,
                )
                # VCS requirements cannot participate in --require-hashes. Install
                # the foundation separately from its immutable full commit pin;
                # all registry dependencies above remain hash-locked.
                foundation = next(
                    requirement
                    for requirement in importlib.metadata.requires("local-vision-mcp")
                    if requirement.startswith("agentic-mcp-foundation")
                )
                import re

                if not re.fullmatch(
                    r"agentic-mcp-foundation\s*@\s*git\+https://github\.com/"
                    r"GeorgeQLe/agentic-mcp-foundation\.git@[0-9a-f]{40}",
                    foundation,
                ):
                    raise RuntimeError("Foundation dependency must pin its upstream Git commit")
                subprocess.run(
                    ["uv", "pip", "install", "--python", str(python), "--no-deps", foundation],
                    check=True,
                    stdout=sys.stderr,
                )
                subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-deps",
                        *[str(p) for p in sorted(temporary.glob("*.whl"))],
                    ],
                    check=True,
                    stdout=sys.stderr,
                )
            with TemporaryDirectory(prefix="local-vision-installed-smoke-") as temporary:
                temporary = Path(temporary)
                isolated = {**config, "ollama_url": "http://127.0.0.1:9", "omni_enabled": False}
                atomic_json(temporary / "config.json", isolated)
                probe = asyncio.run(
                    smoke(python, temporary / "config.json", temporary, temporary / "runtime")
                )
            if probe["producer"]["version"] != importlib.metadata.version("local-vision-mcp"):
                raise RuntimeError("Installed producer version does not match source version")
            from .coordinator import build_identity

            if probe["producer"] != build_identity():
                raise RuntimeError("Installed producer build does not match source build")
            files = {}
            for path in payload.rglob("*"):
                # Verify all installed executable/data files, not cache-generated bytecode.
                if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                    digest = hashlib.sha256()
                    with path.open("rb") as content:
                        while block := content.read(1024 * 1024):
                            digest.update(block)
                    files[str(path.relative_to(payload))] = digest.hexdigest()
            manifest = {
                "protocol": 1,
                "contract_major": 1,
                "release": release,
                "python": "venv/bin/python",
                "module": "local_vision_mcp",
                "cli_module": "local_vision_mcp.cli",
                "files": files,
                "acceptance": probe,
            }
            atomic_json(home / "prepared.json", manifest)
            # Validate before publishing. Old payloads are retained for live clients and recovery.
            launcher.validate(home, "prepared.json")
            if previous is not None:
                launcher.validate(home)
            plugin_version = json.loads(
                (source / "plugins/local-vision/.codex-plugin/plugin.json").read_text()
            )["version"]
            if plugin_version != probe["producer"]["version"]:
                raise RuntimeError("Plugin version does not match producer version")

            def verify_plugin():
                declaration = json.loads((home / "plugin/.mcp.json").read_text())["mcpServers"][
                    "local-vision"
                ]
                with TemporaryDirectory(prefix="local-vision-plugin-smoke-") as temporary:
                    temporary = Path(temporary)
                    atomic_json(
                        temporary / "config.json",
                        {
                            **config,
                            "ollama_url": "http://127.0.0.1:9",
                            "omni_enabled": False,
                        },
                    )
                    observed = asyncio.run(
                        smoke(
                            python,
                            temporary / "config.json",
                            temporary,
                            temporary / "runtime",
                            declaration,
                            home,
                        )
                    )
                    if (
                        observed["producer"] != probe["producer"]
                        or observed["tool_catalog_sha256"] != probe["tool_catalog_sha256"]
                    ):
                        raise RuntimeError("Installed plugin identity/catalog mismatch")

            publish(
                home,
                {
                    "control/launcher.py": Path(launcher.__file__).read_bytes(),
                    "settings.json": json.dumps(
                        {"config": config, "roots": [str(p) for p in roots]}
                    ).encode(),
                    "last-healthy.json": previous or json.dumps(manifest).encode(),
                    "active.json": json.dumps(manifest).encode(),
                },
                {"plugin": source / "plugins/local-vision"},
                verify=verify_plugin,
            )
            return {
                "installed": True,
                "home": str(home),
                "plugin": str(home / "plugin"),
                "release": release,
                "acceptance": probe,
                "next": "Enable this local plugin in your MCP host; refresh an existing connection",
            }
        except Exception:
            # Unpublished payloads are safe to remove; active payloads remain available.
            if (
                not (home / "active.json").exists()
                or json.loads((home / "active.json").read_text()).get("release") != release
            ):
                shutil.rmtree(payload)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=home_directory())
    commands = parser.add_subparsers(dest="command", required=True)
    installation = commands.add_parser("install")
    installation.add_argument("--root", action="append", required=True)
    installation.add_argument("--omni", action="store_true")
    for name in ("doctor", "ensure", "update", "verify", "stop"):
        command = commands.add_parser(name)
        if name == "stop":
            command.add_argument("--backend", action="store_true")
        if name in ("ensure", "update"):
            command.add_argument("--with-omni", action="store_true")
        if name in ("ensure", "update", "verify"):
            command.add_argument("--key", default=None)
    job = commands.add_parser("job")
    job.add_argument("id")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("id")
    args = parser.parse_args()
    try:
        if args.command == "install":
            result = install(args)
        else:
            config, roots = settings(args.home)
            # An explicitly selected installation uses its corresponding shared runtime.
            os.environ.setdefault("LOCAL_VISION_RUNTIME_DIR", str(args.home / "runtime"))
            remote = RemoteService(config, roots)
            if args.command == "doctor":
                result = remote.status()
            elif args.command in ("ensure", "update"):
                result = remote.manage(args.command, args.with_omni, args.key)
            elif args.command == "verify":
                result = remote.call(
                    "submit",
                    {
                        "operation": "verify",
                        "request": {},
                        "idempotency_key": args.key or uuid.uuid4().hex,
                    },
                )
            elif args.command == "job":
                result = remote.get_job(args.id)
            elif args.command == "cancel":
                result = remote.cancel_job(args.id)
            else:
                result = remote.call("retire", {"stop_backend": args.backend})
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
