"""User-owned Ollama provisioning and lifecycle, callable without a shell or sudo."""

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import zstandard
from agentic_mcp_foundation import LifecycleError, processes
from agentic_mcp_foundation.transport import file_lock


def extract_archive(archive: Path, destination: Path):
    """Extract only ordinary files, directories and links contained within destination."""
    root = destination.resolve()
    with tarfile.open(archive) as source:
        for member in source:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Ollama archive path escapes installation directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                target.chmod(member.mode & 0o755)
            elif member.issym() or member.islnk():
                link = (
                    target.parent / member.linkname if member.issym() else root / member.linkname
                ).resolve()
                if not link.is_relative_to(root):
                    raise ValueError("Ollama archive link escapes installation directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.issym():
                    target.symlink_to(member.linkname)
                else:
                    os.link(link, target)
            else:
                raise ValueError("Unsupported special file in Ollama archive")


class Runtime:
    def __init__(self, config: dict, url: str, directory: Path | None = None):
        self.config, self.url = config, url
        self.directory = directory or Path(
            os.getenv("LOCAL_VISION_RUNTIME_DIR")
            or (
                Path(os.environ["AGENTIC_MCP_HOME"]) / "runtime"
                if os.getenv("AGENTIC_MCP_HOME")
                else Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
                / "local-vision-mcp/runtime"
            )
        )
        self.lock = threading.RLock()
        self.job = None
        self.process = processes.recover(self.directory / "ollama-owner.json")
        self.closed = threading.Event()
        self.progress_hook = None

    @property
    def binary(self):
        # Linux archives use bin/; Darwin's universal archive puts its binary and
        # Metal libraries together at the root. Keep the upstream layout intact.
        for name in ("bin/ollama", "ollama"):
            managed = self.directory / "current" / name
            if managed.is_file():
                return str(managed)
        return shutil.which("ollama")

    def version(self):
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.get(self.url + "/api/version")
                response.raise_for_status()
                data = response.json()
                return data.get("version") if isinstance(data, dict) else None
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    def status(self):
        version = self.version()
        with self.lock:
            return {
                "ollama_url": self.url,
                "ollama_version": version,
                "ollama_reachable": version is not None,
                "ollama_binary": self.binary,
                "managed_directory": str(self.directory),
                "owns_server": self.process is not None and self.process.poll() is None,
                "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
                "job": dict(self.job) if self.job else None,
            }

    def launch(self, action: str, with_omni: bool = False):
        if action not in ("ensure", "update"):
            raise ValueError("action must be ensure or update")
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("Runtime is closed")
            if self.job and self.job["state"] == "running":
                return dict(self.job)
            self.job = {
                "id": uuid.uuid4().hex,
                "action": action,
                "with_omni": with_omni,
                "state": "running",
                "phase": "starting",
                "error": None,
            }
            threading.Thread(target=self._run, args=(action, with_omni), daemon=True).start()
            return dict(self.job)

    def phase(self, message: str):
        if self.closed.is_set():
            raise RuntimeError("Runtime closed during provisioning")
        with self.lock:
            self.job["phase"] = message[:500]
        if self.progress_hook is not None:
            self.progress_hook(message)

    @property
    def usage_lock(self):
        return self.directory / (
            "backend-" + hashlib.sha256(self.url.encode()).hexdigest()[:16] + ".lock"
        )

    def _run(self, action, with_omni):
        try:
            with file_lock(self.usage_lock, blocking=False):
                self._run_locked(action, with_omni)
        except LifecycleError as exc:
            with self.lock:
                self.job.update(state="failed", error=str(exc), failure=exc.report())

    def _run_locked(self, action, with_omni):
        previous = None
        activated = False
        was_running = False
        try:
            # Serialize installs and model provisioning across POSIX MCP processes.
            import fcntl

            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / "provision.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        "Another server is provisioning this runtime; retry later"
                    ) from exc
                running = self.version() is not None
                if self.process is None or self.process.poll() is not None:
                    self.process = processes.recover(self.directory / "ollama-owner.json")
                owns = self.process is not None and self.process.poll() is None
                was_running = owns
                if action == "update" and running and not owns:
                    raise RuntimeError(
                        "Ollama is managed by another process. This server cannot restart it; "
                        "stop that service before requesting a managed update."
                    )
                if action == "update" or (not running and not self.binary):
                    previous = (
                        (self.directory / "current").resolve()
                        if (self.directory / "current").exists()
                        else None
                    )
                    release = self.install(activate=False)
                    self.phase("draining and activating Ollama")
                    if owns:
                        self.stop()
                    activated = True
                    self.activate(release)
                    running = False
                if not running:
                    if owns and not activated:
                        self.stop()
                    self.start()
                self.pull()
                if with_omni:
                    self.download_omni()
                if activated:
                    self.phase("health-checking activated Ollama")
                    self.healthcheck()
                    pending = self.directory / ("healthy-" + uuid.uuid4().hex)
                    pending.symlink_to(
                        (self.directory / "current").readlink(), target_is_directory=True
                    )
                    pending.replace(self.directory / "last-healthy")
                self.phase("ready")
                with self.lock:
                    self.job["state"] = "succeeded"
        except Exception as exc:  # noqa: BLE001 - report failures from the background job to MCP
            rollback = None
            if activated:
                # Rollback is cleanup and must proceed even when cancellation was requested.
                hook, self.progress_hook = self.progress_hook, None
                try:
                    self.stop()
                    if previous is not None:
                        self.activate(previous)
                        if was_running:
                            self.start()
                    else:
                        (self.directory / "current").unlink(missing_ok=True)
                        if was_running:
                            self.start()
                except Exception as recovery:  # noqa: BLE001 - retain both failures
                    rollback = str(recovery)
                finally:
                    self.progress_hook = hook
            with self.lock:
                message = str(exc) + (f"; rollback failed: {rollback}" if rollback else "")
                self.job.update(state="failed", error=message[:2000])
                if isinstance(exc, LifecycleError):
                    self.job["failure"] = exc.report()

    def activate(self, release):
        pending = self.directory / ("current-" + uuid.uuid4().hex)
        pending.symlink_to(release.name, target_is_directory=True)
        pending.replace(self.directory / "current")

    def install(self, activate=True):
        system = platform.system()
        arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
        if system not in ("Linux", "Darwin") or not arch:
            raise RuntimeError("Managed Ollama supports Linux/WSL2 and macOS on x86_64 or arm64")
        if system == "Darwin":
            version = platform.mac_ver()[0]
            if not version or int(version.split(".")[0]) < 14:
                raise RuntimeError("Managed Ollama requires macOS 14 or newer")
        asset = "ollama-darwin.tgz" if system == "Darwin" else f"ollama-linux-{arch}.tar.zst"
        url = (
            "https://github.com/ollama/ollama/releases/latest/download/" + asset
            if system == "Darwin"
            else "https://ollama.com/download/" + asset
        )
        self.phase("downloading latest Ollama")
        release = self.directory / ("release-" + uuid.uuid4().hex)
        release.mkdir()
        try:
            with TemporaryDirectory(dir=self.directory) as tmp:
                archive = Path(tmp) / asset
                with (
                    httpx.Client(follow_redirects=True, timeout=120) as client,
                    client.stream("GET", url) as response,
                    archive.open("wb") as output,
                ):
                    response.raise_for_status()
                    count = 0
                    for chunk in response.iter_bytes(1024 * 1024):
                        count += len(chunk)
                        if count > 8 * 1024**3:
                            raise RuntimeError("Ollama download exceeded 8 GiB")
                        self.phase(f"downloading Ollama: {count // 1024**2} MiB")
                        output.write(chunk)
                self.phase("extracting Ollama")
                unpacked = Path(tmp) / "ollama.tar"
                import gzip

                with archive.open("rb") as compressed, unpacked.open("wb") as output:
                    reader = (
                        gzip.GzipFile(fileobj=compressed)
                        if system == "Darwin"
                        else zstandard.ZstdDecompressor().stream_reader(compressed)
                    )
                    count = 0
                    while chunk := reader.read(1024 * 1024):
                        count += len(chunk)
                        if count > 16 * 1024**3:
                            raise RuntimeError("Ollama archive exceeded 16 GiB unpacked")
                        self.phase("extracting Ollama")
                        output.write(chunk)
                extract_archive(unpacked, release)
            binary = release / ("ollama" if system == "Darwin" else "bin/ollama")
            subprocess.run(
                [str(binary), "--version"],
                check=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            files = {}
            for path in sorted(release.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                    files[str(path.relative_to(release))] = digest.hexdigest()
            (release / "manifest.json").write_text(json.dumps({"source": url, "files": files}))
            if activate:
                self.activate(release)
            return release
        except Exception:
            shutil.rmtree(release)
            raise

    def start(self):
        self.phase("starting Ollama")
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("Runtime is closed")
            if not self.binary:
                raise RuntimeError("Ollama binary is missing")
            managed = self.directory / "current/manifest.json"
            if managed.is_file():
                root = managed.resolve().parent
                for name, expected in json.loads(managed.read_text())["files"].items():
                    path = (root / name).resolve()
                    if not path.is_relative_to(root):
                        raise RuntimeError("Invalid managed Ollama manifest path")
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                    if digest.hexdigest() != expected:
                        raise RuntimeError(f"Managed Ollama integrity check failed: {name}")
            with (self.directory / "ollama.log").open("wb") as log:
                self.process = subprocess.Popen(
                    [self.binary, "serve"],
                    env={**os.environ, "OLLAMA_HOST": self.url},
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                processes.remember(self.directory / "ollama-owner.json", self.process)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.phase("waiting for Ollama")
            if self.process.poll() is not None:
                raise RuntimeError(f"Ollama exited; see {self.directory / 'ollama.log'}")
            if self.version() is not None:
                # Re-record once serving: a wrapper or interpreter that re-execs itself
                # (python.org macOS builds do) changes the executable after launch.
                processes.remember(self.directory / "ollama-owner.json", self.process)
                return
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"Ollama did not become ready; see {self.directory / 'ollama.log'}")

    def pull(self):
        model = self.config["vision_model"]
        self.phase(f"pulling {model}")
        with (
            httpx.Client(timeout=120, trust_env=False) as client,
            client.stream("POST", self.url + "/api/pull", json={"model": model}) as response,
        ):
            response.raise_for_status()
            success = False
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                self.phase(f"{model}: {data.get('status', '')}")
                success = data.get("status") == "success"
            if not success:
                raise RuntimeError("Ollama model pull ended without success")

    def download_omni(self):
        if not self.config.get("omni_enabled"):
            raise RuntimeError("Omni is disabled in the server configuration")
        try:
            import qwen_omni_utils  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Launch the MCP server with uv run --extra omni for audio support"
            ) from exc
        self.phase(f"downloading {self.config['omni_model']}")
        snapshot_download(self.config["omni_model"])

    def healthcheck(self):
        with httpx.Client(timeout=5, trust_env=False) as client:
            version = client.get(self.url + "/api/version")
            version.raise_for_status()
            data = version.json()
            if not isinstance(data, dict) or not isinstance(data.get("version"), str):
                raise RuntimeError("Ollama health check returned no version")  # noqa: TRY004
            response = client.post(
                self.url + "/api/show", json={"model": self.config["vision_model"]}
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Ollama returned a malformed model report")  # noqa: TRY004
            if data.get("remote_model") or data.get("remote_host"):
                raise RuntimeError("Configured model redirects to a cloud backend")
            if "vision" not in data.get("capabilities", []):
                raise RuntimeError("Configured model does not advertise vision support")

    def capabilities(self):
        """Readiness is separate from connection health and measured accuracy qualification."""
        vision_blockers, audio_blockers = [], []
        try:
            self.healthcheck()
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            vision_blockers.append(
                {
                    "code": "VISION_NOT_READY",
                    "message": str(exc)[:500],
                    "repair": {"tool": "local_media_manage", "arguments": {"action": "ensure"}},
                }
            )
        dependencies = all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "qwen_omni_utils", "soundfile")
        )
        if not self.config.get("omni_enabled"):
            audio_blockers.append(
                {"code": "OMNI_DISABLED", "message": "Enable Omni in configuration"}
            )
        if not dependencies:
            audio_blockers.append(
                {
                    "code": "OMNI_EXTRA_MISSING",
                    "message": "Install the runtime with --omni and reconnect",
                }
            )
        cached = False
        if dependencies:
            try:
                from huggingface_hub import snapshot_download

                snapshot = Path(snapshot_download(self.config["omni_model"], local_files_only=True))
                indices = list(snapshot.glob("*.safetensors.index.json"))
                shards = {
                    name
                    for index in indices
                    for name in json.loads(index.read_text()).get("weight_map", {}).values()
                }
                cached = (snapshot / "config.json").is_file() and (
                    all((snapshot / name).is_file() for name in shards)
                    if shards
                    else bool(list(snapshot.glob("*.safetensors")))
                )
            except (OSError, ValueError):
                cached = False
        if not cached:
            audio_blockers.append(
                {
                    "code": "OMNI_WEIGHTS_MISSING",
                    "message": "Audio weights not cached",
                    "repair": {
                        "tool": "local_media_manage",
                        "arguments": {"action": "ensure", "with_omni": True},
                    },
                }
            )
        ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        missing_ffmpeg = {
            "code": "FFMPEG_MISSING",
            "message": "Host agent must install FFmpeg and ffprobe using the OS package manager",
        }
        if not ffmpeg:
            audio_blockers.append(missing_ffmpeg)
        if self.config.get("vision_backend") == "omni":
            vision_blockers = [b for b in audio_blockers if b["code"] != "FFMPEG_MISSING"]
        return {
            "image": {"ready": not vision_blockers, "blockers": vision_blockers},
            "video": {
                "ready": not vision_blockers and ffmpeg,
                "blockers": vision_blockers + ([] if ffmpeg else [missing_ffmpeg]),
            },
            "audio": {"ready": not audio_blockers, "blockers": audio_blockers},
            "qualified": False,
            "qualification": "Run workload benchmarks; readiness does not establish accuracy",
        }

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                (self.directory / "ollama-owner.json").unlink(missing_ok=True)

    def close(self):
        self.closed.set()
        self.stop()
