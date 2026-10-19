# Agent-operated local-vision

## Bootstrap and install

Python 3.10+ and `uv` are required. The managed supervisor supports Linux/WSL2 and has
macOS 14+ compatibility for Apple Silicon and Intel. MacBook hardware inference validation remains
pending; see [the MacBook guide](macos-validation.md).
FFmpeg/ffprobe and GPU drivers are host prerequisites. The installer does not require Ollama
to exist and does not download models. It installs pinned Python dependencies, builds the local wheel and the foundation wheel from its pinned Git commit, verifies the installed MCP command and publishes a self-contained plugin/runtime.

From this checkout, the host agent runs:

```bash
python3 scripts/install.py install --root /absolute/media/root
# For audio dependencies, add --omni. Repeat --root for multiple existing directories.
```

The default installation home is `~/.local/share/local-vision-mcp` (respects `XDG_DATA_HOME`);
use `python3 scripts/install.py --home /path/to/installation install --root /media` to override.
`LOCAL_VISION_HOME` selects that home in the plugin bootstrap. The installed plugin is `HOME/plugin`.
Enable that local plugin through your host's supported plugin installation flow, then refresh its
MCP connection. This installer prepares the package; it does not edit host marketplace or user
configuration. Non-plugin clients can launch `python3 HOME/control/launcher.py` as a stdio server.
The plugin and launch command have no source-checkout or host-cache dependency.

Python payload updates use the same guarded install command. The previous verified payload remains
available. A source checkout with `uv.lock` is needed to build a new version, but never to run an
installed version. The stable bootstrap checks payload file hashes before execution. A new MCP
generation reconnects to its matching coordinator; compatible old generations retire only when idle.

## Readiness and provisioning

The bundled `use-local-vision` skill explains agent routing and recovery. `local_media_status`
returns producer identity, hardware, current pool state, recent durable jobs and per-modality
`capabilities`. A connected server may correctly report that vision/audio is not ready.

```json
{"action":"ensure","idempotency_key":"setup-vision-1"}
```

Call `local_media_manage` with that request and poll `get_media_job` using its returned `id`.
`ensure` reuses a running Ollama or an installed binary, starts one if necessary, installs latest
Ollama when absent, and pulls the configured vision model. `with_omni:true` also downloads audio
weights; the installed payload must include the Omni extra and its config must enable Omni.
Use `action:"update"` for an explicit latest-Ollama upgrade. The server never executes arbitrary
shell commands or caller-supplied installer URLs.

Configuration and allowed roots live in `HOME/settings.json`, separately from plugin caches.
`LOCAL_VISION_CONFIG`, `LOCAL_VISION_ROOTS` and `LOCAL_VISION_RUNTIME_DIR` remain overrides.
Update settings deliberately and reconnect; a changed configuration/root set selects another
coordinator scope and receipt ledger. Keep the old settings to recover jobs from the previous scope.
The managed Ollama directory is `HOME/runtime`; Python payloads live in `HOME/payloads`.

## Durable analysis

```json
{
  "path":"/absolute/media/root/screen.png",
  "question":"Quote the visible error code.",
  "kind":"image",
  "max_tokens":128,
  "idempotency_key":"inspect-screen-1"
}
```

Call `submit_media_job`, retain its ID/key and poll `get_media_job`. The receipt survives client
disconnection, and successful results bind to the media SHA256 and producer build. The worker
analyzes a temporary snapshot matching that hash. Reuse the same key only for identical intent;
changed content requires a fresh key. `cancel_media_job` supports queued cancellation and cooperative
active cancellation. Jobs interrupted by coordinator loss remain inspectable and are not replayed.

The original `analyze_*` tools remain synchronous conveniences backed by the same durable execution
path. Explicit job submission is preferable when the caller needs to retain an ID across a lost
tool response or disconnect. Receipts retain questions, paths and text results on local disk; no
automatic retention/deletion policy is implemented. Raw snapshots and extracted audio are temporary.

## Doctor, recovery and stop

When MCP cannot start, the host agent can diagnose the stable payload without importing MCP:

```bash
python3 HOME/control/launcher.py --doctor
python3 HOME/control/launcher.py --rollback
```

Rollback changes the server-payload pointer to a verified prior payload. It does not terminate active
jobs; generation replacement still requires an idle coordinator. A missing/corrupt rollback target
fails with repair instructions. Run the repository installer if no valid installed payload exists.

The installed host-agent CLI is also available independently of MCP:

```bash
python3 HOME/control/launcher.py --cli doctor
python3 HOME/control/launcher.py --cli ensure --key repair-1
python3 HOME/control/launcher.py --cli job JOB_ID
python3 HOME/control/launcher.py --cli verify --key probe-1
python3 HOME/control/launcher.py --cli stop
python3 HOME/control/launcher.py --cli stop --backend
```

`doctor` starts/attaches to the lightweight coordinator but never loads models or downloads weights.
The standalone `--doctor` instead checks only the installed payload, so it remains usable when the
application cannot start. `stop` retires an idle coordinator; Ollama remains available. `stop --backend`
also stops a verified owned Ollama when no other scope is using it. Unrelated services are never killed.
Owned Ollama processes persist through MCP/coordinator reconnections and have Linux/macOS identity records
for recovery after supervisor loss. Active model calls have no hard cancellation guarantee.

## Qualification

`verify_local_media` submits a real, model-backed color probe with an idempotency key; poll its job.
A passing probe establishes basic vision operation. Readiness/probe success do not claim broad
accuracy or speed. Use `local-vision-bench` for workload-specific qualification; current diagnostic
benchmarks have not met the documented 90% / 30-second target.

For an isolated installed live acceptance run:

```bash
python3 scripts/install.py --home /tmp/vision-acceptance install --root /absolute/media/root
python3 scripts/live_acceptance.py --home /tmp/vision-acceptance \
  --model qwen3.5:2b --output /tmp/vision-acceptance-report.json
```

This explicitly downloads Ollama/models and runs inference. Optionally supply `--reuse-models
/path/to/ollama/models` to reuse readable cached blobs. A dedicated endpoint/model cache protects
other Ollama services; the test stops its backend afterward. Installation acceptance in CI is offline
with respect to models and verifies startup/identity/catalog separately from backend readiness.

Before repository extraction, implementation acceptance on 2026-10-18 passed all 46 tests, lint/format checks, and the actual
installed plugin launch from outside the checkout. Installed producer build
`de77fa469741535f7611a91f8250f6ad921deb63ee86f70c199f5434f0503726` passed a real
Qwen3.5 2B probe using managed Ollama 0.40.6, answering `Red`. A prior build also passed with
Qwen3.5 4B. These checks establish basic operation; audio and broad workload quality were not
qualified by these probes. The dedicated acceptance backend was stopped afterward.

After extraction, Linux validation passed 38 local-vision tests and 12 foundation tests. The
standalone installer fetched foundation commit `4f28f324205499d0229190c1843a4af9679622ff`
from GitHub, built the consumer wheel, and passed the actual installed plugin launch/catalog smoke
checks outside the checkout. macOS archive selection/layout is covered by simulated tests on Linux;
physical MacBook acceptance remains pending.
