# Production readiness and deferred benchmarks

Benchmark testing is paused at the user's request. Resume explicitly; do not launch hosted-model
runs or long local benchmarks automatically. The CLI/TUI implementation is saved, but the small
smoke tests below do **not** establish perception parity or production readiness.

## Evidence already collected

- [x] Implement `local-vision-bench compare` and simulated `demo`, with a live TUI, plain CI output,
  shared labeled cases, deterministic correctness, latency, token usage and estimated costs.
- [x] Add Ollama/Omni and native Codex/Claude Code adapters, model configuration examples,
  per-category reports, partial-result checkpoints, deadlines and a soft cost dispatch guard.
- [x] Full unrestricted test run: 50 passed. Subsequent cost-normalization edits were checked with
  the 12 comparison tests and Ruff; repeat the full suite before a release.
- [x] Build the source distribution and wheel successfully.
- [x] Exercise the simulated TUI in a real terminal.
- [x] Local Qwen 3.5 4B: two cases passed (about 37.9s and 18.3s).
- [x] Native GPT 6.1 Sol and Claude Fable 5.1: one OCR case each passed (about 6.0s and 3.6s),
  with identical prepared-image hashes and returned token usage. Claude reported an API-equivalent
  estimate of about $0.1097; Codex cost remained unknown without configured rates. These are
  connectivity checks, not a representative model comparison or subscription invoice.

Local artifacts, intentionally git-ignored:

- `benchmarks/results/cli-local-smoke-01/`
- `benchmarks/results/cli-native-smoke-01/`
- `/tmp/local-vision-tui-demo/` (temporary; may disappear)

See [CLI usage](docs/benchmark-cli.md) and [evaluation/routing design](docs/perception-parity.md).

## Resume the benchmark campaign

- [ ] Freeze a representative held-out dataset: initially 300–1,000 unique scenes from actual
  workflows, grouped by original scene so crops, nearby frames and repeated questions cannot
  inflate sample size. Separate development/tuning data from the final test set.
- [ ] Label ground truth independently of model answers. Audit ambiguous examples and score OCR,
  counts, layout/coordinates, charts, absent evidence, uncertainty, prompt injection, changes and
  temporal events. Distinguish malformed JSON from semantic errors during failure analysis.
- [ ] Verify access to all requested baselines: `gpt-6.1-sol`, `gpt-6-astra`, `claude-opus-5-5`,
  and `claude-fable-5-1`. Only Sol/Fable have received the tiny native smoke test so far. Record
  requested/resolved identifiers, CLI versions, reasoning settings and unknown resolution honestly.
- [ ] Record local weight digests/quantization, hardware/VRAM/RAM, Ollama and optional Omni
  versions, preprocessing, context/output limits and prompts. Match image bytes, crops, sampled
  video frames/timestamps, questions and schema across every compared model.
- [ ] Calibrate costs from pilot token usage, including CLI system prompts, image tokens, reasoning,
  cached reads and cache writes. Configure verified rates and conservative forecasts. Set a user
  approved spending allowance before hosted runs. Exercise the soft guard with concurrent requests,
  missing usage, failures, timeouts and overruns; it is not a provider-enforced spending cap.
- [ ] Run the frozen shared workload against local candidates and each native baseline. Separate
  cold start, warm inference and queue time; report strict pass rate, field score, errors, p50/p95,
  throughput, token usage, cost and unknown costs overall and per category. Keep raw API evaluations
  separate from native CLI harness evaluations if API adapters are added.
- [ ] Add scene-paired comparisons and confidence intervals before claiming parity. Choose an
  acceptable quality-loss margin before seeing results; the design proposes two percentage points
  relative to each native baseline, subject to a paired 95% confidence interval. Repeats of the same
  scene must not count as independent evidence. Increase samples where uncertainty is too wide.
- [ ] Evaluate agent task success end to end: the same host agent with native vision, with local
  reports, and eventually with local-first escalation. Score the final action/task outcome and
  information lost in the local report, not only the perception worker's answer.
- [ ] Test audio/music separately: transcription/WER where appropriate, speaker/event attribution,
  timing, instrument labels, tempo/pitch with tolerances, and silence/absent-source cases. Native CLI
  adapters currently attach images and silent video frames, not audio. Do not infer audio parity
  from image scores or compare audio against an unsupported baseline.
- [ ] Implement consensus and native-review routing only after measuring independent error patterns.
  Compare single model, diverse ensemble and disagreement-triggered native review under equal
  budgets; measure false agreement, escalation recall, residual error, latency and cost. Validate
  routing thresholds on development data and evaluate once on held-out scenes.

## Validate the benchmark implementation before trusting its results

- [ ] Repeat full Ruff checks/format verification and pytest after final edits. Exercise the installed
  wheel's CLI, not only the editable checkout. Preserve the manifest/config/environment and reports.
- [ ] Check live rendering in WSL, Linux and macOS terminals: small/large windows, resize, Unicode,
  alternate-screen restoration, non-TTY logs and Ctrl-C during setup, generation and final reporting.
- [ ] Verify exact image/frame equivalence across local and native adapters for crops, EXIF rotation,
  resize, video boundaries and short events. Confirm expected answers never reach provider payloads,
  session files or readable fixture metadata through enabled tools.
- [ ] Test interrupted/timed-out native subprocess trees, CLI crashes, malformed/missing usage,
  auth/rate-limit failures, disk-full/report-write failures and abrupt termination. Confirm partial
  results are recoverable, unknown costs stay unknown and no implicit retries duplicate charges.
- [ ] Extend usage-event fixtures when supported CLIs change their schemas, especially cache-write
  counters and resolved-model reporting. Verify configured prices against provider-reported totals.
- [ ] Confirm cross-model comparisons retain an identical denominator and explicitly account for
  omitted audio cases. Review per-category failures rather than relying on one aggregate score.

## Prove deployment and operational readiness

- [ ] Publish and enable the prepared Linux/macOS CI workflow. It remains local at
  `.github/workflows/ci.yml`: the GitHub credential previously rejected workflow pushes because
  it lacked `workflow` scope. Keep ordinary code pushes independent of that permission issue.
- [ ] Require passing CI on supported Python versions (currently planned: 3.10 and 3.12) for both
  this repository and `agentic-mcp-foundation`; validate the pinned foundation commit is fetchable
  from a clean install and packaged dependencies resolve reproducibly.
- [ ] Test clean WSL2 and Linux installations from published artifacts, then installation/update on
  an actual MacBook. Record macOS version, Apple Silicon/Intel architecture, RAM, acceleration,
  FFmpeg/Ollama availability, permissions and cold/warm behavior. A Linux smoke test is insufficient
  evidence for MacBook compatibility or performance.
- [ ] Exercise install, repeated/idempotent install, update, rollback and repair under interrupted
  downloads, hash mismatch, incompatible binaries, network failure and failed health checks. Verify
  atomic activation, recovery of the previous working release and stable launcher paths.
- [ ] Verify ownership boundaries: external Ollama must survive shutdown/update requests; managed
  processes must be tracked by identity and recovered safely after coordinator/host restarts.
- [ ] Validate Codex and Claude Code configuration discovery and real MCP calls after restart on
  WSL and macOS. Cover stdio and HTTP clients, concurrent sessions, reconnects and stale endpoints.
- [ ] Run sustained load/soak tests at expected client concurrency. Record queue depth, rejection
  behavior, timeouts, cancellation, memory/VRAM growth, GPU contention, process/file-descriptor leaks,
  coordinator loss and recovery. Set explicit warm/cold latency and capacity targets before running.
- [ ] Test path/root restrictions, symlinks and media mutation, malformed/oversized inputs, decoder
  failures, adversarial media instructions and untrusted model output. Require zero unauthorized
  filesystem access or actions. Confirm runtime management cannot execute arbitrary model commands.
- [ ] Validate telemetry and operator recovery: actionable health/readiness status, bounded/redacted
  errors, request IDs, model/version provenance, rollback instructions and cost/usage visibility.
  Define media/report retention and ensure credentials/private media do not leak to logs or artifacts.
- [ ] Promote a versioned release and update the persistent WSL installation after qualification.
  This commit saves the new benchmark in the repository; it does not automatically replace the
  already-installed runtime release. Repeat installed-wheel and host acceptance checks on promotion.

## Release acceptance record

- [ ] Agree on the target workload, quality-loss margin, per-category minimums, latency/capacity
  targets, budget and supported platform matrix before declaring readiness.
- [ ] Attach immutable dataset hashes, model/configuration provenance, CI links, held-out reports,
  load/soak results, clean-install/MacBook evidence, and rollback/recovery results to the release.
- [ ] Require all safety/ownership/recovery checks to pass, no unexplained runtime errors on the
  qualification workload, and the agreed quality/latency gates to hold with sufficient samples.
- [ ] Document remaining limitations and the operator's escalation policy. Do not claim general
  vision/audio parity or production readiness from synthetic fixtures, mocks or a handful of passes.
