# Paused benchmark: resume here

Paused at the user's request on 2026-10-20, 04:23 UTC. Implementation and local artifacts are saved in this working tree; nothing was committed, pushed, or installed into global agents.

## Checkpoint

- Corrected screening run: **251/512 completed**, status `Interrupted`, with 251 started and 251 completed receipts; no outstanding requests. **261 screening jobs remain.**
- Shadow comparison: **not started**, 64 workflows remain.
- Temporary Ollama server on port 11435 and telemetry observer are stopped. The preexisting shared server on 11434 remains available (version 0.30.6).
- Frozen source, config, and manifest hashes still match `provenance-screening-v2.json`.
- Final verification: **90 tests passed in 13.31 seconds**; Ruff lint and formatting checks and `git diff --check` passed.
- Native vision remains primary. No local task has been qualified or promoted.

All result paths below are relative to `benchmarks/results/structured-output-20261020/` (gitignored, retained locally):

- `screening-v2/report.json`, `report.md`, `requests.jsonl`: preserved raw results.
- `pending-jobs.json`: exact missing `(model_name, case_id, repeat)` jobs, plus manifest fingerprint.
- `pause.json`: checkpoint and cleanup verification.
- `provenance-screening-v2.json`: frozen file hashes.
- `environment.json`, `owned-runtime.json`, `runtime-log-evidence.json`, `telemetry.jsonl`: runtime evidence.
- `video-regression-v2/`: completed 27-case development regression, 18 strict passes; interpretation limits below.

## Resume procedure

1. Read [the protocol and limitations](structured-output-benchmark.md), `benchmarks/structured-output-v2.protocol.json`, the frozen provenance, and `pending-jobs.json`. Recheck file hashes before inference.
2. The harness currently has **no automatic resume**. Implement a pending-job continuation or create per-arm manifests for only the missing jobs, writing to **new output directories**. Preserve the existing report. Combine disjoint rows only after verifying identical frozen media, questions, schemas, models, scoring, and repeat IDs; record the restart and new runtime metadata. Do not blindly rerun `/tmp/run-structured-screening.py`, which expects empty output directories and would restart screening.
3. Restart only an owned isolated Ollama runtime on port 11435 using the command below; leave shared 11434 untouched. Restart telemetry with a new log/marker, since `telemetry.stop` now exists. Archive or replace the old stop marker deliberately.
4. Finish the 261 missing screening jobs using `benchmarks/compare.structured-screening.json`, seed 42, parallelism 1, timeout 300, and the existing corrected manifest `benchmarks/generated/structured-screening-v2-20261020/manifest.json` (fixture seed 2026102003).
5. Run the frozen 16-scene, four-arm shadow comparison using `benchmarks/compare.structured-shadow.json` and `benchmarks/generated/structured-shadow-v2-20261020/manifest.json`. Native CLIs answer independently; consensus applies only to explicitly tested local-model experiments, not native baselines. Shadow always calls native and never accepts a local answer as final.
6. Generate qualification reports for completed phases with `benchmark/qualification_report.py`; audit literal OCR strings separately because the existing field scorer normalizes case and whitespace. Report strict formatting failures separately from schema-valid visual mistakes. Document actual hosted tokens, local tokens separately, latency, coverage, failures, and routing overhead. The current shadow harness uses direct Service calls and excludes MCP transport overhead.
7. Recheck tests if code changes. Keep native default unless independent qualification and measured end-to-end benefit meet the policy; these related synthetic scenes alone cannot qualify routing.

Owned runtime command (model store exists; do not modify shared runtime):

```bash
env OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
  OLLAMA_NOPRUNE=true OLLAMA_NO_CLOUD=true \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=30m ollama serve
```

Frozen local model: `qwen3.5:4b`, Q4_K_M, digest `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`, context 16384, temperature 0. Runtime used CPU vision projection and all 33 language layers on GPU. Native baselines: Codex `gpt-6.1-sol` medium and Claude `claude-opus-5-5` medium.

## Implementation and interpretation

Implemented caller-provided response schemas through image/video/durable APIs, strict independent JSON/schema validation, conservative shadow routing, fixture generation, regression corpus, qualification summaries, and tests. Ollama's format parameter is not a guarantee: observed fenced answers are rejected without repair. Valid schema also does not establish correct visual content; corrected screening includes genuine counting and table-selection errors. Global installed skills and agent dispatch were not changed.

The original `screening/` run (63 completed, 64 started) used an ambiguous blue counting header and is **development-only**. Preserve it and its amendment/recovery artifacts; do not pool it with corrected `screening-v2/`. The initial manual count review was retracted in `counting-ambiguity-amendment.json`. The corrected renderer uses a neutral header and explicit counting scope, with a new seed.

The completed 27-case video development rerun had 18 strict passes and no execution errors. Its comparison to the older 17/27 pilot is not causal evidence of improvement because strict fence handling changed. Video remains native-default. Fresh screening has 128 synthetic scenes across four candidate tasks, drawn from related renderer families; shadow has 16 preselected scenes. Neither provides independent real-world qualification or proves token savings.

All raw artifacts and existing changes, including preexisting untracked `.github/`, were preserved. Resume from this checkpoint; do not regenerate or overwrite frozen fixtures/results.
