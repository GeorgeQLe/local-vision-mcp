# Local consensus benchmark

This experiment compares native CLI perception, the existing single-pass MCP route,
and an experimental three-pass local consensus route, using the same hosted model
and effort within each CLI. It does not change the installed MCP server.

**Only the local Qwen passes vote.** One Claude or Codex agent calls the MCP tool;
the tool runs local overview, detail and verification passes, reduces their votes
locally, and returns one combined report. Labels such as "Codex consensus" in raw
progress logs identify the host using this local route. The report separates the
first local vote's accuracy from the individual host's final answer accuracy.

## Frozen protocol

- Hosted arms: Codex `gpt-6.1-sol`, Claude `claude-opus-5-5`, medium effort.
- Local model: installed `qwen3.5:4b`, temperature 0, 8,192 context tokens,
  128 output tokens per inference. No downloads or model selection on test answers.
- Held-out generator seed: `2026101902`, two image-only variants, 16 questions
  per arm, six arms, **96 fresh CLI sessions**. This is an exploratory synthetic
  benchmark: new values and images, with the same layouts as the earlier benchmark.
- Categories: OCR, tables, full-image small text, cropped text, missing-field
  abstention, present-field control, embedded instructions, and counting.
- One repeat, shuffled dispatch seed 42, one CLI session at a time, 300-second
  session deadline. Failures count against accuracy. Record unknown usage explicitly.
- Native arms receive pixels; MCP arms receive paths and must record a successful
  perception tool call. Rubrics stay outside provider inputs. The MCP tool catalog,
  hosted instructions and permissions are identical between single and consensus.
- Consensus: overview and enlarged detail views start concurrently (cap two),
  then a verifier checks their proposed JSON answers against the enlarged image
  on the detail replica. The verifier retains the detail prompt prefix; smoke
  logs nevertheless showed recurrent-model cache invalidation and repeated image
  encoding, so reuse is not assumed in the measured results.
  The enlarged view is at most 2× and 2,048 pixels on its longest edge. A local
  reducer requires two exact JSON matches. Invalid or truncated passes cannot vote;
  no majority returns an unresolved observation. The host gets one report per call,
  containing only the reduced answer, vote status, coverage and aggregate metrics.
- The verifier sees candidate answers, so its vote is correlated. Record wrong
  unanimous answers against hidden ground truth; agreement is not a confidence score.
- Record exact task accuracy, hosted input/output/cache tokens, local inference
  calls and tokens, end-to-end latency, local request timing and overlap. Different
  tokenizers are not combined. No monetary or energy savings claim is inferred.
- Private local audits include all passes, errors and timings, and are collected
  after the CLI finishes. They are never included in the MCP response.
- Local replicas are warmed before measurement. Both replicas remain resident
  throughout all arms; hosted sessions stay fresh. Report this memory footprint.
  Shuffled arms reuse the same questions and local runtimes, so local cache reuse
  is possible. These timings describe this warm comparison, not cold starts.

The preliminary smoke run found that Ollama 0.30.6 logs
`model architecture does not currently support parallel requests` for `qwen35`
and starts its runner with `-np 1` despite `OLLAMA_NUM_PARALLEL=2`.
Therefore real fan-out uses two temporary serving replicas on ports 11435 and
11436, sharing the installed read-only model store. Both use the same model and
context. The shared runtime on port 11434 is untouched. On the 8 GiB RTX 2080
SUPER, both replicas report GPU-resident language-model weights while Ollama
selects CPU processing for the vision projector. Concurrent requests and resident
replicas do not prove simultaneous GPU kernel execution.

This tests the combined effect of stronger local prompts, enlarged views,
verification and voting. An improved result cannot be attributed to voting alone.
Sixteen related synthetic questions cannot establish general effectiveness parity.

## Reproduce

Start two dedicated Ollama servers with the installed model store and ports
matching `benchmarks/compare.consensus.json`. Use `OLLAMA_NUM_PARALLEL=1` for each
replica (the first smoke server requested 2, but the runtime forced it to 1).
Set `OLLAMA_MODELS` to your existing model directory; do not run a new server on
the shared runtime's port. Warm each server with one image request, and keep both
loaded during the comparison. Stop only these dedicated processes afterward.

```sh
.venv/bin/local-vision-bench fixtures \
  --output benchmarks/generated/consensus-heldout-20261019 \
  --seed 2026101902 --variants 2 --images-only
.venv/bin/local-vision-bench compare \
  --manifest benchmarks/generated/consensus-heldout-20261019/manifest.json \
  --config benchmarks/compare.consensus.json \
  --output benchmarks/results/consensus-heldout-20261019 --no-tui
.venv/bin/python -m local_vision_mcp.benchmark.consensus_report \
  benchmarks/results/consensus-heldout-20261019/report.json
```

An exit status of 1 can mean the completed experiment missed the accuracy gate;
inspect `report.json` status and errors before interpreting it as a harness failure.
Raw artifacts are gitignored. The generated `consensus.md`, `consensus.json` and
`efficiency.json` preserve route-specific measurements.

## Results

The October 19, 2026 run completed **96/96 sessions in 76.20 minutes**, with no
provider execution errors and known hosted usage for every session. Each row below
contains the same 16 questions. Claude and Codex each remain individual agents;
only the local Qwen passes vote.

| Individual host | Perception route | Final answers correct | First local vote correct | Hosted tokens | Mean seconds |
|---|---|---:|---:|---:|---:|
| Codex | Native | 16/16 | — | 208,553 | 4.07 |
| Codex | MCP: single local pass | 12/16 | — | 709,993 | 21.84 |
| Codex | MCP: local model consensus | 12/16 | 12/16 | 2,791,901 | 117.15 |
| Claude | Native | 16/16 | — | 63,305 | 2.84 |
| Claude | MCP: single local pass | 14/16 | — | 227,037 | 19.48 |
| Claude | MCP: local model consensus | 13/16 | 12/16 | 205,692 | 120.33 |

**This configuration did not achieve native effectiveness or hosted-token savings
against native perception.** Local consensus used 13.39× the native hosted tokens
with Codex and 3.25× with Claude, with mean latency 28.81× and 42.31× respectively.
Against single-pass MCP, consensus left Codex accuracy unchanged and reduced Claude
accuracy by one question. Claude used 9.4% fewer hosted tokens than its single-pass
MCP route, while Codex used 3.93× as many. Hosted totals include cached input and
are not billing estimates or directly comparable across hosts.

The local audits explain several failures:

- Both full-image build identifiers lost their required `build-` prefix in
  unanimous local votes. These are two unique failing questions, reproduced under
  each host, rather than four independent scenes. The cropped versions passed.
- Local consensus fixed `crop_ocr-011` relative to single-pass MCP under both hosts.
  It regressed on `abstention-012` with Codex, and on `small_text-002` and
  `abstention-004` with Claude. Net accuracy hides those paired changes.
- For absent commit SHAs, two wrong local votes could overrule a verifier that
  correctly returned null. Claude recovered `abstention-012` after a second MCP
  call; its first local vote remains a failure. That session took 270.54 seconds
  and six local inference passes. Both visible-SHA controls passed with consensus.

Codex's consensus arm made 16 MCP calls / 48 local passes, consuming 98,462 local
input and 732 output tokens. Claude's made 17 MCP calls / 51 local passes,
consuming 105,795 input and 1,015 output tokens. One invalid local pass did not vote;
one follow-up call was unresolved. The single-pass arms used 17 and 20 local
passes, respectively. Local and hosted tokenizers remain separate.

All 454 telemetry samples reported both replicas loaded, with total GPU memory
between 7,576 and 7,738 MiB on the 8,192 MiB RTX 2080 SUPER. Language weights were
GPU resident, with CPU vision encoding. The first two local requests overlapped
in the audits; this does not establish simultaneous GPU kernel execution. Ollama
was 0.30.6, Codex CLI 0.163.0 and Claude Code 2.1.304. The temporary servers were
stopped after measurement; the shared server remained available.

Artifacts are in `benchmarks/results/consensus-heldout-20261019/`: `report.json`
contains session-level results and private local audits; `consensus.json` and
`consensus.md` separate first local vote from host accuracy; `efficiency.json`
contains paired usage; `telemetry.jsonl`, `runtime.json`, `provenance-before.json`
and `provenance-after.json` preserve runtime and integrity evidence. Image hashes,
configuration, scoring and inference code matched the pre-run freeze. During the
run, reporting alone changed to clarify local-only consensus and expose first-vote
accuracy following the user's clarification.

The implementation passed all 62 tests, Ruff checks and formatting checks. This
small synthetic run supports rejecting this configuration as an efficiency win;
it does not establish how other local models, crops or real workloads would fare.
