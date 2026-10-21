# Same-agent local-vision benchmark

## Results — October 19, 2026

The completed comparison **does not support equal effectiveness or lower hosted token
usage** for the current MCP route on these tasks.

| Host | Native correct | MCP correct | Native host tokens | MCP host tokens | MCP/native tokens | Mean seconds native / MCP |
|---|---:|---:|---:|---:|---:|---:|
| Codex, GPT 6.1 Sol (requested) | 24/24 (100%) | 18/24 (75%) | 314,141 | 999,597 | 3.18× | 4.48 / 18.48 |
| Claude Code, Opus 5.5 | 24/24 (100%) | 21/24 (87.5%) | 95,066 | 326,017 | 3.43× | 4.70 / 20.88 |

All 96 requests completed without runtime errors. All hosted token usage was available.
The 48 MCP sessions made 52 verified local analysis calls, including four follow-up calls
from Claude. No local reports were truncated. The quality gate failed, as expected from
the measured correctness; this is a completed experiment, not an interrupted run.

Identifier failures omitted the `build-` prefix, including one cropped identifier for
both hosts. Codex also supplied build identifiers as commit SHAs in two absence-check
cases. Native inspection had no failures in this run. Claude sometimes recovered from
an incorrect local observation by asking a second question; those calls and their usage
are included. No rubric or prompt was tuned after observing these failures.

Input totals include cached tokens. The JSON summary separates cache reads and writes:
Codex uncached input was 90,384 native versus 129,956 with MCP; Claude uncached input was
79,066 versus 78,635. Output was 269 versus 4,489 for Codex, and 424 versus 6,035 for
Claude. Consequently, the total-token multipliers must not be interpreted as monetary
cost multipliers. Local inference counts remain separate from hosted usage.

The additional tool context and agent steps are plausible contributors to the overhead;
this experiment does not isolate their individual contributions. Compact tool catalogs,
batching, clearer identifier extraction, and uncertainty-based native escalation are
follow-up hypotheses, not demonstrated fixes. Any revised route should be evaluated on
new held-out tasks rather than qualified against this now-observed corpus.

Local artifacts (generated media and raw results are Git-ignored):

- `benchmarks/results/agent-efficiency-20261019-isolated/efficiency.md` and `efficiency.json`:
  paired accuracy, usage, regressions, and latency.
- `report.json`, `report.md`, and `requests.jsonl`: per-request answers, exact rubrics,
  protocol evidence, scores, usage, requested/resolved model metadata, and CLI versions.
- `ollama-models.json`, `ollama-version.json`, and `provenance.json`: model digests,
  runtime version, GPU information, and source/dataset fingerprints.

Environment: Codex CLI 0.163.0, Claude Code 2.1.304, Python 3.12.12, Ollama 0.30.6,
Qwen 3.5 4B Q4_K_M, NVIDIA RTX 2080 SUPER (8 GiB), driver 617.42. Codex's event stream
did not report a resolved model; its table entry is the explicitly requested identifier.
The full isolated comparison took approximately 19.4 minutes.

Validation of the reusable harness: 57 regression tests passed; Ruff lint/format and
Git whitespace checks passed. Artifact audit confirmed 96 unique requests, complete
pairs, unchanged source media hashes, verified MCP use, and complete host usage.

## Experiment

The experiment compares the same authenticated CLI/model/reasoning setting with native
image attachments and with local-vision MCP. It tests whether routing perception locally
preserves task correctness and reduces total reported hosted tokens. Neither outcome
is assumed in advance.

## Frozen workload and configuration

- Dataset: `benchmarks/generated/agent-efficiency-20261019/manifest.json`.
- Generator seed: `20261019`; three variants, images only.
- 24 questions across OCR, tables, small text, cropped OCR, absence, presence controls,
  embedded instructions, and counting. Related questions share scenes; these are not
  24 independent real-world samples.
- Hosts: Codex CLI requesting `gpt-6.1-sol` and Claude Code requesting `claude-opus-5-5`,
  both at medium reasoning effort.
- Local perception: installed Ollama `qwen3.5:4b`; no weight downloads or model fallback.
- Matrix: `benchmarks/compare.agent-efficiency.json`; 96 sessions, one per arm/question.
- Execution: shuffled with scheduler seed 42, sequential dispatch, 120-second per-session
  deadline. Scoring uses the deterministic manifest rubrics; errors fail the task.
- Each native arm receives prepared pixels with the same crop/resize applied locally.
  MCP arms receive paths and requested tool arguments, with no image attachment.
- Only the requested analysis tool is preapproved. CLI shell tools and web access are
  disabled; Codex's local `view_image` tool is explicitly disabled. Agent-visible inputs
  exclude rubrics and expected answers. Actual MCP calls and results are recorded.
- A fresh temporary CLI session and in-process stdio MCP server are created per request.
  Ollama model residency and provider prompt caches can persist between requests.
  This tests the MCP tool surface, not shared-coordinator throughput or autonomous adoption.

## Reproduce

```bash
.venv/bin/local-vision-bench fixtures \
  --output benchmarks/generated/agent-efficiency-20261019 \
  --variants 3 --seed 20261019 --images-only
.venv/bin/local-vision-bench compare \
  --manifest benchmarks/generated/agent-efficiency-20261019/manifest.json \
  --config benchmarks/compare.agent-efficiency.json \
  --output benchmarks/results/agent-efficiency-20261019-isolated \
  --timeout 120 --no-tui
.venv/bin/python -m local_vision_mcp.benchmark.efficiency \
  benchmarks/results/agent-efficiency-20261019-isolated/report.json
```

Use fresh output directories for another run. A quality-gate exit of 1 still saves the
completed measurements. Hosted sessions consume the CLI accounts' usage.

## Interpretation

The primary comparison is exact task correctness and **total hosted input plus output
tokens**, including cached input and tool-related agent steps. Cached and uncached input
remain available separately in the JSON summary. Local-model token counts use a different
tokenizer and are reported separately; they are not added to hosted usage.

Token counts are actual CLI-reported usage. Monetary estimates, where available, are
API-equivalent estimates rather than subscription invoices. Latency includes CLI startup,
MCP startup, inference, and the final agent answer. Cold/warm caches are not separately
controlled, so monetary and latency findings are exploratory.

These synthetic one-question image tasks can expose a failure of the quality/efficiency
claim on this workload. They cannot establish general parity, video/audio parity, or
coding-agent productivity. A broader claim requires representative held-out real tasks,
a predefined acceptable quality loss, and sufficient independent observations.

Exploratory pilot and interrupted pre-isolation runs remain under `benchmarks/results/`.
Only the completed `agent-efficiency-20261019-isolated` run belongs in the final comparison.
