# Live perception benchmarks

`local-vision-bench compare` runs the same labeled cases through local Ollama/Omni models and
native Codex/Claude Code CLI sessions. Its TUI shows completion progress, elapsed time, ETA,
running cases, per-model strict pass rate, field score, error count, latency, and estimated cost.
The existing `fixtures` and `run` commands remain available for local load/worker matrices.

## Try the dashboard

From the repository in WSL, Linux or macOS:

```bash
uv sync --extra bench --extra dev
uv run local-vision-bench demo --output benchmarks/results/demo-01 --tui
```

The demo uses simulated results, never starts models or agent CLIs, and never incurs provider
charges. Use a new result directory for each invocation. The dashboard uses the alternate screen;
the report location remains visible when it exits. Terminals at least 120 columns wide work best.

## Run installed local models

```bash
uv run local-vision-bench fixtures --output benchmarks/generated/compare-01 --variants 3 --images-only
uv run local-vision-bench compare \
  --manifest benchmarks/generated/compare-01/manifest.json \
  --config benchmarks/compare.local.json \
  --output benchmarks/results/local-01 --tui
```

Ollama must be running with the selected model installed. This command never downloads weights,
installs software or stops the existing runtime. For Omni, install the `omni` extra and use a model
entry with `"backend": "omni"`, `"model": "Qwen/Qwen2.5-Omni-3B"`, and `"device": "cpu"` or `"cuda"`.
GPU/device selection affects both speed and memory. Avoid benchmarking alongside active MCP jobs.

## Compare native CLI vision

Install and log in to `codex` and `claude` first. The runner uses those CLIs' existing authentication;
it does not require a separate API key. Edit the example to retain only models your account can use:

```bash
uv run local-vision-bench compare \
  --manifest benchmarks/generated/compare-01/manifest.json \
  --config benchmarks/compare.native.json \
  --output benchmarks/results/native-01 --limit 4 --tui
```

Native entries send the selected media to the respective provider and consume the CLI account's
usage. These are CLI harness comparisons: system prompts, provider-side image processing and
reasoning/output limits can differ from raw APIs. A requested answer length is included in the
native prompt; only the local Service enforces the manifest's output token cap. Codex records the
requested model and leaves resolved model unknown when its event stream does not report it.
Claude records model identifiers from `modelUsage` when available. CLI version metadata is saved.
No automatic model fallback is configured.

Images use the same local crop/resize helpers as MCP. Videos use identical uniformly sampled
PNG frames and timestamps, not native video upload. Native requests record prepared-image hashes.
Codex receives image attachments; Claude receives base64 image blocks through streaming JSON
stdin. No image-reading tool or local-vision MCP call is needed. Each case gets a fresh temporary
working directory and session. User MCP servers, skills and hooks are disabled for these calls;
Claude tools are disabled, and Codex runs read-only with shell/web search/multi-agent disabled.
This is harness isolation for comparison, not a security boundary against a hostile CLI binary.
Rubrics and expected answers remain in the scoring process and reports, outside provider payloads.

The runner selects the shared modality intersection so all models have the same denominator.
Mixed native/local comparisons currently exclude audio and videos requesting audio. An explicit
`--kind audio` requires every model to use Omni. The report lists exactly which cases were selected.
This does not establish audio/music parity with native APIs or agents.

## Costs and budget estimates

Local provider fees are zero; GPU hardware and electricity costs are excluded. Claude can report
an API-equivalent cost estimate. Codex cost stays **unknown** unless token rates are configured.
Neither value is the incremental invoice for a subscription CLI session. Cached input and Claude
cache creation are accounted for separately; missing required usage or rates stays unknown.
Reasoning tokens already included in output usage are not counted twice.

Optional model fields (the numbers below are illustrative, **not current model pricing**):

```json
{
  "name": "My native model",
  "backend": "codex",
  "model": "gpt-6.1-sol",
  "effort": "medium",
  "pricing": {"input": 2, "cached_input": 0.2, "output": 10},
  "estimated_tokens": {"input_tokens": 12000, "cached_input_tokens": 0, "output_tokens": 1000}
}
```

Prices are USD per million tokens. Input usage/forecasts include cached and cache-created tokens;
for Claude cache writes, add `pricing.cache_creation_input` and optionally
`estimated_tokens.cache_creation_input_tokens`. Forecasts should include CLI system prompts,
image tokens and reasoning. Set conservative estimates based on a pilot run.

`--max-cost 1.00` stops dispatch when completed estimates plus reserved forecasts for active and
next requests exceed $1. Every hosted model then needs both pricing and token forecasts. Errors
and missing costs reserve their forecast instead of counting as free. This is a **soft dispatch
guard**, not a provider spending cap: an individual request can exceed its forecast. No automatic
retries are made. Projected total is a pre-run forecast, not an invoice or accuracy guarantee.

## Reports and automation

Each output directory contains:

- `requests.jsonl`: flushed started/completed events, answers, grades, latency, token usage and costs.
- `report.json`: atomic checkpoints, selected cases/rubrics, workload fingerprint, CLI/runtime
  metadata, overall and per-category correctness, costs and request details.
- `report.md`: readable current/final summary.

Use `--no-tui` for CI or redirected logs; non-terminal stdout defaults to plain progress automatically.
`--parallel 1` is the default to avoid GPU contention. Increase it for throughput experiments; local
model queues and hosted rate limits affect the resulting latency. `--repeats`, `--seed`, `--kind`,
`--category`, `--limit`, and `--timeout` control the workload. CLI deadlines include startup and
inference; local backend HTTP inference has its own timeout and active inference must drain.
Ctrl-C stops new dispatch, terminates active CLI process groups and waits for active local work.
Completed results are retained. Abrupt process/power loss can leave unmatched started events;
those requests might have consumed usage. There is no automatic replay/resume.

Exit status: `0` completed and every model meets `--min-accuracy` (default 0.9) with zero errors;
`1` quality gate failed; `2` configuration/setup failure or soft budget stop; `130` interrupted.
The simulated demo returns success independently of its deliberately simulated failures.

Strict correctness means every expected field matches the deterministic rubric; mean field score
also shows partial correctness. Invalid JSON and provider errors score zero. Truncated local
answers cannot pass. These measurements cover the selected labeled tasks, not general perception
parity. Freeze representative held-out scenes before using the results to choose a model or
fallback policy. Consensus, native escalation and paired confidence intervals remain future work.

Adapter references: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode),
[Claude headless mode](https://code.claude.com/docs/en/headless),
[Claude image input](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode),
and [Rich Live](https://rich.readthedocs.io/en/latest/live.html).
