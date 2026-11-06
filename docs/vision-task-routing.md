# Vision task routing and model qualification

Use task-specific qualification rather than declaring a local model equivalent to
native vision. The October 19, 2026 evidence identifies useful local candidates
and clear failure modes. It does not yet qualify an automatic production router.
The analysis and frozen reference suites are preserved for future small models.

The conservative rules are now encoded in
[`benchmarks/routing.policy.json`](../benchmarks/routing.policy.json), with offline
recommendation and report-rejection helpers in `benchmark/routing.py`. The primary
recommendation is native; only an explicit shadow request for a tested image
candidate and the recorded model digest enables a local observation alongside it.
Known failure tasks, other models and video remain outside that shadow envelope.
There are no promoted local tasks, and installed-server dispatch is not changed.
Malformed responses, missing required fields and runtime token-limit stops are
rejected. Valid JSON alone never promotes a local answer. The Service now also
marks Ollama `done_reason: length` as truncated at the source.
The real-video pilot subsequently added dense-table exact-row extraction to the
current model's explicit native rules after wrong-row answers in both a still
control and its video counterpart. It did not promote any video category.

## What the current model did well

These are **first local observations**, before host correction, from the two
completed Qwen 3.5 4B single-pass MCP experiments. Each category has five generated
scene variants, each queried under Claude and Codex: ten observations, not ten
independent scenes. The runs had different local context settings; these pooled
counts describe failures and are not a controlled comparison of configurations.
Full per-run settings, usage, timing and case IDs remain in `task-gap.json`.

| Task | First local correct | Final host correct | Interpretation |
|---|---:|---:|---|
| Large, clear error identifiers | 10/10 | 10/10 | Strong candidate for focused local OCR |
| Small numbers of separated colored rectangles | 10/10 | 10/10 | Candidate for simple visual counting |
| Clearly labeled, visible commit SHA | 10/10 | 10/10 | Candidate for visible-field extraction; does not establish absence detection |
| Read an error despite an embedded instruction | 10/10 | 10/10 | Passed this specific injection pattern; not broad injection resistance |
| Maximum Requests in a three-row table | 6/10 | 10/10 | Correct service was visible in failed responses; two truncated and two malformed JSON responses required host repair |
| Exact build identifier after cropping | 6/10 | 6/10 | Crop helps, but omitted prefixes still cause failures |
| Exact tiny build identifier in the full scene | 0/10 | 2/10 | Native for current model: systematic `build-` prefix loss |
| Detect an absent commit SHA | 2/10 | 7/10 | Native for current model: substituted a build identifier for a missing SHA |

Native perception passed 10/10 observations in each category. Current local
consensus passed 4/4 first observations in each of the first five categories and
cropped OCR, but 0/4 on tiny full-scene identifiers and 0/4 on absence checks.
Those consensus counts cover only two scene variants under two hosts. Voting
did not resolve correlated mistakes, so consensus is not the default escalation.

A separate [real-video pilot](video-benchmark.md) tests matched still controls,
multi-frame perception and sampling coverage on short Windows video clips.
It completed 81 sessions: local scored 7/9 on still controls and 10/18 on video;
each native host scored 9/9 and 18/18. Five local strict failures contained correct
values under renamed keys, while a six-frame sequence missed black frames that
were actually supplied. Output contracts and temporal coverage need separate
qualification; more frames alone are not a reliable fix.

Table errors demonstrate the difference between perception, formatting and host
reasoning. The local model selected the correct service in all audited table
responses, but this does not make a malformed or truncated report a qualified
structured result. The analysis treats runtime `done_reason: length` as truncated
even when the historical tool response's `truncated` flag was false. Historical
final host scores are preserved unchanged.

No tested category showed fewer **total hosted tokens** with MCP than its matched
native route. Thus accuracy candidates are not yet efficiency wins. Current
fresh-session/tool overhead, retries and cache behavior are included. A compact
local report in a longer, existing host session might amortize overhead; that is
a hypothesis to test with complete workflows, not a savings claim.

## Proposed routing boundary

Start with one local pass for narrowly described, low-impact extraction or
counting on clear inputs, in a shadow trial. Table maximum queries need a strict
output schema and truncation checks. Crop before resizing when the region is
known. Do not assume that enlarging a whole screenshot recovers exact characters.

Choose native directly for exact tiny identifiers, missing-field decisions,
ambiguous labels, incomplete coverage, or tasks outside the qualified task/input
envelope. Dense charts, complex spatial reasoning, degraded photos, video and
audio remain **untested here**, rather than demonstrated native-only tasks.
An absent-field question cannot be routed using whether the answer is actually
absent: that is hidden ground truth. Until presence/absence is jointly qualified,
questions asking for an optional field should use native.

The future router should use the requested operation, known input properties and
externally supplied region/labels before inference. After inference, schema,
length, coverage and domain checks can reject bad outputs. A checksum or known
field label is stronger evidence than a model confidence claim. A plausible
identifier or a unanimous vote does not establish correctness. Route unresolved
outputs to native; do not fabricate a replacement or repeatedly vote on the same
view. Correct-but-wrong-field outputs must remain in the evaluation.

## Benchmark the boundary before promotion

The saved protocol is [task-gap.protocol.json](../benchmarks/task-gap.protocol.json).
Its thresholds are proposed starting targets, not measured guarantees.

1. **Keep these cases as a fixed regression set.** Save every observed failure and
   its required answer. New models must preserve literal prefixes and abstention,
   not just improve aggregate scores. This observed set is no longer held out.
2. **Create development and hidden evaluation sets from different source scenes.**
   Include real screenshots and new layouts. Sweep font size, text density,
   contrast, blur, crop boundaries, rotation, table size, count/occlusion and
   confusing adjacent labels. Pair present/absent fields. Separate severity from
   probability; do not lower the standard merely because most questions are easy.
3. **Measure the local model directly first.** Freeze model digest, quantization,
   prompt, output format, crop/resize and runtime. Score exact fields, semantic
   mistakes, malformed/truncated output and abstention separately. Host repairs
   must not count as local success. Keep the native baseline on the same pixels.
4. **Freeze and test the actual hybrid route.** The router cannot read category
   labels supplied only by the evaluator, expected answers or local correctness.
   Route using information available in production. Measure the share handled
   locally, accuracy among accepted local answers, missed escalations, avoidable
   escalations, final native-assisted quality, and fallback frequency. Include
   local-first-then-native work and errors in every denominator and usage total.
5. **Gate quality and economics separately.** A suggested initial local-acceptance
   target is at least 99% exact correctness with a one-sided 95% lower confidence
   bound at least 99%, and no known critical failure. Under independent sampling,
   300 successes / 300 trials gives a lower bound of approximately 99.01%.
   Repeated hosts or many questions on one screenshot do not supply 300 independent
   trials. Also predefine an acceptable paired quality loss against native and a
   minimum coverage; a router accepting nothing cannot qualify as useful.
6. **Validate savings in real workflows.** Compare always-native with the frozen
   hybrid in fresh and existing sessions. Record hosted input/output/cache tokens,
   native image uploads, local calls, fallback work, memory, and end-to-end median
   and p95 latency. Predefine a savings target, for example at least 20% fewer
   hosted tokens at matched quality. Keep local tokens and monetary estimates
   separate. Confirm a proposed win on independent held-out workflows.

Publish qualifications by task and input envelope, tied to exact model/runtime
and preprocessing fingerprints. A model, prompt, crop policy or serving change
requires requalification. Run small shadow audits after deployment, retain new
failures in the regression corpus, and suspend a category when it drifts. A future
small model can expand the qualified envelope without changing the native
baseline or silently weakening the scoring rules.

## Reproduce the current gap analysis

This command reads saved results and starts no inference or hosted sessions:

```sh
.venv/bin/python -m local_vision_mcp.benchmark.task_gap \
  benchmarks/results/agent-efficiency-20261019-isolated/report.json \
  benchmarks/results/consensus-heldout-20261019/report.json \
  --output benchmarks/results/task-gap-20261019
```

It writes per-run `task-gap.json` and `task-gap.md`, including first local accuracy,
host recovery, failure reasons, distinct media hashes, per-category paired native
comparisons, local tokens, hosted usage and latency. Missing original-question
observations and unknown usage remain explicit. No routing policy is activated.
The source-report SHA-256 fingerprints allow later audits of the analysis.

Recreate the reference suites using the seeds and configurations in the
[single-pass](agent-efficiency-benchmark.md) and [consensus](consensus-benchmark.md)
reports. Raw results and generated images are gitignored; the generator, protocol,
configurations and diagnostic findings are saved in the repository.
