# Structured local output and conservative shadow routing

The optional `response_schema` argument is supported by `analyze_image`, silent
`analyze_video` and `submit_media_job` with the Ollama backend. It reaches both
synchronous and durable execution and participates in durable request identity.
Omitting it preserves the previous free-form behavior and receipt identity.

For example, ask for a count with this contract:

```json
{
  "path": "/absolute/media/root/screen.png",
  "question": "Count the blue shapes. Use null if unreadable.",
  "response_schema": {
    "type": "object",
    "properties": {"count": {"type": ["integer", "null"]}},
    "required": ["count"],
    "additionalProperties": false
  }
}
```

Schemas must be self-contained JSON object schemas, at most 32 KiB and 16 nesting
levels. References are rejected. Validation uses JSON Schema Draft 2020-12;
format annotations are not domain verification. Audio/Omni and experimental
consensus do not implement this contract and reject it explicitly.

The backend sends Ollama's `format` schema and adds the schema to the prompt,
following [Ollama's documented interface](https://docs.ollama.com/capabilities/structured-outputs).
The Service validates the returned answer independently and adds
`output_validation: {"valid": true|false, "reason": "..."}`. It preserves the
answer for audit and does not rename keys, repair values, strip fences, retry or
raise a successful visual claim from validation alone. Callers must reject
`valid: false` or `truncated: true`. Required nullable fields permit abstention
without changing keys. Valid output can still select the wrong row or object.

In development, Ollama 0.30.6 with Qwen 3.5 4B and `think:false` returned fenced
JSON despite a schema request. An upstream [issue reports ignored format
constraints for this combination](https://github.com/ollama/ollama/issues/14645).
Five initial development observations are retained in `video-regression`; that
run was stopped. The revised instruction explicitly forbids Markdown fences.
`video-regression-v2` reruns all 27 previously observed pilot cases. This is a
combined schema-request/prompt experiment, not an isolated estimate of grammar
masking's benefit. No runtime upgrade or model swap is included.

The first screening attempt (seed 2026102002) was also stopped after the task
review found an ambiguous counting prompt: a blue title bar could be counted as a
blue shape. All its completed observations are development data. The replacement
seed 2026102003 uses neutral gray headers and white backgrounds for count scenes,
and explicitly excludes page decorations. The original media, questions, labels,
answers and partial checkpoints are preserved; they are not relabeled or pooled.
An amendment retracts the initial interpretation of these off-by-one outputs as
confirmed visual mistakes. One started local request lacks a final receipt after
interruption and may have consumed unmeasured local usage; it remains explicit.

## Frozen screening

The [protocol](../benchmarks/structured-output-v2.protocol.json) freezes 128 new
synthetic scenes, 32 each for clear large OCR, separated blue-object counting,
explicit visible labeled fields and simple table maxima. Schemas contain field
types and nullability, never expected values or gold enums. Native CLIs receive
the same public contract in their prompts. The unconstrained local arm omits the
schema request and grounding instruction but retains the original task question.
All four arms face the same strict JSON/schema validation and existing field
scorer. That scorer normalizes string case and whitespace; literal text
deviations should be audited separately before claiming exact identifier
qualification. The primary scores are never repaired from alternate keys.

The screening seed differs from development. Eight related renderer families
vary placement, dimensions, background, typeface, count, table size and labels.
They share a generator with development; they are **not** hidden, independently
sourced real-world qualification. Even 32/32 independent successes would yield
only about a 91.1% one-sided 95% lower bound. This screening cannot pass the
proposed 99% lower-bound gate and does not promote any local route.

```sh
.venv/bin/python -m local_vision_mcp.benchmark.qualification_fixtures \
  --output benchmarks/generated/structured-screening-v2-20261020 \
  --seed 2026102003 --per-task 32 --split screening
.venv/bin/local-vision-bench compare \
  --manifest benchmarks/generated/structured-screening-v2-20261020/manifest.json \
  --config benchmarks/compare.structured-screening.json \
  --output benchmarks/results/structured-output-20261020/screening-v2 \
  --parallel 1 --timeout 300 --seed 42 --no-tui
.venv/bin/python -m local_vision_mcp.benchmark.qualification_report \
  benchmarks/results/structured-output-20261020/screening-v2/report.json \
  --output benchmarks/results/structured-output-20261020/screening-v2/analysis
```

Use empty output directories; existing results are never overwritten. These
configs use an independently owned Ollama on port 11435, with installed weights,
16K context, temperature zero, thinking disabled and a 128-token answer budget.
Codex and Claude each use native perception individually, with no consensus.
The native baseline uses the existing CLI adapters and identical sampled pixels.

## Measured shadow workflow

`perception: "shadow"` with `local_model` and `routing_policy` selects an
executable benchmark route. It uses a public `routing_task` supplied alongside
the production request, not hidden rubric/category metadata. Only the recorded
model digest, image modality and candidate task permit a local shadow pass.
Unknown requests/models, video and known failure tasks skip local inference.
Native still supplies every primary answer; no local output is promoted.

The fixed workflow subset contains the first four screening cases per category,
selected before any screening result. It compares both native baselines against
both native-plus-local-shadow arms on 16 scenes, 64 complete workflows. Local
runs before native; native receives the original pixels and no local report.
The shadow adapter calls the same local Service directly; MCP transport and host
tool-schema overhead are not included in these workflow measurements.
Local errors do not suppress native work. Native calls, local calls, accepted
local count, errors, end-to-end latency and both tokenizers' usage are retained.
This measures the current policy's real shadow overhead. It cannot demonstrate
savings from a local-accepting router because that route has not been qualified.

The [shadow config](../benchmarks/compare.structured-shadow.json) runs through the
same `compare` command with its frozen 16-case manifest. `qualification_report`
separates hosted usage from nested local observations, including their failures.
No global agent configuration, installed runtime payload or default dispatch is
changed by these benchmark adapters. The source plugin skill documents the
same conservative boundary for a future installed generation.

## Artifacts

Raw media and results remain local and gitignored under
`benchmarks/generated/structured-*-20261020` and
`benchmarks/results/structured-output-20261020`. Frozen provenance records bind
source/config/manifest hashes; reports retain all answers and failure cases.
Telemetry records model residency and GPU memory. Development, screening and
shadow results must remain separate. Observed failures become regression data
for future model digests rather than justification for post-hoc routing rules.
