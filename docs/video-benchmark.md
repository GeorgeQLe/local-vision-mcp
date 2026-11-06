# Real-video capability pilot

## Frozen experiment — October 19, 2026

The Windows source directory is `/mnt/c/Users/Owner/Videos`. An FFprobe inventory
found 195 video files, 194 readable, with 164 at most 60 seconds long. Many are
alternate exports. A 0.067-second recording was excluded as unusable. Originals
were left unchanged; eight selected files were copied into the benchmark's
allowed media directory and assigned neutral clip filenames.

The selected clips last 3.133, 3.133, 9.664, 12, 12, 12, 15.25 and 33.267 seconds.
They contain a webcam capture, desktop capture, narrated visual animation, three
silent computing animations, an app demo and an unboxing. A gameplay-folder
filename was not used as a content label: the selected clip actually shows an app.
The webcam and desktop belong to one recording session; three animations share
a template. These are eight source files, not eight independent content families.

- Models: installed Qwen 3.5 4B through the same local Service used by MCP; native
  Codex `gpt-6.1-sol` and Claude `claude-opus-5-5`, medium effort.
- 27 tasks per model, **81 sessions**: nine static image controls and eighteen
  video cases, one repeat, shuffled seed 42, sequential execution, 300-second
  session timeout. No model downloads, consensus or automatic retries.
- Static tasks: people counting, exact table OCR, large heading OCR, labeled-field
  extraction and object presence with a negative control. Each of nine video
  tasks has a still control using its exact final sampled frame.
- Temporal tasks: visible black/person states and numeric values in specific
  memory/total boxes in the first and last supplied frames. Each is tested with
  one, three and six uniformly spaced midpoint frames.
- Every prepared frame has a maximum edge of 1,024 pixels. Native agents receive
  the same sampled pixels as local, not a continuously decoded video upload.
  Controls preserve those pixels, though PNG encoders may differ.
- Local temperature 0, 16,384 context tokens, 128 output tokens. The local runtime
  is warmed with an unrelated color probe. Prompt caches/model residency may
  persist; these are warm, shuffled observations rather than controlled cold runs.
  Local prompt caches from the interrupted transport-debugging prefix were retained.
- Labels were manually established by the supervising agent from source review
  before any tested provider ran. These are not independently human-adjudicated
  labels. Questions, exact sampled-view rubrics, configuration, source copies
  and inference code were fingerprinted before measurement. No held-out prompt
  tuning is allowed during the run. Rubrics never enter provider inputs.
- Audio is excluded even where source clips have audio. The study says nothing
  about speech, sound events, music or audio/visual synchronization.

## Perception versus sampling

A sampled-view answer is graded against what the supplied frames actually show.
For example, the camera's midpoint shows a person but not its initial black
screen. The correct one-frame answer omits the black state, yet that input does
not cover the known source transition. Both facts must remain visible.

Frozen reference annotations record three one-frame cases missing a known source
state/value transition. This is a coverage limitation, not automatically a model
error. The reference states/endpoints were established from source review; they
are not an exhaustive annotation of every intermediate event. Native agents
cannot recover unobserved events from the identical samples either. Resampling,
scene-change analysis or a different video input path must address that gap.

The pilot compares direct local perception with native CLI perception. It does
not execute a hybrid router or measure MCP-host overhead, and zero hosted tokens
for a direct local call is not evidence of end-to-end hosted-token savings.

## Artifacts and reproduction

The private generated dataset is `benchmarks/generated/video-gap-20261019/`.
It contains `manifest.json`, source copies, still controls, review contact sheets,
`selected.json` with original paths and hashes, and the frozen `build-pilot.py`.
Generated media and results are gitignored. They contain personal source footage;
the comparison sends only selected frames to the authorized native CLI hosts.

`benchmarks/results/video-gap-20261019/` contains `inventory.json`, pre-run code,
configuration and manifest fingerprints, warmup/runtime metadata, telemetry and
the `comparison-fixedstdin/` session results. The first `comparison/` attempt was
interrupted at 7/81: a CLI stdin retry bug stranded large real-image payloads.
A delayed-reader / 2 MiB-input regression reproduced the failure. The fix provides
the complete input through an anonymous temporary file and retains deadline,
cancellation and output-size checks. `provenance-restart.json` records that only
the transport changed; media, labels, prompts, model settings and sampling stayed
fixed. The restarted experiment reruns all 81 sessions. Interrupted observations
are not pooled into its scores. Run a dedicated Ollama server on port 11435
with the existing model store. Never replace the shared runtime on port 11434.

```sh
.venv/bin/local-vision-bench compare \
  --manifest benchmarks/generated/video-gap-20261019/manifest.json \
  --config benchmarks/compare.video-gap.json \
  --output benchmarks/results/video-gap-repeat/comparison \
  --timeout 300 --no-tui
.venv/bin/python -m local_vision_mcp.benchmark.video_report \
  benchmarks/results/video-gap-repeat/comparison/report.json
.venv/bin/python -m local_vision_mcp.benchmark.task_gap \
  benchmarks/results/video-gap-repeat/comparison/report.json \
  --output benchmarks/results/video-gap-repeat/task-gap
```

Use new directories. The video report separates frame counts, matched still/video
controls, incorrect answers and known input coverage gaps. The task-gap report
provides per-category detail and keeps local tokenizer usage separate from hosted
usage. No category is automatically promoted. Future models must pass the fixed
regressions and new source-disjoint held-out tasks.

## Results

The corrected run completed **81/81 sessions in 14.95 minutes**, with no provider
execution errors and known token usage for every session. The accuracy-gate exit
was 1 because local missed the quality target; the experiment itself completed.

| Input subset | Local Qwen | Native Codex | Native Claude |
|---|---:|---:|---:|
| Matched still controls | 7/9 | 9/9 | 9/9 |
| One-frame temporal cases | 1/3 | 3/3 | 3/3 |
| Three-frame video cases | 9/12 | 12/12 | 12/12 |
| Six-frame temporal cases | 0/3 | 3/3 | 3/3 |
| All video cases | 10/18 | 18/18 | 18/18 |
| Overall | **17/27** | **27/27** | **27/27** |

The three- and six-frame aggregate subsets contain different task mixes. Within
the three paired temporal tasks, increasing sample count did not improve strict
accuracy. That is a diagnostic result on these tasks, not a general claim that
more frames reduce model accuracy.

### What worked

Local passed all large-heading OCR cases, people-counting cases and microphone
presence/absence cases. It also read array contents and temperature/threshold
fields correctly in both image and video versions. Static video questions passed
8/9, compared with 7/9 matched still controls. There were no pairs where a passing
still control became a failing static-video answer; the app-summary video passed
while its still control had renamed output fields. This does not establish broad
image/video equivalence: the matched subset has nine questions on eight related
source files.

### What failed

Ten local strict failures were reviewed after scoring; primary grades were not
changed. `failure-review.json` records the answers and exploratory diagnoses:

- **Five output-contract failures with the expected values:** app counts used
  `due_today` / `complete_today` instead of `due` / `complete`; four multi-frame
  value-change responses used expanded field names rather than `first` / `last`.
  These are candidates for a separately tested structured-output fix, not proof
  that native vision is needed to read those values. No post-hoc alias repair is
  counted as benchmark success.
- **Two wrong-row extractions:** the market-table still and video selected
  `US Tech 100` instead of the requested top `US 30` row. The video answer also
  contained an invalid unquoted numeric literal with a thousands separator.
  Repeating frames did not fix the row-selection error.
- **Two single-frame temporal answers with null values:** despite instructions
  to repeat the one visible value, local omitted the final value or both values.
  This mixes temporal task interpretation and extraction; the experiment does
  not isolate the cause.
- **One missed supplied state:** with six camera frames, local returned only
  `person`, omitting the two supplied black frames. The three-frame version
  correctly returned black then person. Both native hosts passed both versions.

The known input coverage gaps remain separate: all three midpoint-only temporal
inputs miss a reference state/value transition. Even correct native answers to
those sampled-view questions do not recover the original video's missing events.
Resampling addresses input coverage; model selection addresses interpretation of
the available frames. Video qualification needs both.

### Runtime and scope

Mean end-to-end time per task was 23.70 seconds local, 5.28 Codex and 4.21 Claude.
Local means were 10.39 seconds for still controls, 9.09 for one-frame temporal,
28.47 for three frames and 59.17 for six frames. These are warm observations under
uncontrolled background machine activity and cache reuse, not causal latency
comparisons of sampling density.

The 88 telemetry samples approximately scoped to the corrected benchmark all had
the local model resident, with total system GPU memory between 4,726 and 4,911 MiB.
The RTX 2080 SUPER has 8,192 MiB; model allocations were about 3.395 GB of GPU
memory. Ollama explicitly disabled vision-projector offload due to limited VRAM,
so vision encoding ran on CPU while language layers were offloaded to GPU.
Ollama was 0.30.6; Codex CLI 0.163.0; Claude Code 2.1.304; Python 3.12.12.

Native hosted totals were 374,277 tokens for Codex and 132,733 for Claude, including
cache usage. Local used 43,770 input and 545 output tokens under its own tokenizer.
These are not comparable tokenizers or billing estimates, and there is no measured
hybrid-token saving here. The direct-local arm does not include a hosted agent.

Post-run integrity checks found no changed benchmark media, original Windows
sources, configuration, prompts or inference code against the restarted freeze.
After measurement, the conservative image policy added explicit native handling
for dense-table exact-row extraction; that policy was not used by the providers.
Video remains unqualified and the primary recommendation stays native. The next
qualification stage still needs representative source-disjoint real tasks and an
actual frozen hybrid workflow; this pilot does not satisfy the 30–50 independent
examples per candidate category screening target.

The temporary runtime was stopped, the shared runtime remained available, and
final verification covered 73 tests plus lint, formatting and whitespace checks.
The generated `video.json` / `video.md` and task-gap reports retain per-case and
per-category detail. Raw telemetry includes the debugging prefix and post-run
idle time; `telemetry-final-run.jsonl` and `provenance-after.json` scope the reported
measurements approximately using the final report timestamp and elapsed duration.
