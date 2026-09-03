# September optimization track

## Objective

Improve NInfer as a single-host, single-GPU inference engine for coding engines. The fixed product
target is Qwen3.8-27B on one RTX 5090 (`sm_120a`), with one resident model instance, CUDA Graphs,
prefix reuse, Vision, speculative decoding, and one to eight active requests.

The primary product metrics are:

- committed decode tokens per second at representative batch sizes and context depths;
- time to first token for large prompts and reused coding-agent conversations;
- p95 inter-token latency while another request is prefilling;
- useful speculative tokens per round and per unit of GPU time;
- correctness and output quality on coding, reasoning, tool-call, and structured-output traffic.

Optimizations for multi-node deployment, generic model discovery, preemptive large-scale continuous
batching, and hardware other than the registered RTX 5090 target are out of scope.

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` completed and supported by relevant evidence
- `[-]` rejected by measurement or superseded

## Work order

### 1. Re-establish the MTP7 versus DFlash2 baseline

Status: `[~]`

The existing head-to-head measurements predate the DFlash2 selector-lattice and selector-scale
fixes. They cannot select the production backend. Repeat the comparison before changing either
implementation.

Protocol:

- use the same current `ninfer-serve.exe` binary for both sides;
- use the same Qwen3.8-27B `nvfp4full-v2` artifact for both sides;
- compare MTP with `--draft-tokens 7 --adaptive-mtp --lm-head-draft` against DFlash2 with
  `--draft-tokens 7`; DFlash2 requires its full-vocabulary proposal head and rejects
  `--lm-head-draft`;
- hold KV dtype, context capacity, RAM tier, CUDA Graph mode, sampler, output budget, concurrency,
  prefix-reuse state, and corpus order constant;
- collect full-precision `--request-log-jsonl` records, including per-slot acceptance;
- cover shallow, medium, and deep contexts and coding-agent traffic;
- report C=1 first, then the supported serving concurrency points that materially differ;
- compare committed tok/s, wall time, tokens/round, per-slot survival, TTFT, and output validity.

Completion criterion: a same-build, same-artifact result that identifies where MTP7 or DFlash2 is
faster and whether a fixed backend is adequate. The result must also establish whether adaptive
DFlash2 or an internal backend-selection policy is worth implementing.

Current notes:

- the root `.bck.bat` preset is historical and still describes the v1/MTP3 lane; do not use it for
  this comparison;
- `build-ninja/apps/ninfer-serve.exe` is present;
- the current DFlash2 preset has local user changes and must not be overwritten by this track;
- the existing `run_serve_corpus.py` runner only models MTP3 and the older 35B DFlash route, so it
  cannot be used unchanged for this Qwen3.8 MTP7/DFlash2 comparison.

First clean depth discriminator, 2026-09-01:

| True prompt | Backend | Completion | Finish | Prefill tok/s | TTFT | Decode tok/s | Tokens/round | Acceptance |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 24,091 | MTP7 adaptive | 512 | output limit | 8,566.6 | 2,879 ms | 150.1 | 3.12 | 59.3% |
| 24,091 | DFlash2-7 | 320 | stop token | 7,365.4 | 3,335 ms | 146.7 | 3.40 | 34.3% |
| 98,470 | MTP7 adaptive | 224 | stop token | 4,532.7 | 22,002 ms | 103.6 | 2.84 | 46.5% |
| 98,470 | DFlash2-7 | 351 | stop token | 4,202.9 | 23,733 ms | 131.9 | 3.82 | 40.2% |
| 196,337 | MTP7 adaptive | 512 | output limit | 2,846.1 | 69,515 ms | 79.1 | 3.03 | 54.4% |
| 196,337 | DFlash2-7 | 512 | output limit | 2,643.9 | 74,784 ms | 144.0 | 5.75 | 68.1% |

Conditions: current `build-ninja/apps/ninfer-serve.exe`, the same
`qwen3_8_27b_nvfp4full-v2.ninfer`, RTX 5090, HQ-E8-2B KV, C=4 with one request submitted at a
time, explicit 400,000-token shared KV capacity, CUDA Graphs on, prefix reuse off, greedy sampling,
one 32-token warmup before each side, and 512 requested output tokens. Raw full-precision records
are in `profiles/sept-opt-clean-{mtp7,dflash2}.jsonl`.

Initial interpretation:

- DFlash2 decode is 2.3% slower at 24K, 27.3% faster at 98K, and 82.0% faster at 196K;
- DFlash2 pays a TTFT penalty of 0.46, 1.73, and 5.27 seconds respectively on these cells;
- using phase rates, the TTFT penalty is recovered after roughly 840 generated tokens at 98K and
  925 generated tokens at 196K; short tool/classifier calls can therefore still favor MTP even at
  depth;
- greedy termination differs between the backends on the 24K and 98K synthetic fixtures. The
  phase-throughput result remains useful, but the fixed backend cannot be selected until the
  behavior is audited on real coding traffic;
- these fixtures are repetitive synthetic depth discriminators. The next required subtest is a
  replayable coding/reasoning/tool corpus, followed by the relevant concurrency points.

### 2. Reduce decode launch fragmentation

Status: `[x]`

The current 27B decode step contains roughly 630 kernels. CUDA Graph replay removes repeated host
submission work, but it does not remove device-side dependency gaps, drain/ramp loss, or redundant
materialization.

Candidate changes, in order:

1. capture one representative graph replay with Nsight Systems and quantify gaps plus kernels below
   5 microseconds;
2. extend Programmatic Dependent Launch to the decode kernels that currently lack it;
3. fuse Q norm, K norm, and RoPE;
4. fold `sigmoid_mul` into the attention reducer and bypass the reducer when `splits == 1`;
5. fuse input/post-mixer norms into their consuming projections where the public Op result permits;
6. fuse the 27B norm and GDN control projection;
7. fold `gated_rmsnorm` into the GDN output-projection prologue.

Expected end-to-end opportunity: approximately 8-12% decode throughput, subject to profiling.
Every fused route must be qualified directly against its independent semantic oracle.

First measured pass, 2026-09-02:

- Nsight Systems captured 16 ordinary-decode CUDA Graph replays on the RTX 5090 with the current
  build, `qwen3_8_27b_nvfp4full-v2.ninfer`, HQ-E8-2B, greedy sampling, and no other NInfer,
  Mystify, or NVIDIA Overlay process present. The baseline graph has 534 kernel nodes per replay,
  not the approximately 630 estimated by the older desk note.
- Representative steady replay 8 spans 11.779 ms. It contains 230 kernels below 5 microseconds
  (43.1% of launches), totaling 760.698 microseconds (6.36% of cumulative kernel time). Positive
  dependency gaps total only 24 microseconds; none is at least 1 microsecond. Across all 16
  replays, sub-5-microsecond launches are 42.7-45.9% and positive gaps are 24-27 microseconds,
  again with none at least 1 microsecond. Extending PDL is therefore not admitted by this trace.
- The current source already contains the broad PDL chain, fused Q/K norm plus RoPE, attention
  sigmoid-gate folding, and the fused 27B norm/control route. Those roadmap candidates predate
  this measurement and must not be counted as new September gains.
- A fused post-mixer RMSNorm plus NVFP4 LinearSwiGLU prototype passed its numerical oracle but
  reduced median end-to-end decode from 78.46 to 61.22 tok/s (-22.0%). The persistent-CTA design
  disrupted the existing PDL/weight-streaming schedule and was fully removed; do not retry that
  design without a materially different scheduling strategy.
- The first retained optimization folds the ordinary hidden-state scatter into partial block zero
  of the already-launched multiblock sampler. Its independent contract test checks exact sampled
  tokens, exact BF16 destination contents including untouched slots, read-only inputs, guards, and
  workspace accounting. The optimized graph has 533 nodes per replay and no standalone
  `scatter_bf16x8_kernel`. Across the 16 captures, the old scatter plus two sampling kernels cost
  7.054 microseconds per replay on average; the two fused-path sampling kernels cost 5.832
  microseconds, saving 1.222 microseconds at the tail.
- A tightly interleaved rebuild-level CLI A/B at 128 generated tokens measured baseline decode
  81.16 / 81.74 / 81.69 tok/s (median 81.69) and fused decode 81.91 / 82.00 / 81.96 tok/s
  (median 81.96, +0.33%). This is a structural launch reduction with a small non-regressive
  end-to-end result, not evidence for the full 8-12% item-level opportunity.
- `ninfer_sampling_test` passes, including the new fused-route semantic oracle, and both
  `ninfer.exe` and `ninfer-serve.exe` build successfully. `git diff --check` is clean. No benchmark
  process remains running, no commit was created, and the user's local DFlash2 preset change was
  not modified.

Profiles: `profiles/nsys/sept-k2-baseline-tg16.{nsys-rep,sqlite}` and
`profiles/nsys/sept-k2-scatter-sampling-tg16.{nsys-rep,sqlite}`. Next, use the remaining short
kernel census to choose a producer-side or reducer-side fusion; do not add another persistent
consumer that serializes the established PDL chain.

Second measured pass, 2026-09-02:

- The retained 533-node profile was censused by kernel role. Per replay, the dominant short
  launches are 81 wide RMSNorms (about 290 microseconds including one first-use outlier), 48 GDN
  recurrent kernels (about 192 microseconds), 48 GDN gated RMSNorms (about 101 microseconds), 16
  attention reducers (about 44 microseconds), and 16 already-fused Q/K norm plus RoPE kernels
  (about 59 microseconds). These are semantic computations, not redundant copies analogous to the
  removed scatter.
- The proposed `splits == 1` attention-reducer bypass does not apply to the measured small-T decode
  route: its active policy has a minimum of four splits. Folding the reducer into the producer
  would require a last-CTA atomic/counter protocol while retaining essentially the same reduction
  work; it is not admitted without an operator microbenchmark that first demonstrates a useful
  saving.
- Producer-side GDN gated-RMSNorm fusion is not a local epilogue change. The recurrent kernel
  partitions each 128-element value head across independent state tiles, so no producer CTA owns
  the complete normalization row. Folding it into the 320-CTA output GEMV instead would duplicate
  row-statistic or gated-input work across consumers. This is lower-confidence than the rejected
  persistent consumer and is deferred unless a materially different schedule is designed.
- The two NVFP4 GEMV families plus the final W8 projection account for about 10.4 ms of the
  11.8-ms shallow replay. The measured graph has no material device dependency gaps and the
  remaining independently removable launch work is below the original 8-12% estimate. Decode
  launch fragmentation is therefore closed with the one retained 534-to-533-node fusion; the
  original item-level opportunity is refuted by profiling.
- A post-census smoke on the same current build, v2 artifact, RTX 5090, HQ-E8-2B, greedy sampling,
  and an otherwise NInfer-idle GPU completed 16 output tokens correctly at 76.40 decode tok/s.
  The short output budget is a functional feedback-loop result, not a new throughput baseline.

If launch elimination is revisited, the only bounded experiment still worth considering is an
HQ small-T last-CTA reducer microbenchmark with exact attention-output comparison and an immediate
end-to-end rejection gate. The stronger next decode investigation is the dominant NVFP4
GEMV/LinearSwiGLU schedule itself, not further fusion of sub-5-microsecond semantic kernels.

### 3. Make speculative work adaptive

Status: `[ ]`

MTP already supports an internally selected physical width. DFlash2 currently pays for a fixed
seven-token proposal even when late proposal positions have poor survival.

Candidate design:

- pre-capture exact-batch graph definitions for DFlash2 widths one through seven;
- select width from filtered per-position survival and measured round cost;
- optimize expected committed tokens per unit of GPU time, not acceptance percentage alone;
- retain one whole-batch backend and width per round; do not split active rows into cohorts;
- consider MTP/DFlash2 selection only at a seam that preserves compact whole-batch execution.

The baseline in item 1 is the admission gate for this work.

### 4. Deepen prefix reuse for coding-agent workloads

Status: `[ ]`

The present retained-prefix path is single-owner and checkpoint-limited. Coding engines repeatedly
reuse system prompts, repository context, tool schemas, conversation history, and agent branches.

Candidate design:

- content-addressed lookup over tokens, positions, media identity, and frontend mode;
- multiple semantic checkpoints at message and tool boundaries;
- immutable refcounted KV pages and copy-on-write growth from the selected checkpoint;
- matching GDN, ReplaySSM, hidden, position, and speculative-backend snapshots;
- VRAM-first retention with the existing pinned-host RAM tier as the second tier;
- active-request admission remains more important than cache residency.

Success is measured primarily in computed-prefill tokens, TTFT, and serving capacity on real coding
sessions, not in isolated decode tok/s.

### 5. Accelerate long-prompt prefill attention

Status: `[ ]`

At contexts above roughly 32K, causal attention dominates prefill. Start with the zero-code
discriminator: compare `--prefill-chunk 1024`, `2048`, and `4096` at 64K and 128K.

If the measurement confirms wave-quantization loss, investigate:

- a persistent CTA scheduler with longest-work-first assignment;
- FP16 per-tile PV accumulation with FP32 running accumulation on the INT8 route;
- overlap of softmax with QK/PV MMA work;
- adaptive prefill chunk size that protects decode latency under concurrent serving.

Estimated opportunity from the existing analysis: about 15% less prefill time at 64K and 30-40%
at approximately 260K, to be confirmed end to end.

### 6. Skip low-value long-context attention work

Status: `[ ]`

For HQ-E8-2B, further codec micro-optimization is less promising than avoiding K/V rows that cannot
materially affect the result.

Candidate sequence:

1. bound-managed BLASST-style V skipping with skipped-mass metadata carried through split reduction;
2. Quest-style page-level K selection in the rotated frame;
3. always retain attention sinks and the recent window;
4. union selected pages across the Q heads in each GQA group;
5. consider asymmetric HQ-K plus INT4-G64-V storage if combined selection is not effective.

These are numerically approximate production profiles. They require needle retrieval, coding
corpus, long-context quality, and attention-sink gates before becoming defaults.

### 7. Improve NVFP4 W4A4 prefill efficiency

Status: `[ ]`

The current W4A4 GEMM reaches roughly 53-59% of the RTX 5090 FP4 peak on registered shapes.

Candidate work:

- profile the real shapes and compare them with the matching CUTLASS SM120 examples;
- test two versus three TMA stages for `MlpGateUp`;
- improve overlap of TMA, `ldmatrix`, and MMA issue;
- fuse A4 activation quantization into RMSNorm/SwiGLU producer epilogues.

Expected opportunity: 10-15% on short-prompt prefill if the main GEMMs approach 75% of peak, plus
roughly 3% from removing standalone activation-quantization passes.

### 8. Specialize the artifact and closed execution package further

Status: `[ ]`

The single-model/single-GPU contract permits explicit choices that a generic runtime cannot make:

- evaluate the remaining BF16 exception parents in `nvfp4full`;
- evaluate a smaller or faster `lm_head` representation;
- remove unreachable target/profile dispatch from the final Qwen3.8 execution package;
- precompute all admitted shapes, graph profiles, and route choices;
- retain only changes whose quality and end-to-end gains are demonstrated.

The host implementation language is not itself a performance lever while one CUDA Graph replay is
the steady-state host action. A host rewrite must therefore be justified by ownership or
maintainability goals, not counted as a decode optimization.

## Measurement discipline

For every item:

1. freeze binary, artifact, workload, sampler, context/KV settings, concurrency, and GPU-idle
   conditions;
2. measure at the level of the claim: operator, graph replay, request phase, or complete serving
   workload;
3. retain only concise provenance and summarized results;
4. verify numerical or behavioral correctness using the applicable independent oracle;
5. update this file with the result, decision, and next work item;
6. reject an optimization when the relevant end-to-end result does not improve.

## Results log

### 2026-09-01 — rejected initial MTP7 run

Status: `[-]`

The first attempted MTP7 run was started while another NInfer process was already resident on the
GPU. The other process was closed after the new server had been launched, but the startup and early
measurement conditions were not isolated. All measurements from `profiles/sept-opt-mtp7.*` in that
attempt are rejected and must not be used in an A/B comparison. The owned benchmark server was
stopped and the process list was verified to contain no remaining `ninfer*` process before retrying.

No September optimization result has been admitted yet. Item 1 remains in progress.

### 2026-09-01 — clean MTP7/DFlash2 depth discriminator

Status: `[x]` as an item-1 subtest; item 1 remains `[~]`.

The clean same-build/same-artifact depth discriminator is recorded under item 1. It confirms a
strong context-depth crossover and a TTFT-versus-long-generation tradeoff, so adaptive DFlash2 or
an internal backend-selection policy remains a justified candidate. It does not complete item 1
because the synthetic greedy outputs diverged and no replayable real coding corpus has yet been
run on both backends.
