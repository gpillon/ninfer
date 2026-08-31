# gpillon fork — local engine changes

This is the maintainer reference for the work landed on `gpillon/coding`, stacked on top of the
cometkim integration branch (`cometkim/dev`, fork base `a082cbeb`). It records what changed and
why; commit subjects are cited as the audit trail, not reproduced verbatim.

The driving problem across most of this work is concurrent-request latency for coding-agent
traffic (Qwen Code and similar tools): a burst of subagent requests sharing most of their prompt
but sent to a single global prefill lane, which serialized redundant prefill and produced
multi-second time-to-first-token ladders. The host-RAM KV cache, prefix-reuse/admission work, and
tagged lanes below are three angles on that same problem; adaptive MTP and the device/perf fixes
are largely orthogonal, aimed at raw decode throughput and portability to a second GPU
(RTX 5090 Laptop).

## Host-RAM KV cache tier

A cache that snapshots finished or still-active GPU lanes into pinned host RAM, so a later request
can restore instead of re-prefilling.

**Base plumbing** (cherry-picked from `dylan/ninfer`, then repaired where the auto-merge silently
dropped or duplicated code): a system-RAM KV cache for finished chats, LRU eviction of the finished
GPU lane instead of lowest-index, and RAM usage/copy-time logging.

**Eviction policy**: plain FIFO by capture order let bursts of short one-shot classifier captures
evict long agent-conversation checkpoints about to be reused. Eviction is now two-tier: new
captures start probationary; a successful restore marks the content lineage (hash of the leading
tokens) hot, so the next capture for that lineage starts protected and is only evicted once every
probationary entry is gone.

**Active-lane sharing for identical concurrent requests**: reuse previously matched only against
lanes whose request had fully finished, so a sibling arriving while the source was still decoding —
the common case — got nothing. `find_admission_lane` now scans active lanes for a resumable
checkpoint whose leading tokens exactly match an incoming prompt (≥2048-token match), snapshots
that lane into the RAM cache on demand, and lets the sibling restore. First landed with three known
gaps, all closed in the immediate follow-up:
- duplicate per-sibling captures of the same lane during a burst — fixed by making a record track
  outstanding claims instead of a single pinned flag, so one snapshot serves every sibling in the
  burst;
- repeated re-capture/re-hash on every decode round after a non-reuse attempt — fixed by having
  each request remember the snapshot base it already tried per lane;
- an inverted no-op comment on the disabled-RAM-tier path — corrected.

**Two silent-corruption bugs found and fixed**, both discovered by symptom (wrong answers, not
crashes) and root-caused rather than patched around:
- *Rewrite-checkpoint restores from host RAM answered one request with another request's state.*
  The record's metadata and ledger validated correctly at every level checked — not a race, not
  wrong record selection, not missing state — so the defect had to be in the packed
  rewrite-checkpoint payload itself. Mitigated by disabling only that restore path from host RAM
  (`plan_match` now offers host-RAM records for append-at-frontier reuse only); the underlying
  packed-checkpoint defect is still open, and the note in the commit points at
  `pack_slot_to_host`/`unpack_slot_from_host` as the next place to look.
- *Any host-RAM restore under a hyperquant KV dtype served a request with another sequence's
  attention keys.* Hyperquant keeps an exact-key side store (sink rows + a recent ring in
  `PagedKVCache::residual_k_/residual_v_/ring_valid_`) indexed by block-table row, not by sequence,
  and the RAM cache record had no section for it — a restore bound a fresh row whose side store
  still held the previous tenant's keys, and the decode kernel read them unconditionally for the
  sink rows. First mitigated by disabling the RAM tier entirely whenever a side store is in use
  (`ram_tier_usable()`), then properly fixed by capturing `residual_k_`/`residual_v_`/`ring_valid_`
  into the record (six new sections, a header version bump to 2, strict read-header rejection of
  mismatched versions). Verified with adversarial row-recycling repros at 4x the original scale.

**Shared system+tools prefix for subagent bursts**: Qwen Code subagents share ~98% of their prompt
(system prompt + tool schemas) and diverge only in their task text, past the point reuse used to
match. A prefill chunk now splits exactly on the system/tools-to-first-user-message boundary (the
frontend supplies the boundary from `CompiledChatTemplate::render`, converted from a byte to a
token offset and gated to clean 4096-byte-minimum blocks), so the boundary state falls out of the
existing checkpoint machinery with no new state slot. This surfaced the same side-store gap as
above in the hand-built capture source used here (`capture_shared_prefix_boundary` didn't go
through the canonical builder and so missed `text_cache`/`backend_cache`/`residual_row`), fixed in
the same commit.

**Dynamic shared-prefix boundary capture and exact sizing**: boundary selection previously only
considered the static system+tools frontier, so it could propose a boundary that was already past
the winning plan's actual reuse point — hit exactly by identical prompts, where the longest-common-
prefix reaches the whole prompt — and drop it silently after paying for an extra chunk split.
Selection now takes the highest `reuse_base` any plan still in the running could contribute.
Admission separately estimated a capture's host-RAM cost from constants tuned for one 27B HQ/MTP
profile; `KVRamCache::capture_bytes()` is now the single sizing authority, built from the same
header/section-layout helpers the real capture uses, so a preflight can never disagree with the
capture it is preflighting. A `RamCapturePolicy::PreserveExisting` mode lets an admission-selected
capture fail cleanly instead of evicting an existing record — checked against the real allocator
first-fit span geometry (`HostPinnedArena::can_alloc()`), not just aggregate free bytes, since a
fragmented arena can have enough total space with no span large enough. See
`../../HANDOFF_RAM_ADMISSION_DETAILED.md` (untracked, local) for the full session record, including
the cross-stream `cudaMemset`/pageable-`cudaMemcpy` race that was making the KV-RAM test suites
look intermittently red for an unrelated reason.

## Admission and prefix-reuse measurement

Before committing to the active-lane-sharing mechanism above, real sibling-prefix overlap was
measured with no behavior change: `try_admit_one()` already holds a snapshot of every pending
request's prepared prompt, so `record_sibling_prefix_sample()` compares the request being admitted
against every other queued request's tokens and records the longest match (≥64-token floor) into
`RuntimeStats`, surfaced on the periodic throughput log line.

## Tagged request lanes (`@main` / `@agents` / `@classifier`)

A trailing `@main`, `@agents`, or `@classifier` suffix on the wire model id
(`split_model_request_class`) is threaded through `ExecutionOptions` → `RequestPlan` →
`SequenceState` as a `RequestClass`. Lane admission ranks `@main`-owned retained lanes last among
victim candidates — never excludes them, only reorders — so a burst of short-lived agent/classifier
requests recycles younger lanes first instead of evicting the long-lived main conversation for a
full re-prefill. Untagged traffic defaults to `Agents` and schedules exactly as before. Log schema
bumped to v14 with an appended `class=` field.

## Adaptive MTP verification width

- Width selection now picks a stable batch-wide MTP window from request-local accepted-prefix
  survival (sustained score wins + fresh tail evidence required, to avoid widening churn on a mixed
  request stream) while preserving exact-width CUDA Graph and ReplaySSM state; extended through K8
  with width occupancy/transition telemetry across CLI, serving, and benchmarks.
- Round-cost ownership moved into exact variants, selecting against the batch's current and future
  frontier maximum, so long-context startup and confidence probes stop using physically-dominated
  K6–K8 routes while still allowing K5 when tail evidence supports it.
- `MtpAdaptiveBatchController::round_cost()` previously read a single hardcoded, nearly-flat cost
  curve — real per-round wall-clock time is now fed into a per-(batch size, context-length band,
  width) bounded EWMA, and `round_cost()` derives a calibration factor from whichever width in a
  bucket has the most measured evidence and applies it across the whole prior curve for that
  bucket. Verified against production traces where the flat curve had left the width selector
  unable to narrow down despite acceptance swinging 40–96%.
- A cost-curve case for a `Qwen38Nvfp4LegacyW8` profile from an unrelated, not-ported
  endpoint-format-split commit was dropped in favor of the baseline curve, since that profile
  doesn't exist in this tree.
- GDN input projection retuned (16x128 W4A4 MMA tile for T=7..9, halving padded activation rows)
  and MTP verification retuned for laptop 5090 timing characteristics.

## Hardware portability and device fixes

- **RTX 5090 Laptop compatibility** (not this machine's own GPU — carried for portability, code
  path only): cooperative kernel launch support, and MTP verification/GDN projection tuning
  distinct from the desktop 5090 profile this fork actually runs on.
- **swiglu**: optimized for small batch sizes (NVFP4).
- **100% CPU during decode**: `cudaDeviceScheduleAuto` resolves to spin on this box, so the
  per-round `device.synchronize()` busy-waited a host core for the whole millisecond-scale GPU
  round. Host-side waits now use blocking sync (`cudaDeviceScheduleBlockingSync` at context
  creation, `synchronize()` routed through a `cudaEventBlockingSync` event) — hand-adapted from an
  upstream fix that assumed async-copy infrastructure this fork doesn't carry.

## Serve and streaming robustness

- **Tool-call XML leak**: the salvage path appended the unparseable remainder of a malformed or
  unclosed `<tool_call>` block — raw XML included — to visible content. Malformed blocks are now
  dropped silently (earlier calls still salvaged, later prose becomes content); a second, related
  bug let `ToolCallStreamFilter::finish()` replay the same raw XML on stream teardown once a tool
  marker had been confirmed, tripping a content-length invariant in `http_server.cpp` and crashing
  tool-capable streaming requests outright. The pre-existing test asserting the old all-or-nothing
  contract was rewritten against the new salvage contract rather than left red.
- **Warmup**: decoupled from `--pending-timeout-ms` (warmup now gets an explicit 60s override, so a
  tight configured deadline can't expire mid-warmup on a healthy engine) and made fail-fast, with
  auto KV-capacity bounds documented at the point of failure.
- **Frontend stream-marker handling** (`<think>` close markers, split across decode tokens): a
  short series of fixes addressing review-found regressions — holding a split marker until it
  resolves instead of scanning tokens in isolation, flushing a held marker prefix as content at
  termination instead of dropping it, routing raw-output sessions around marker cleanup entirely
  (raw mode must expose the stream byte-for-byte), and discarding rather than flushing a marker
  that completes exactly at termination.
- Non-text content parts on tool messages are now rejected (ported from upstream `79c292bc`).
- A duplicated `handle_props` definition left over from a webui/artifact merge was removed.

## Testing added

`test_admission_policy.cpp`, `test_arena.cpp` (host-arena fragmented-span behavior),
`test_kv_ram_cache*.cpp` (base, large, opt, perf variants — preflight/capture size agreement,
dynamic-boundary preservation, `PreserveExisting` behavior under pressure, the hyperquant
residual-slot footprint), `test_engine_ram_real.cpp` under both `qwen3_6_27b` and
`qwen3_6_35b_a3b` targets, `test_mtp_adaptive.cpp`, frontend marker tests, and a dedicated
MTP-RAM-vs-VRAM restore comparison.

## Local tooling

A local `.bat`-based CLI wrapper with memory support (`aa8b5dd7`) — environment tooling, not engine
code.

## Known open items

- The rewrite-checkpoint host-RAM corruption (above) is mitigated, not cured — the packed
  checkpoint payload itself still needs the same treatment the hyperquant side store got.
- `ninfer_tool_call_parser_test`'s suffix test still needs a product decision on salvage vs.
  all-or-nothing as the intended contract (currently aligned to salvage).
- `ninfer_qwen3_6_27b_load_plan_test` fails to link on a pre-existing `cudart_static`/`cudart`
  symbol conflict, unrelated to this work.
- `DeviceBuffer::fill()`'s default-stream `cudaMemset` racing against pageable `cudaMemcpy` seeding
  is fixed test-side only; other suites likely carry the same latent race.
