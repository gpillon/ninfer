# VRAM retained-lane LRU

When a FullReset (or equal-reuse dirty cover) must destroy a retained VRAM
bundle, pick the least-recently-admitted free dirty lane instead of the
lowest lane index.

Working-tree draft may already exist. Reconcile it to this document; do not
treat the draft as the spec.

## Policy

Unchanged:

- Longer `reusable_prompt_tokens` wins.
- Equal reuse: empty lane beats dirty. Ticks are consulted only when both
  candidates are dirty. Do not min-tick empty vs dirty (`retained_use_tick`
  is 0 for empty).
- RAM wins only if strictly longer than the VRAM winner.
- In-flight slots (`slots_[lane] != nullptr`) are never victims. Active
  requests are never preempted and never captured for the host tier.
- If `find_admission_lane` returns no lane, the queued request waits.
  Capture and VRAM eviction run only inside `admit_planned_request` after a
  lane was selected. If active lanes already occupy enough pool pages that
  no free lane can admit even after reclaiming every other free retained
  bundle (`can_admit_lane_after_retained_eviction` false on every free
  lane), that queued request does not dump any GPU lane. A later smaller
  backfill candidate that *can* fit may still admit and may capture+evict
  free retained lanes for itself; the blocked head does not.
- Host RAM stays exclusive FIFO (capture appends tail; consume erases on
  VRAM load; occupancy is live host residents).
- `uint64` tick wrap is out of scope.

Changed: among equal-reuse **dirty free** lanes, cover the LRU retained
chat. Same recency pick for:

1. `consider_vram` dirty tie (FullReset cover / site 2).
2. `first_ram_lane` when only dirty lanes are feasible (RAM restore cover /
   site 3).
3. `evict_retained` page-reclaim loop: LRU-first among **other** free
   retained lanes (`!= selected`, `slots_ == nullptr`, `has_retained_lane`),
   never the selected lane.

Tie (equal `use_tick`): lowest lane index. That is the normal tie-break,
including everyone still at 0. It is not an error and must not warn.

LRU is a tie-break inside the current admission pass. Do not prefer an
in-place-infeasible LRU lane over a feasible MRU lane. Pass 1 remains
`can_admit_lane`; pass 2 remains `can_admit_lane_after_retained_eviction`.

Site-1 loop drops as many other free retained lanes as needed until the
selected plan fits, not all of them up front. Each dropped lane is a
separate best-effort host capture then VRAM destroy. If the image does not
fit the host budget, `drops` increments and VRAM is still released.

## Recency clock

Per-lane `use_tick++` is wrong: `clear_lane` zeros the victim while
FullReset `ordered_reset` does not, so a just-admitted FullReset can look
older than a continued chat and the next cover spills the MRU.

Required:

- `std::uint64_t use_tick = 0` on `SequenceState`.
- `std::uint64_t next_use_tick = 1` on `ProgramImplCore`.
- On successful occupy, `sequence.use_tick = next_use_tick++`.
- One bump site: after non-throwing `advance_prefill` inside
  `start_prefill_lane`, then return that result (VRAM hit, RAM restore, and
  FullReset all go through it). Do not bump before `advance_prefill`.
- Zero `use_tick` only in `clear_lane`, not in `ordered_reset`. Failed
  occupy (throw in `start_prefill_lane`) must not keep a new tick; the
  catch path already `clear_lane`s.
- `retained_use_tick(lane)` returns 0 if `!has_retained_lane(lane)`.
  Compare ticks only among dirty free candidates.
- Do not bump on `plan_match`, `capture_retained_lane`,
  `restore_ram_entry`, `consume_ram_entry`, or completion/`retained = true`.
- Do not pack `use_tick` into `RamCaptureSource` / restore. Restore may
  leave a stale tick on the target; `slots_[lane]` is already set so
  admission does not observe it. The occupy bump overwrites it.
- No setter on the family `Program` surface. No 35B-only twin.

## Eviction-loop invariant

Pass 2 true means reclaiming all other retained pages is enough for the
selected plan. The admit loop then drops other free retained LRU-first
until `can_admit_lane` is true.

If that search finds no remaining other free retained victim, throw the
existing `logic_error` (`retained eviction did not make admission
feasible`). Do not spin. Do not fall back to index order (the victim set
is empty; ranking does not matter). Do not warn-and-continue.

That throw is the same HEAD path. LRU does not add a new way to hit it.
It is an unreachable invariant if pass 2 is honest. Do not add tests that
inject a lying probe.

## Implementation

1. Clock + getter (`program.h`, `program_impl.h`, `runtime.h`,
   `api_impl.h`). Zero tick only in `clear_lane`. No RAM pack. No setter.
2. Bump after non-throwing `advance_prefill` in `start_prefill_lane`.
3. `consider_vram`: do **not** copy HEAD's
   `reuse == selected_reuse && (!selected_dirty || dirty)` then add a tick
   check after it — that skip already includes both-dirty and would make
   LRU dead. Split: empty still beats dirty (first empty wins). Only the
   both-dirty arm becomes min `retained_use_tick`; equal ticks keep
   current (lowest index).
4. `first_ram_lane`: first feasible empty; else min tick among feasible
   dirty, then lowest index (full scan).
5. Eviction: while `!can_admit_lane(selected, plan)`, pick min-tick other
   free retained; capture+evict+`invalidate_lane_plans`. If no victim,
   throw HEAD's `logic_error`. Site 2/3 capture of the selected lane
   unchanged. Never capture/evict the selected lane in this loop.
6. Docs: `docs/maintainer/concurrent-inference-architecture.md` §6.4 and
   §6.5 and `docs/maintainer/paged-kv-cache.md` §10.4. State VRAM dirty-tie
   LRU occupy tick; host remains exclusive FIFO; no eviction unless a lane
   was selected.

## Tests

27B `test_engine_ram_real.cpp`, C=2, identify **which chat** is VRAM vs RAM
(not capture count alone). Fresh Engine (not serve warmup).
`verify_ram_tier` must not dirty a lane.

| Case | Setup | Observe |
|---|---|---|
| Empty still wins | A, then B; stop | B FullReset, captures unchanged; A still `VramResident` |
| Dirty tie = first admit | A, B, then C; no continue | C FullReset; A `HostRam`; B still `VramResident` |
| Continue refreshes recency | A, B, continue A, then C | C FullReset; B `HostRam`; A still `VramResident` |
| RAM cover LRU dirty | Third lineage already on host; A MRU dirty, B LRU dirty; RAM hit strictly longer | Restore covers B (site 3 captures B); A stays VRAM |

Keep existing empty-lane C=2/C=3 and `exercise_vram_wins` /
`longer-RAM-beats-VRAM`. Do not bolt identity onto the empty-lane case
that also continues A.

Site-1 eviction order under C=2 with one other retained is identical to
index order. A C=3 “need one extra other lane” case is the only witness
for loop order; add it only if cheap with the existing pooled C=3 fixture.
Do not test the no-victim `logic_error`.

Serve warmup (`generate("hi")`) occupies a VRAM lane. Engine tests are
cold. After warmup at C=2, “A then B, no capture” is false (B covers
warmup). User A, B, then C still spills A. Do not remove warmup.

## Out of scope

Host-RAM LRU, lineage-dedup, consume timing, removing serve warmup, new
CLI flags, 35B occupancy twin, tick wrap, warning-on-invariant,
index-order fallback when the victim set is empty.
