# VRAM retained-lane LRU

When a FullReset (or equal-reuse dirty cover) must destroy a retained VRAM
bundle, pick the least-recently-admitted free dirty lane instead of the
lowest lane index.

## Policy

Unchanged:

- Longer `reusable_prompt_tokens` wins.
- Equal reuse: empty lane beats dirty.
- RAM wins only if strictly longer than the VRAM winner.
- In-flight slots (`slots_[lane] != nullptr`) are never victims.
- Host RAM stays exclusive FIFO (capture appends tail; consume erases on
  VRAM load; occupancy is live host residents).

Changed: among equal-reuse **dirty free** lanes, cover the LRU retained
chat. Same recency pick for:

1. `consider_vram` dirty tie (FullReset cover / site 2).
2. `first_ram_lane` when only dirty lanes are feasible (RAM restore cover /
   site 3).
3. `evict_retained` page-reclaim loop: LRU-first among **other** free
   retained lanes (`!= selected`), never the selected lane.

Tie (equal `use_tick`): lowest lane index.

LRU is a tie-break inside the current admission pass. Do not prefer an
in-place-infeasible LRU lane over a feasible MRU lane. Pass 1 remains
`can_admit_lane`; pass 2 remains `can_admit_lane_after_retained_eviction`.

## Recency clock

Per-lane `use_tick++` is wrong: `clear_lane` zeros the victim while
FullReset `ordered_reset` does not, so a just-admitted FullReset can look
older than a continued chat and the next cover spills the MRU.

Required:

- `std::uint64_t use_tick = 0` on `SequenceState`.
- `std::uint64_t next_use_tick = 1` on `ProgramImplCore`.
- On successful occupy, `sequence.use_tick = next_use_tick++`.
- One bump site: end of successful `start_prefill_lane` (VRAM hit, RAM
  restore, and FullReset all go through it).
- Zero `use_tick` only in `clear_lane`, not in `ordered_reset`.
- `retained_use_tick(lane)` returns 0 if `!has_retained_lane(lane)`.
  Compare ticks only among dirty free candidates.
- Do not bump on `plan_match`, `capture_retained_lane`,
  `restore_ram_entry`, `consume_ram_entry`, or completion/`retained = true`.
- No setter on the family `Program` surface. No 35B-only twin.

## Implementation

1. SequenceState + Program clock and getter (`program.h`, `program_impl.h`,
   `runtime.h`, `api_impl.h`).
2. `consider_vram`: on equal reuse and both dirty, smaller
   `retained_use_tick`; equal ticks keep current selection.
3. `first_ram_lane`: first feasible empty; else min tick then lowest index.
4. Eviction loop: while `!can_admit_lane(selected, plan)`, capture+evict
   LRU other free retained. Site 2/3 capture of the selected lane unchanged.
5. Docs: `docs/maintainer/concurrent-inference-architecture.md` §6.5 (and
   §6.4 if it still implies index order) and
   `docs/maintainer/paged-kv-cache.md` §10.4. Host RAM remains exclusive
   FIFO.

## Tests

27B `test_engine_ram_real.cpp`, C=2, identify **which chat** is VRAM vs RAM
(not capture count alone):

| Case | Setup | Observe |
|---|---|---|
| Empty still wins | A, then B | B FullReset, captures unchanged; A still `VramResident` |
| Dirty tie = first admit | A, B, then C | C FullReset; A `HostRam`; B still `VramResident` |
| Continue refreshes recency | A, B, continue A, then C | C FullReset; B `HostRam`; A still `VramResident` |
| RAM cover LRU dirty | Both dirty, A more recent, RAM hit strictly longer | Restore covers B (site 3 captures B); A stays VRAM |

Keep existing empty-lane C=2/C=3 and `exercise_vram_wins` / longer-RAM-beats-VRAM.

Site-1 eviction order under C=2 with one other retained is identical to
index order. A C=3 “need one extra other lane” case is the only witness
for loop order; add it only if cheap with the existing pooled C=3 fixture.

## Out of scope

Host-RAM LRU, lineage-dedup, consume timing, removing serve warmup, new
CLI flags, 35B occupancy twin.
