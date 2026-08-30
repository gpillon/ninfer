#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

enum class BackfillClass : std::uint8_t {
    None,
    Persistent,
    Temporal,
};

struct ActiveAdmissionSnapshot {
    std::uint64_t request_id = 0;
    AdmissionResources resources;
    std::uint64_t remaining_work_quanta = 0;
    std::uint64_t backfill_epoch        = 0;
    BackfillClass backfill_class        = BackfillClass::None;
};

enum class ProtectionPhase : std::uint8_t {
    Open,
    Drain,
};

struct AdmissionProtection {
    std::uint64_t epoch_id        = 0;
    std::uint64_t head_request_id = 0;
    AdmissionResources head_resources;
    std::array<std::uint64_t, kMaximumConcurrency> incumbent_ids{};
    std::array<std::uint64_t, kMaximumConcurrency> donor_ids{};
    std::size_t incumbent_count   = 0;
    std::size_t donor_count       = 0;
    std::uint64_t temporal_credit = 0;
    ProtectionPhase phase         = ProtectionPhase::Open;
};

// Scheduler-owned value policy for an idle lane's retained state. A reservation is temporal:
// it exists only while an earlier queued Main request has exact reusable state on that lane.
// Reserved lanes are not victims for later requests; all other retained state remains reclaimable.
struct RetainedLaneCandidate {
    std::uint32_t lane             = 0;
    RequestClass owner             = RequestClass::Agents;
    std::uint64_t use_tick         = 0;
    bool reserved_for_earlier_main = false;
};

// Lowest-value eligible retained state: Classifier, then Agents, then Main; LRU within a class.
[[nodiscard]] bool retained_lane_is_better_victim(const RetainedLaneCandidate& candidate,
                                                  const RetainedLaneCandidate& incumbent) noexcept;

[[nodiscard]] std::optional<std::uint32_t>
choose_retained_lane_victim(std::span<const RetainedLaneCandidate> candidates) noexcept;

[[nodiscard]] bool admission_resources_fit(const AdmissionResources& used,
                                           const AdmissionResources& capacity) noexcept;

// Freezes the currently active requests and selects the earliest projected completion prefix
// whose release makes the protected head componentwise feasible.
[[nodiscard]] AdmissionProtection make_admission_protection(
    std::uint64_t epoch_id, std::uint64_t head_request_id, const AdmissionResources& head_resources,
    std::span<const ActiveAdmissionSnapshot> active, const AdmissionResources& capacity);

// Tests the cumulative future-frontier invariant, including every still-active persistent
// backfill from this epoch and the proposed candidate.
[[nodiscard]] bool persistent_backfill_is_safe(const AdmissionProtection& protection,
                                               std::span<const ActiveAdmissionSnapshot> active,
                                               const AdmissionResources& candidate,
                                               const AdmissionResources& capacity) noexcept;

// Projected distance to the last still-active frozen donor. Later admissions never contribute.
[[nodiscard]] std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept;

// True once the head would fit if current-epoch temporal borrowers were absent. This recognizes
// both the frozen donor frontier and an earlier opportunity created by any incumbent release.
[[nodiscard]] bool
protected_head_safe_without_temporal(const AdmissionProtection& protection,
                                     std::span<const ActiveAdmissionSnapshot> active,
                                     const AdmissionResources& capacity) noexcept;

// A candidate host-RAM boundary capture: either a checkpoint already resident on an actively
// decoding lane (ActiveLaneCheckpoint, `lane` set) or a frontier partway through the admitted
// request's own prefill that other pending requests also share (SourcePrefillBoundary).
enum class BoundaryCaptureKind : std::uint8_t {
    None,
    ActiveLaneCheckpoint,
    SourcePrefillBoundary,
};

// One pending request's leading-token overlap with a candidate boundary's source. `common_tokens`
// is the exact-match length (an LCP, not an estimate); `already_reused` is the reuse this consumer
// would get anyway from a plan already in hand (e.g. an existing host-RAM match), so the boundary
// only gets credit for tokens it would newly save.
struct BoundaryConsumer {
    std::uint32_t common_tokens  = 0;
    std::uint32_t already_reused = 0;
};

struct BoundaryCandidate {
    BoundaryCaptureKind kind    = BoundaryCaptureKind::None;
    std::uint32_t lane          = 0;
    std::uint32_t frontier      = 0;
    std::size_t consumer_begin  = 0;
    std::size_t consumer_count  = 0;
};

// Fixed and per-token host-RAM cost of one capture, and the free budget it must fit inside without
// forcing an eviction. `minimum_frontier` is the noise floor below which a capture is not worth its
// fixed cost regardless of how many consumers share it.
struct BoundaryCaptureBudget {
    std::uint64_t record_fixed_bytes     = 0;
    std::uint64_t record_bytes_per_token = 0;
    std::uint64_t ram_free_bytes         = 0;
    std::uint32_t minimum_frontier       = 2048;
};

// Selects at most one boundary to capture this admission event, maximizing aggregate tokens saved
// across consumers rather than any single consumer's LCP: score(B) = sum over consumers whose
// common_tokens >= B of max(0, B - already_reused). A shorter, cheaper boundary shared by many
// consumers can outscore a longer one that only helps its single best match. Ties prefer the
// smaller (cheaper) frontier. Candidates below minimum_frontier or whose fixed+variable cost would
// not fit ram_free_bytes are rejected outright, never forcing an eviction for a speculative
// capture. Returns nullopt if no candidate has positive score.
[[nodiscard]] std::optional<BoundaryCandidate> choose_boundary_capture(
    std::span<const BoundaryCandidate> candidates, std::span<const BoundaryConsumer> consumers,
    const BoundaryCaptureBudget& budget) noexcept;

// Converts a source/consumer LCP into the latest boundary the source request can actually
// capture. A prefill boundary must remain strictly before the source prompt's final token.
[[nodiscard]] std::optional<std::uint32_t>
source_prefill_capture_frontier(std::uint32_t common_tokens, std::uint32_t source_prompt_tokens,
                                std::uint32_t minimum_frontier) noexcept;

} // namespace ninfer::runtime
