#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_snapshot.h"
#include <ninfer/targets/qwen3_6/mtp_adaptive_cost.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
//
// The plan types declare their moves/move-assignments here and define them per variant in
// api_impl.h with explicit bodies: any implicit definition needs the complete detail impl types
// (only the exact target TUs have them), and MSVC 19.44 does not emit out-of-line `= default`
// explicit specializations of these moves (LNK2019 at the final Windows link).
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

    // Records a shared-prefix boundary chosen by admission's boundary-capture policy
    // (runtime::choose_boundary_capture) for this plan's prefill to land a chunk on and snapshot.
    // Silently ignored if frontier does not lie strictly between this plan's own reuse_base and
    // prompt_tokens -- the plan that wins admission can have a different reuse_base than the one
    // the boundary was scored against. Bumps summary().service_work_quanta by one prefill unit to
    // account for the extra chunk split this adds to the prefill.
    void set_shared_capture_boundary(std::uint32_t frontier) noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] RequestPlan<Variant> plan_ram_reuse(const PreparedPrompt& prompt,
                                                      const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t lane) const noexcept;
    // Class of the request that last occupied the lane, for lanes that still hold retained
    // state; RequestClass::Agents (the default) for every other lane.
    [[nodiscard]] RequestClass retained_owner_class(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool capture_retained_lane(std::uint32_t lane);
    [[nodiscard]] bool capture_active_lane_for_siblings(std::uint32_t lane);
    [[nodiscard]] std::uint32_t active_lane_sibling_base(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::span<const TokenId> active_lane_tokens(std::uint32_t lane) const noexcept;
    void restore_ram_entry(std::uint32_t lane, std::uint64_t entry_id,
                           const RequestPlan<Variant>& plan);
    void claim_ram_entry(std::uint64_t entry_id);
    void release_ram_entry(std::uint64_t entry_id);
    void consume_ram_entry(std::uint64_t entry_id);
    [[nodiscard]] qwen3_6::detail::KvRamSnapshot kv_ram_snapshot() const noexcept;
    qwen3_6::detail::KvRamCopySeconds harvest_kv_ram_copy_seconds();
    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
