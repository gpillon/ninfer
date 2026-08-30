#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/prepared_prompt.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          admission_capacity_(instance.program->admission_capacity()) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

private:
    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }
        const auto ram            = instance_.program->kv_ram_snapshot();
        snapshot.kv_ram_captures      = ram.captures;
        snapshot.kv_ram_restores      = ram.restores;
        snapshot.kv_ram_evictions     = ram.evictions;
        snapshot.kv_ram_drops         = ram.drops;
        snapshot.kv_ram_save_seconds  = ram.save_seconds;
        snapshot.kv_ram_load_seconds  = ram.load_seconds;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time)
            : id(request_identity), prompt(std::move(input)), output(std::move(output_session)),
              prompt_summary(summary), prepare_seconds(frontend_seconds),
              options(std::move(request_options)), deadline(limit), submitted(submit_time) {}

        const std::uint64_t id;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        std::optional<Plan> ram_plan;
        std::uint64_t ram_index_version = 0;
        // Per lane, the last sibling-snapshot base attempted for this request and the RAM index
        // version observed after that attempt. Capacity pressure is retryable only after another
        // cache-state change; invariant/CUDA failures propagate.
        std::array<std::uint32_t, kMaximumConcurrency> sibling_capture_bases{};
        std::array<std::uint64_t, kMaximumConcurrency> sibling_capture_versions{};
        // Set by maybe_capture_boundary when the winning candidate this tick is a
        // SourcePrefillBoundary: a frontier partway through this request's own prefill that other
        // pending requests share, chosen for aggregate benefit rather than any single LCP. Applied
        // to the winning plan in admit_planned_request via RequestPlan::set_shared_capture_boundary.
        // Overwritten (not accumulated) on every find_admission_lane call for this request, since
        // the pending queue -- and therefore the best boundary -- can change between attempts.
        std::optional<std::uint32_t> boundary_capture_frontier;
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;
        double kv_ram_save_seconds           = 0;
        double kv_ram_load_seconds           = 0;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane          = 0;
        bool evict_retained         = false;
        std::uint64_t ram_entry_id  = 0;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
        request->ram_plan.reset();
        request->ram_index_version = 0;
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens   = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path      = request->begin->prefix_reuse_path;
            result.prefix_reuse_source    = request->begin->prefix_reuse_source;
        }
        result.kv_ram_save_seconds = request->kv_ram_save_seconds;
        result.kv_ram_load_seconds = request->kv_ram_load_seconds;
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    // Per-request failure boundary. Device-level failures never travel as exceptions
    // (cuda_check aborts the process), so a host exception escaping one request's forward or
    // publication work is request-scoped by construction: fail that request, discard its lane,
    // and keep serving every other request. Only exceptions from shared-engine work (round
    // membership, the batched decode execution itself) reach fail_all.
    void fail_active_request(std::uint32_t lane, const std::shared_ptr<Request>& request,
                             std::exception_ptr error) noexcept {
        instance_.program->abort_lane(lane);
        complete_error(request, std::move(error));
        remove_completed_slot(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        consume_service_work(request, 1);
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            instance_.program->abort_lane(lane);
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        try {
            const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, step, cancel_at_boundary);
        } catch (...) {
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            fail_active_request(lane, request, std::current_exception());
            return;
        }
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    void ensure_ram_candidate(const std::shared_ptr<Request>& request) {
        if (!request->options.execution.allow_prefix_reuse) {
            request->ram_plan.reset();
            return;
        }
        const std::uint64_t version = instance_.program->kv_ram_index_version();
        if (request->ram_index_version == version) { return; }
        request->ram_plan.reset();
        Plan plan =
            instance_.program->plan_ram_reuse(request->prompt, *request->base_plan);
        if (plan.summary().reusable_prompt_tokens > 0 &&
            plan.summary().ram_entry_id != 0 &&
            plan.summary().reuse_source == PrefixReuseSource::HostRam) {
            request->ram_plan.emplace(std::move(plan));
        }
        request->ram_index_version = version;
    }

    // A retained Main lane is a last-resort victim in ordinary traffic. While the protected FIFO
    // head is itself Main, its best exact retained match is temporarily reserved from backfill;
    // the reservation disappears with that head/protection epoch, so it cannot become a permanent
    // pin or alter the no-preemption admission contract.
    [[nodiscard]] bool defers_to_main_lane(std::uint32_t lane,
                                           RequestClass incoming) const noexcept {
        return incoming != RequestClass::Main && instance_.program->has_retained_lane(lane) &&
               instance_.program->retained_owner_class(lane) == RequestClass::Main;
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency>
    protected_main_lane_reservations(const std::shared_ptr<Request>& incoming) {
        std::array<bool, kMaximumConcurrency> reserved{};
        if (incoming->options.execution.request_class == RequestClass::Main || !protection_) {
            return reserved;
        }
        const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
        if (queued.empty() || queued.front()->id != protection_->head_request_id ||
            queued.front()->options.execution.request_class != RequestClass::Main) {
            return reserved;
        }

        const std::shared_ptr<Request>& main = queued.front();
        std::optional<std::uint32_t> best_lane;
        std::uint32_t best_reuse = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr || !instance_.program->has_retained_lane(lane) ||
                instance_.program->retained_owner_class(lane) != RequestClass::Main) {
                continue;
            }
            ensure_lane_plan(main, lane);
            const std::uint32_t reuse = main->lane_plans[lane]->summary().reusable_prompt_tokens;
            if (reuse > best_reuse) {
                best_lane  = lane;
                best_reuse = reuse;
            }
        }
        if (best_lane) { reserved[*best_lane] = true; }
        return reserved;
    }

    [[nodiscard]] RetainedLaneCandidate retained_lane_candidate(
        std::uint32_t lane,
        const std::array<bool, kMaximumConcurrency>& reservations) const noexcept {
        return RetainedLaneCandidate{
            .lane                      = lane,
            .owner                     = instance_.program->retained_owner_class(lane),
            .use_tick                  = instance_.program->retained_use_tick(lane),
            .reserved_for_earlier_main = reservations[lane],
        };
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        ensure_ram_candidate(request);
        if (maybe_capture_boundary(request)) {
            // The capture bumped the KVRamCache index version, so this re-plans against the
            // just-published sibling snapshot instead of reusing the cached (pre-capture) plan.
            ensure_ram_candidate(request);
        }
        const Plan* ram_plan = request->ram_plan ? &*request->ram_plan : nullptr;
        const std::uint32_t ram_reuse =
            ram_plan != nullptr ? ram_plan->summary().reusable_prompt_tokens : 0U;
        const std::uint64_t ram_entry_id =
            ram_plan != nullptr ? ram_plan->summary().ram_entry_id : 0ULL;

        const RequestClass incoming_class = request->options.execution.request_class;
        const auto main_reservations       = protected_main_lane_reservations(request);

        auto first_ram_lane = [&](bool after_eviction) -> std::optional<std::uint32_t> {
            std::array<RetainedLaneCandidate, kMaximumConcurrency> dirty{};
            std::size_t dirty_count = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                const bool feasible =
                    after_eviction
                        ? instance_.program->can_admit_lane_after_retained_eviction(lane, *ram_plan)
                        : instance_.program->can_admit_lane(lane, *ram_plan);
                if (!feasible) { continue; }
                if (!instance_.program->has_retained_lane(lane)) { return lane; }
                dirty[dirty_count++] = retained_lane_candidate(lane, main_reservations);
            }
            return choose_retained_lane_victim(
                std::span<const RetainedLaneCandidate>(dirty.data(), dirty_count));
        };

        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        bool selected_dirty          = false;
        bool selected_defers         = false;
        auto consider_vram           = [&](std::uint32_t lane, std::uint32_t reuse, bool evict) {
            if (main_reservations[lane]) { return; }
            const bool dirty  = instance_.program->has_retained_lane(lane);
            const bool defers = defers_to_main_lane(lane, incoming_class);
            // Deference outranks reuse: a lane that only reuses is cheaper to give up than a
            // 96k-token conversation is to rebuild. Within one deference level the ordering is
            // unchanged -- most reuse, then a clean lane over a retained one, then oldest use.
            if (selected && defers != selected_defers) {
                if (defers) { return; }
            } else if (selected) {
                if (reuse < selected_reuse) { return; }
                if (reuse == selected_reuse) {
                    if (!selected_dirty) { return; }
                    if (dirty) {
                        const RetainedLaneCandidate candidate =
                            retained_lane_candidate(lane, main_reservations);
                        const RetainedLaneCandidate incumbent =
                            retained_lane_candidate(selected->lane, main_reservations);
                        if (!retained_lane_is_better_victim(candidate, incumbent)) { return; }
                    }
                }
            }
            selected        = LaneChoice{.lane = lane, .evict_retained = evict, .ram_entry_id = 0};
            selected_reuse  = reuse;
            selected_dirty  = dirty;
            selected_defers = defers;
        };
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane(lane, plan)) {
                consider_vram(lane, reuse, false);
            }
        }
        if (ram_plan != nullptr && ram_reuse > 0) {
            if (const std::optional<std::uint32_t> ram_lane = first_ram_lane(false); ram_lane) {
                const bool defers = defers_to_main_lane(*ram_lane, incoming_class);
                if (!selected || (selected_defers && !defers) ||
                    (defers == selected_defers && ram_reuse > selected_reuse)) {
                    selected = LaneChoice{.lane           = *ram_lane,
                                          .evict_retained = false,
                                          .ram_entry_id   = ram_entry_id};
                    selected_reuse  = ram_reuse;
                    selected_dirty  = instance_.program->has_retained_lane(*ram_lane);
                    selected_defers = defers;
                }
            }
        }
        if (selected) { return selected; }
        //
        // No lane can admit directly. Retry allowing retained eviction.
        //
        selected.reset();
        selected_reuse  = 0;
        selected_dirty  = false;
        selected_defers = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan)) {
                consider_vram(lane, reuse, true);
            }
        }
        if (ram_plan != nullptr && ram_reuse > 0) {
            if (const std::optional<std::uint32_t> ram_lane = first_ram_lane(true); ram_lane) {
                const bool defers = defers_to_main_lane(*ram_lane, incoming_class);
                if (!selected || (selected_defers && !defers) ||
                    (defers == selected_defers && ram_reuse > selected_reuse)) {
                    selected = LaneChoice{.lane           = *ram_lane,
                                          .evict_retained = true,
                                          .ram_entry_id   = ram_entry_id};
                }
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        const bool ram_hit       = choice.ram_entry_id != 0;
        if (ram_hit) {
            if (!request->ram_plan ||
                request->ram_plan->summary().ram_entry_id != choice.ram_entry_id) {
                throw std::logic_error("selected RAM admission has no matching request plan");
            }
        } else if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        Plan& winning_plan = ram_hit ? *request->ram_plan : *request->lane_plans[lane];

        bool ram_claimed   = false;
        bool ram_consumed  = false;
        bool target_started = false;
        if (ram_hit) {
            instance_.program->claim_ram_entry(choice.ram_entry_id);
            ram_claimed = true;
        }

        auto release_ram_if_needed = [&]() {
            if (ram_claimed && !ram_consumed) {
                instance_.program->release_ram_entry(choice.ram_entry_id);
                ram_claimed = false;
            }
        };

        try {
            if (choice.evict_retained) {
                const auto main_reservations = protected_main_lane_reservations(request);
                while (!instance_.program->can_admit_lane(lane, winning_plan)) {
                    std::array<RetainedLaneCandidate, kMaximumConcurrency> candidates{};
                    std::size_t candidate_count = 0;
                    for (std::uint32_t retained_lane = 0; retained_lane < max_concurrency_;
                         ++retained_lane) {
                        if (retained_lane == lane || slots_[retained_lane] != nullptr ||
                            !instance_.program->has_retained_lane(retained_lane)) {
                            continue;
                        }
                        candidates[candidate_count++] =
                            retained_lane_candidate(retained_lane, main_reservations);
                    }
                    const std::optional<std::uint32_t> victim = choose_retained_lane_victim(
                        std::span<const RetainedLaneCandidate>(candidates.data(), candidate_count));
                    if (!victim) {
                        // can_admit_lane_after_retained_eviction is target-level and deliberately
                        // knows nothing about scheduler reservations. If feasibility depended on
                        // the protected Main match, leave this backfill queued and let the Main
                        // head consume its state when the frozen resource frontier matures.
                        release_ram_if_needed();
                        publish_runtime_stats();
                        return AdmissionProgress::None;
                    }
                    (void)instance_.program->capture_retained_lane(*victim);
                    instance_.program->evict_retained_lane(*victim);
                    invalidate_lane_plans(*victim);
                }
            }

            if (!ram_hit && winning_plan.summary().reusable_prompt_tokens == 0 &&
                instance_.program->has_retained_lane(lane)) {
                (void)instance_.program->capture_retained_lane(lane);
            }

            Plan selected_plan = std::move(winning_plan);
            if (ram_hit) {
                request->ram_plan.reset();
            } else {
                request->lane_plans[lane].reset();
            }
            if (!erase_pending(request)) {
                const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
                request->kv_ram_save_seconds += copies.save;
                request->kv_ram_load_seconds += copies.load;
                release_ram_if_needed();
                return AdmissionProgress::None;
            }
            release_planning_state(request);

            // Bake in the dynamic boundary chosen (if any) by maybe_capture_boundary against the
            // plan that actually won admission -- it may have a different reuse_base than the one
            // the boundary was scored against, so this validates internally and is a silent no-op
            // if the frontier no longer lies strictly inside the reusable suffix.
            if (request->boundary_capture_frontier) {
                selected_plan.set_shared_capture_boundary(*request->boundary_capture_frontier);
            }

            const RequestPlanSummary summary = selected_plan.summary();
            if (backfill_class == BackfillClass::Temporal) {
                if (!protection_ || protection_->epoch_id != backfill_epoch ||
                    summary.service_work_quanta > protection_->temporal_credit) {
                    throw std::logic_error("temporal backfill lost its protected credit");
                }
                protection_->temporal_credit -= summary.service_work_quanta;
            }
            clear_protection_if_head(request);

            const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            if (ram_hit) {
                instance_.program->restore_ram_entry(lane, choice.ram_entry_id, selected_plan);
            }
            target_started = true;
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient);
            const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds = copies.save;
            request->kv_ram_load_seconds = copies.load;
            if (ram_hit) {
                instance_.program->consume_ram_entry(choice.ram_entry_id);
                ram_consumed = true;
            }
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
            return AdmissionProgress::RanGpuUnit;
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            const auto copies              = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds += copies.save;
            request->kv_ram_load_seconds += copies.load;
            release_ram_if_needed();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            // The request is failed and its lane state discarded; rethrowing here would
            // fail_all the engine over one request's host-side error (the req-3 class).
            return AdmissionProgress::ControlProgress;
        }
    }

    // Exact leading-token match length between two token sequences (an LCP), truncated to the
    // shorter of the two.
    [[nodiscard]] static std::uint32_t
    common_prefix_length(std::span<const TokenId> a, std::span<const TokenId> b) noexcept {
        const auto [mismatch_a, mismatch_b] = std::mismatch(a.begin(), a.end(), b.begin(), b.end());
        return static_cast<std::uint32_t>(mismatch_a - a.begin());
    }

    // Single decision point for every host-RAM boundary capture triggered at admission: both a
    // checkpoint already resident on an actively-decoding lane (ActiveLaneCheckpoint -- the case
    // the former maybe_capture_sibling_source handled alone) and an LCP-driven frontier partway
    // through `request`'s own prefill that other pending requests also share
    // (SourcePrefillBoundary). Evaluating both families together and picking at most one by
    // aggregate benefit (runtime::choose_boundary_capture) avoids the two ever producing
    // overlapping records for the same source lane, and picks the candidate that helps the most
    // pending requests rather than whichever lane happens to come first by index.
    //
    // ActiveLaneCheckpoint is captured immediately (same D2H cost and race-free reasoning as
    // before: the sibling is by definition present at its own admission, and the source lane's
    // state is directly readable on this same worker thread). SourcePrefillBoundary is not
    // captured here -- it only sets request->boundary_capture_frontier, which
    // admit_planned_request later hands to the winning plan via
    // RequestPlan::set_shared_capture_boundary; the Program lands a chunk on it and captures
    // during this request's own prefill.
    //
    // Returns true only when an ActiveLaneCheckpoint capture was published (the caller then
    // re-plans against the bumped KVRamCache index); a SourcePrefillBoundary decision returns
    // false since nothing was captured yet.
    bool maybe_capture_boundary(const std::shared_ptr<Request>& request) {
        // Below this many shared tokens a capture's ~183 MiB fixed cost (GDN recurrent + conv +
        // hyperquant side store, see kv_ram_cache.h) is not worth it regardless of consumer count;
        // matches the previous maybe_capture_sibling_source threshold.
        constexpr std::uint32_t kBoundaryMinimumFrontier = 2048;
        constexpr std::uint64_t kBoundaryRecordFixedBytes = 183ULL * 1024 * 1024;
        // hq-e8-2b target + MTP backend side store, ~9216 B/token (see plan doc derivation from
        // qwen3_6_27b/impl/config.h and PagedKVCache row layout).
        constexpr std::uint64_t kBoundaryRecordBytesPerToken = 9216;

        request->boundary_capture_frontier.reset();
        if (!request->options.execution.allow_prefix_reuse) { return false; }

        const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
        const targets::qwen3_6::PreparedPromptData& admitted_data =
            targets::qwen3_6::PreparedPromptAccess::view(request->prompt);

        std::vector<runtime::BoundaryCandidate> candidates;
        std::vector<runtime::BoundaryConsumer> consumers;

        // Family A: a checkpoint already resident on an actively-decoding lane.
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            const std::uint32_t base = instance_.program->active_lane_sibling_base(lane);
            if (base < kBoundaryMinimumFrontier) { continue; }
            const std::uint64_t capture_version = instance_.program->kv_ram_index_version();
            if (request->sibling_capture_bases[lane] == base &&
                request->sibling_capture_versions[lane] == capture_version) {
                // Already attempted at this exact base against the current cache state -- retrying
                // would repeat a capacity-pressure failure for free (a real invariant failure
                // propagates instead of landing here).
                continue;
            }
            const std::span<const TokenId> lane_tokens = instance_.program->active_lane_tokens(lane);
            if (lane_tokens.size() < base) { continue; }
            const std::span<const TokenId> lane_prefix = lane_tokens.subspan(0, base);

            const std::size_t consumer_begin = consumers.size();
            consumers.push_back(runtime::BoundaryConsumer{
                .common_tokens =
                    common_prefix_length(lane_prefix, std::span<const TokenId>(admitted_data.token_ids)),
                .already_reused = request->ram_plan
                                      ? request->ram_plan->summary().reusable_prompt_tokens
                                      : 0U,
            });
            for (const std::shared_ptr<Request>& other : queued) {
                if (other->id == request->id) { continue; }
                const targets::qwen3_6::PreparedPromptData& other_data =
                    targets::qwen3_6::PreparedPromptAccess::view(other->prompt);
                consumers.push_back(runtime::BoundaryConsumer{
                    .common_tokens = common_prefix_length(
                        lane_prefix, std::span<const TokenId>(other_data.token_ids)),
                    .already_reused = other->ram_plan
                                          ? other->ram_plan->summary().reusable_prompt_tokens
                                          : 0U,
                });
            }
            candidates.push_back(runtime::BoundaryCandidate{
                .kind           = runtime::BoundaryCaptureKind::ActiveLaneCheckpoint,
                .lane           = lane,
                .frontier       = base,
                .consumer_begin = consumer_begin,
                .consumer_count = consumers.size() - consumer_begin,
            });
        }

        // Family B: an LCP frontier partway through `request`'s own prefill that other pending
        // requests share. The source gains nothing here -- it only pays the split + D2H -- so it
        // is never one of this family's consumers.
        {
            const std::size_t consumer_begin = consumers.size();
            std::vector<std::uint32_t> distinct_lcps;
            const std::uint32_t source_prompt_tokens =
                static_cast<std::uint32_t>(admitted_data.token_ids.size());
            for (const std::shared_ptr<Request>& other : queued) {
                if (other->id == request->id) { continue; }
                const targets::qwen3_6::PreparedPromptData& other_data =
                    targets::qwen3_6::PreparedPromptAccess::view(other->prompt);
                const std::uint32_t common = common_prefix_length(
                    std::span<const TokenId>(admitted_data.token_ids),
                    std::span<const TokenId>(other_data.token_ids));
                consumers.push_back(runtime::BoundaryConsumer{
                    .common_tokens = common,
                    .already_reused = other->ram_plan
                                          ? other->ram_plan->summary().reusable_prompt_tokens
                                          : 0U,
                });
                if (const auto boundary = runtime::source_prefill_capture_frontier(
                        common, source_prompt_tokens, kBoundaryMinimumFrontier)) {
                    distinct_lcps.push_back(*boundary);
                }
            }
            const std::size_t consumer_count = consumers.size() - consumer_begin;
            std::sort(distinct_lcps.begin(), distinct_lcps.end());
            distinct_lcps.erase(std::unique(distinct_lcps.begin(), distinct_lcps.end()),
                               distinct_lcps.end());
            for (std::uint32_t lcp : distinct_lcps) {
                candidates.push_back(runtime::BoundaryCandidate{
                    .kind           = runtime::BoundaryCaptureKind::SourcePrefillBoundary,
                    .lane           = 0,
                    .frontier       = lcp,
                    .consumer_begin = consumer_begin,
                    .consumer_count = consumer_count,
                });
            }
        }

        if (candidates.empty()) { return false; }

        const auto ram = instance_.program->kv_ram_snapshot();
        const runtime::BoundaryCaptureBudget budget{
            .record_fixed_bytes     = kBoundaryRecordFixedBytes,
            .record_bytes_per_token = kBoundaryRecordBytesPerToken,
            .ram_free_bytes = ram.capacity_bytes > ram.used_bytes ? ram.capacity_bytes - ram.used_bytes
                                                                  : 0ULL,
            .minimum_frontier = kBoundaryMinimumFrontier,
        };
        const std::optional<runtime::BoundaryCandidate> chosen =
            runtime::choose_boundary_capture(candidates, consumers, budget);
        if (!chosen) { return false; }

        if (chosen->kind == runtime::BoundaryCaptureKind::ActiveLaneCheckpoint) {
            const bool captured = instance_.program->capture_active_lane_for_siblings(chosen->lane);
            request->sibling_capture_bases[chosen->lane]    = chosen->frontier;
            request->sibling_capture_versions[chosen->lane] =
                instance_.program->kv_ram_index_version();
            return captured;
        }
        request->boundary_capture_frontier = chosen->frontier;
        return false;
    }

    // Measurement-only: records how much leading-token prefix `admitted` shares with the other
    // still-pending requests in `queued`, into cumulative_stats_.sibling_prefix_*. Never changes
    // admission or reuse behavior -- purely observational, so real production traces can show the
    // actual common-prefix distribution across concurrent-sibling bursts before any reuse
    // mechanism is built on top of it.
    void record_sibling_prefix_sample(const Request& admitted,
                                      const std::vector<std::shared_ptr<Request>>& queued,
                                      std::uint32_t selected_reuse) {
        constexpr std::size_t kNoiseFloorTokens = 64;
        const targets::qwen3_6::PreparedPromptData& admitted_data =
            targets::qwen3_6::PreparedPromptAccess::view(admitted.prompt);
        std::size_t best = 0;
        RequestClass best_source = RequestClass::Agents;
        for (const std::shared_ptr<Request>& other : queued) {
            if (other->id == admitted.id) { continue; }
            const targets::qwen3_6::PreparedPromptData& other_data =
                targets::qwen3_6::PreparedPromptAccess::view(other->prompt);
            const auto [mismatch_a, mismatch_b] =
                std::mismatch(admitted_data.token_ids.begin(), admitted_data.token_ids.end(),
                             other_data.token_ids.begin(), other_data.token_ids.end());
            const std::size_t common =
                static_cast<std::size_t>(mismatch_a - admitted_data.token_ids.begin());
            if (common > best) {
                best        = common;
                best_source = other->options.execution.request_class;
            }
        }
        if (best < kNoiseFloorTokens) { return; }
        ++cumulative_stats_.sibling_prefix_samples;
        cumulative_stats_.sibling_prefix_common_tokens_sum += best;
        cumulative_stats_.sibling_prefix_common_tokens_max =
            std::max(cumulative_stats_.sibling_prefix_common_tokens_max,
                     static_cast<std::uint64_t>(best));
        const std::size_t source = static_cast<std::size_t>(best_source);
        const std::size_t target =
            static_cast<std::size_t>(admitted.options.execution.request_class);
        SiblingPrefixClassStats& by_class =
            cumulative_stats_.sibling_prefix_by_class[source * kRequestClassCount + target];
        ++by_class.samples;
        by_class.common_tokens_sum += best;
        by_class.common_tokens_max =
            std::max(by_class.common_tokens_max, static_cast<std::uint64_t>(best));
        if (best > selected_reuse) { by_class.selected_reuse_gap_sum += best - selected_reuse; }
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                const RequestPlanSummary& selected =
                    head_lane->ram_entry_id != 0
                        ? head->ram_plan->summary()
                        : head->lane_plans[head_lane->lane]->summary();
                record_sibling_prefix_sample(*head, queued, selected.reusable_prompt_tokens);
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate_lane->ram_entry_id != 0 ? candidate->ram_plan->summary()
                                                      : candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    record_sibling_prefix_sample(*candidate, queued,
                                                 candidate_plan.reusable_prompt_tokens);
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<std::exception_ptr, kMaximumConcurrency> row_errors{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            OutputDecision decision;
            try {
                decision = request->output.preview(row_tokens, request->budget->remaining(),
                                                   request->budget->limit_reason());
            } catch (...) {
                // Request-scoped output-policy failure: hand the row to the batch resolve as
                // cancelled (discarding its licensed tokens) and fail the request below.
                row_errors[row]     = std::current_exception();
                cancelled[row]      = 1;
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count)) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()));

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (row_errors[row]) {
                fail_active_request(lane, request, std::move(row_errors[row]));
                continue;
            }
            try {
                if (!cancelled[row]) {
                    const auto row_tokens = round.tokens.subspan(
                        row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                    request->generated.insert(request->generated.end(), row_tokens.begin(),
                                              row_tokens.end());
                    request->budget->commit(accepted[row]);
                    consume_service_work(request, accepted[row]);
                }
                auto published = request->output.commit_preview();
                if (!request->first_token && accepted[row] != 0) {
                    request->first_token = Clock::now();
                }
                append_output(request, std::move(published));
                if (terminal[row]) {
                    complete_success(request, finish_reasons[row]);
                    remove_completed_slot(lane);
                }
            } catch (...) {
                fail_active_request(lane, request, std::current_exception());
                cancelled[row] = 1; // keep the row out of the committed-token stats below
            }
        }
        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        publish_runtime_stats();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                if (prefill_lane_) {
                    if (!membership.empty() && !previous_unit_was_decode) {
                        run_decode_round(membership);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_decode_round(membership);
                    previous_unit_was_decode = true;
                    continue;
                }
            } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const AdmissionResources admission_capacity_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
