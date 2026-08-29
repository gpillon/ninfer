#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_snapshot.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::detail {

struct RamCaptureSource {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;

    std::span<const TokenId> ledger;
    const ResidentPrefixIdentity* identity = nullptr;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool hash_c_valid = false;

    const PagedKVAllocation* text      = nullptr;
    const PagedKVPool* text_pool       = nullptr;
    const PagedKVAllocation* backend   = nullptr;
    const PagedKVPool* backend_pool    = nullptr;

    const LinearAttentionStatePool* gdn = nullptr;
    std::int32_t gdn_current_slot       = -1;
    std::int32_t gdn_checkpoint_slot    = -1;

    const Tensor* tail_hidden                = nullptr;
    const Tensor* rewrite_checkpoint_hidden  = nullptr;

    const CyclicKVCache* dflash_local      = nullptr;
    const CyclicKVCache* dflash_checkpoint = nullptr;
    std::int32_t dflash_lane               = 0;

    cudaStream_t stream = nullptr;

    // True for a speculative snapshot of a lane that is still actively serving its own request
    // (captured right as its prefill completes, before decode -- see
    // ProgramImplCore::capture_active_lane_for_siblings), taken on the chance that another
    // pending request shares enough of its leading prompt to be worth restoring from it. Unlike
    // an ordinary terminal-lane capture, more than one sibling may legitimately want to restore
    // from the same entry, so it must not be erased after the first restore.
    bool multi_claim = false;
};

struct RamRestoreTarget {
    std::uint32_t text_dst_pages    = 0;
    std::uint32_t backend_dst_pages = 0;
    PagedKVAllocation* text         = nullptr;
    PagedKVPool* text_pool          = nullptr;
    PagedKVAllocation* backend      = nullptr;
    PagedKVPool* backend_pool       = nullptr;

    LinearAttentionStatePool* gdn     = nullptr;
    std::int32_t gdn_current_slot     = -1;
    std::int32_t gdn_checkpoint_slot  = -1;

    Tensor* tail_hidden               = nullptr;
    Tensor* rewrite_checkpoint_hidden = nullptr;

    CyclicKVCache* dflash_local      = nullptr;
    CyclicKVCache* dflash_checkpoint = nullptr;
    std::int32_t dflash_lane         = 0;

    cudaStream_t stream = nullptr;
};

struct RamRestoredHost {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;
    bool backend_image_present            = false;
    std::vector<TokenId> ledger;
    ResidentPrefixIdentity identity;
};

struct RamMatch {
    std::uint64_t entry_id     = 0;
    PrefixReusePath reuse      = PrefixReusePath::FullReset;
    std::uint32_t reuse_base   = 0;
};

class KVRamCache {
public:
    explicit KVRamCache(std::size_t capacity_bytes);
    ~KVRamCache();

    KVRamCache(const KVRamCache&)            = delete;
    KVRamCache& operator=(const KVRamCache&) = delete;
    KVRamCache(KVRamCache&&)                 = delete;
    KVRamCache& operator=(KVRamCache&&)      = delete;

    [[nodiscard]] std::optional<RamMatch> plan_match(const PreparedPromptData& prompt,
                                                     std::span<const PrefixHash128> hash_chain);

    void claim(std::uint64_t entry_id);
    void release(std::uint64_t entry_id);
    void consume(std::uint64_t entry_id);

    bool capture(const RamCaptureSource& source);
    RamRestoredHost unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target);

    [[nodiscard]] RamRestoredHost load_host(std::uint64_t entry_id) const;

    [[nodiscard]] KvRamSnapshot snapshot() const noexcept;
    KvRamCopySeconds harvest_copy_seconds();
    [[nodiscard]] std::uint64_t index_version() const noexcept { return index_version_; }
    [[nodiscard]] std::uint64_t exact_comparisons() const noexcept { return exact_comparisons_; }

    void test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte);

private:
    enum class Section : std::uint8_t {
        Ledger = 0,
        Identity,
        TextKv,
        BackendKv,
        GdnConvCurrent,
        GdnConvCheckpoint,
        GdnRecurrentCurrent,
        GdnRecurrentCheckpoint,
        TailHidden,
        RewriteCheckpointHidden,
        DflashLocal,
        DflashRewriteCheckpoint,
        Count
    };

    struct Record {
        std::uint64_t id               = 0;
        PrefixHash128 hash_f{};
        PrefixHash128 hash_c{};
        bool hash_c_valid              = false;
        std::uint32_t execution_frontier = 0;
        std::uint32_t checkpoint_frontier = 0;
        bool checkpoint_valid          = false;
        PrefixReusePath checkpoint_path = PrefixReusePath::RestoreTurnCheckpoint;
        void* block                    = nullptr;
        std::size_t bytes              = 0;
        bool pinned                    = false;
        bool copies_timed              = false;
        cudaEvent_t copies_start       = nullptr;
        cudaEvent_t copies_done        = nullptr;
        // Lineage identity (hash of the leading tokens, typically the system prompt + tool
        // schema) and whether that lineage has previously produced a cache hit. Entries whose
        // lineage has demonstrated reuse are evicted only after all non-protected entries are
        // exhausted, so short one-shot captures (classifier calls, etc.) churn out first instead
        // of displacing checkpoints from conversations that keep coming back.
        std::uint64_t origin_hash      = 0;
        bool protected_tier            = false;
        // See RamCaptureSource::multi_claim. consume() releases the claim instead of erasing
        // the record for entries captured this way, so a second/third sibling can independently
        // claim/restore the same entry; it then ages out through ordinary eviction like any
        // other record, no special-cased cleanup required.
        bool multi_claim               = false;
    };

    struct Layout {
        std::size_t header_bytes                               = 0;
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> offset{};
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> length{};
        std::size_t entry_bytes                                = 0;
    };

    [[nodiscard]] Record& require(std::uint64_t entry_id);
    [[nodiscard]] const Record& require(std::uint64_t entry_id) const;
    [[nodiscard]] bool lineage_is_hot(std::uint64_t origin_hash) const;
    void note_lineage_hit(std::uint64_t origin_hash);
    void evict_unpinned();
    void destroy_record(std::uint64_t entry_id, bool count_eviction);
    void begin_copies(Record& record, cudaStream_t stream);
    void record_copies(Record& record, cudaStream_t stream);
    void wait_copies(Record& record);
    void wait_copies_on_stream(Record& record, cudaStream_t stream);
    double harvest_record(Record& record);
    void retire_record(Record& record);
    void reap_retired(bool block);
    void bump_version() noexcept { ++index_version_; }

    struct RetiredCopy {
        void* block           = nullptr;
        cudaEvent_t copies_done = nullptr;
    };

    HostPinnedArena arena_;
    std::deque<std::uint64_t> fifo_;
    std::unordered_map<std::uint64_t, Record> records_;
    // Bounded record of which content lineages (leading-token hash) have previously produced a
    // cache hit, so a new capture for the same conversation can start out protected. Capped and
    // FIFO-evicted independently of the KV entries themselves -- this is a small hint table, not
    // a source of truth.
    std::unordered_map<std::uint64_t, std::uint32_t> lineage_hits_;
    std::deque<std::uint64_t> lineage_order_;
    std::vector<RetiredCopy> retired_;
    std::vector<std::uint64_t> pending_save_ids_;
    std::optional<std::uint64_t> pending_load_id_;
    std::uint64_t next_id_           = 1;
    std::uint64_t index_version_     = 1;
    std::uint64_t captures_          = 0;
    std::uint64_t restores_          = 0;
    std::uint64_t evictions_         = 0;
    std::uint64_t drops_             = 0;
    std::uint64_t exact_comparisons_ = 0;
    double save_seconds_             = 0;
    double load_seconds_             = 0;
    double orphaned_save_seconds_    = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
