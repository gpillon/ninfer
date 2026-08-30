#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {
namespace {

constexpr std::uint32_t kRamMagic   = 0x4D41524E;
constexpr std::uint32_t kRamVersion = 2;
constexpr std::size_t kSectionCount = 18;
constexpr std::size_t kHostAlign    = 8;
constexpr std::size_t kDeviceAlign  = 256;
constexpr std::size_t kFingerprint  = 56;

std::size_t align_up(std::size_t value, std::size_t align) {
    return (value + align - 1) & ~(align - 1);
}

constexpr std::size_t kLineageAnchorTokens = 256;
constexpr std::size_t kLineageCacheCap     = 2048;

// FNV-1a over the leading tokens of a capture's ledger. This is a lineage hint, not an identity
// check: two different conversations sharing the same system prompt/tool schema prefix will
// collide here, which only makes the cache slightly more generous about what counts as "hot" --
// exact-content correctness for reuse itself is still enforced separately via prefix_matches().
std::uint64_t hash_origin(std::span<const TokenId> ledger) {
    const std::size_t count = std::min(ledger.size(), kLineageAnchorTokens);
    std::uint64_t hash      = 1469598103934665603ULL;
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = static_cast<std::uint32_t>(ledger[i]);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

struct Cursor {
    std::uint8_t* p   = nullptr;
    std::uint8_t* end = nullptr;

    void u8(std::uint8_t v) {
        if (p >= end) { throw std::logic_error("RAM entry write overflow"); }
        *p++ = v;
    }
    void u32(std::uint32_t v) {
        u8(static_cast<std::uint8_t>(v));
        u8(static_cast<std::uint8_t>(v >> 8));
        u8(static_cast<std::uint8_t>(v >> 16));
        u8(static_cast<std::uint8_t>(v >> 24));
    }
    void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
    void u64(std::uint64_t v) {
        for (int s = 0; s < 64; s += 8) { u8(static_cast<std::uint8_t>(v >> s)); }
    }
    void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }
    void bytes(const void* data, std::size_t n) {
        const auto* raw = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { u8(raw[i]); }
    }
};

struct InCursor {
    const std::uint8_t* p   = nullptr;
    const std::uint8_t* end = nullptr;

    [[nodiscard]] std::uint8_t u8() {
        if (p >= end) { throw std::logic_error("RAM entry read overflow"); }
        return *p++;
    }
    [[nodiscard]] std::uint32_t u32() {
        const std::uint32_t a = u8(), b = u8(), c = u8(), d = u8();
        return a | (b << 8) | (c << 16) | (d << 24);
    }
    [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t v = 0;
        for (int s = 0; s < 64; s += 8) { v |= static_cast<std::uint64_t>(u8()) << s; }
        return v;
    }
    [[nodiscard]] std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
    void bytes(void* data, std::size_t n) {
        auto* raw = static_cast<std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { raw[i] = u8(); }
    }
    void skip(std::size_t n) {
        if (static_cast<std::size_t>(end - p) < n) {
            throw std::logic_error("RAM entry skip overflow");
        }
        p += n;
    }
};

void write_fingerprint(Cursor& w, const Tensor& plane, PagedKVPlaneOrder order) {
    w.u8(static_cast<std::uint8_t>(plane.dtype));
    w.u8(static_cast<std::uint8_t>(order));
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    for (int i = 0; i < 4; ++i) { w.i32(plane.ne[i]); }
    for (int i = 0; i < 4; ++i) { w.i64(plane.nb[i]); }
}

void check_fingerprint(InCursor& r, const Tensor& plane, PagedKVPlaneOrder order, const char* label) {
    const auto dtype = static_cast<DType>(r.u8());
    const auto stored_order = static_cast<PagedKVPlaneOrder>(r.u8());
    r.skip(6);
    std::int32_t ne[4];
    std::int64_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = r.i32(); }
    for (int i = 0; i < 4; ++i) { nb[i] = r.i64(); }
    if (dtype != plane.dtype || stored_order != order) {
        throw std::logic_error(std::string(label) + " plane dtype/order mismatch");
    }
    for (int i = 0; i < 4; ++i) {
        if (ne[i] != plane.ne[i] || nb[i] != plane.nb[i]) {
            throw std::logic_error(std::string(label) + " plane geometry mismatch");
        }
    }
}

struct HeaderView {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    bool hash_c_valid                     = false;
    std::uint32_t rewrite_frontier        = 0;
    std::uint32_t text_mapped_pages       = 0;
    std::uint32_t backend_mapped_pages    = 0;
    std::uint32_t text_plane_count        = 0;
    std::uint32_t backend_plane_count     = 0;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool has_gdn                          = false;
    bool has_dflash                       = false;
    std::uint32_t cyclic_layers           = 0;
    std::uint32_t cyclic_capacity         = 0;
    std::uint32_t cyclic_padded           = 0;
    std::int32_t cyclic_kv_heads          = 0;
    std::int32_t cyclic_head_dim          = 0;
    std::int32_t cyclic_lane_capacity     = 0;
    std::uint64_t tail_hidden_bytes       = 0;
    std::uint64_t gdn_conv_bytes          = 0;
    std::uint64_t gdn_recurrent_bytes     = 0;
    std::uint64_t cyclic_lane_bytes       = 0;
    std::uint64_t text_residual_slot_bytes    = 0;
    std::uint64_t backend_residual_slot_bytes = 0;
    std::uint64_t ring_valid_slot_bytes       = 0;
    std::array<std::uint64_t, kSectionCount> offset{};
    std::array<std::uint64_t, kSectionCount> length{};
    std::uint64_t entry_bytes             = 0;
    std::size_t header_bytes              = 0;
};

// Fixed prologue: the scalar fields written by write_fixed_header, then kSectionCount
// offset/length pairs and entry_bytes. Six more sections and three more scalars over version 1.
constexpr std::size_t kFixedHeader = 348 + 6 * 16 + 3 * 8;

std::size_t header_bytes_for(std::uint32_t text_planes, std::uint32_t backend_planes) {
    return align_up(kFixedHeader + kFingerprint * (text_planes + backend_planes), kHostAlign);
}

void write_fixed_header(Cursor& w, const HeaderView& h) {
    w.u32(kRamMagic);
    w.u32(kRamVersion);
    w.u32(h.execution_frontier);
    w.u32(h.ledger_frontier);
    w.i32(h.rope_delta);
    w.u32(h.text_kv_valid);
    w.u32(h.mtp_kv_valid);
    w.u32(h.dflash_context_frontier);
    w.u8(h.tail_hidden_valid ? 1 : 0);
    w.u8(h.rewrite_valid ? 1 : 0);
    w.u8(static_cast<std::uint8_t>(h.rewrite_kind));
    w.u8(h.hash_c_valid ? 1 : 0);
    w.u32(h.rewrite_frontier);
    w.u32(h.text_mapped_pages);
    w.u32(h.backend_mapped_pages);
    w.u32(h.text_plane_count);
    w.u32(h.backend_plane_count);
    w.u64(h.hash_f.lo);
    w.u64(h.hash_f.hi);
    w.u64(h.hash_c.lo);
    w.u64(h.hash_c.hi);
    w.u8(h.has_gdn ? 1 : 0);
    w.u8(h.has_dflash ? 1 : 0);
    w.u8(0);
    w.u8(0);
    w.u32(h.cyclic_layers);
    w.u32(h.cyclic_capacity);
    w.u32(h.cyclic_padded);
    w.i32(h.cyclic_kv_heads);
    w.i32(h.cyclic_head_dim);
    w.i32(h.cyclic_lane_capacity);
    w.u64(h.tail_hidden_bytes);
    w.u64(h.gdn_conv_bytes);
    w.u64(h.gdn_recurrent_bytes);
    w.u64(h.cyclic_lane_bytes);
    w.u64(h.text_residual_slot_bytes);
    w.u64(h.backend_residual_slot_bytes);
    w.u64(h.ring_valid_slot_bytes);
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        w.u64(h.offset[i]);
        w.u64(h.length[i]);
    }
    w.u64(h.entry_bytes);
}

HeaderView read_header(const void* block, std::size_t bytes) {
    if (block == nullptr || bytes < kFixedHeader) {
        throw std::logic_error("RAM entry header is truncated");
    }
    const auto* raw = static_cast<const std::uint8_t*>(block);
    InCursor r{raw, raw + bytes};
    HeaderView h;
    if (r.u32() != kRamMagic || r.u32() != kRamVersion) {
        throw std::logic_error("RAM entry magic/version mismatch");
    }
    h.execution_frontier      = r.u32();
    h.ledger_frontier         = r.u32();
    h.rope_delta              = r.i32();
    h.text_kv_valid           = r.u32();
    h.mtp_kv_valid            = r.u32();
    h.dflash_context_frontier = r.u32();
    h.tail_hidden_valid       = r.u8() != 0;
    h.rewrite_valid           = r.u8() != 0;
    h.rewrite_kind            = static_cast<RewriteCheckpointKind>(r.u8());
    h.hash_c_valid            = r.u8() != 0;
    h.rewrite_frontier        = r.u32();
    h.text_mapped_pages       = r.u32();
    h.backend_mapped_pages    = r.u32();
    h.text_plane_count        = r.u32();
    h.backend_plane_count     = r.u32();
    h.hash_f.lo               = r.u64();
    h.hash_f.hi               = r.u64();
    h.hash_c.lo               = r.u64();
    h.hash_c.hi               = r.u64();
    h.has_gdn                 = r.u8() != 0;
    h.has_dflash              = r.u8() != 0;
    r.skip(2);
    h.cyclic_layers           = r.u32();
    h.cyclic_capacity         = r.u32();
    h.cyclic_padded           = r.u32();
    h.cyclic_kv_heads         = r.i32();
    h.cyclic_head_dim         = r.i32();
    h.cyclic_lane_capacity    = r.i32();
    h.tail_hidden_bytes       = r.u64();
    h.gdn_conv_bytes          = r.u64();
    h.gdn_recurrent_bytes     = r.u64();
    h.cyclic_lane_bytes       = r.u64();
    h.text_residual_slot_bytes    = r.u64();
    h.backend_residual_slot_bytes = r.u64();
    h.ring_valid_slot_bytes       = r.u64();
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        h.offset[i] = r.u64();
        h.length[i] = r.u64();
    }
    h.entry_bytes  = r.u64();
    h.header_bytes = header_bytes_for(h.text_plane_count, h.backend_plane_count);
    if (h.entry_bytes > bytes || h.header_bytes > bytes) {
        throw std::logic_error("RAM entry header size is inconsistent");
    }
    return h;
}

const std::uint8_t* section_ptr(const void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<const std::uint8_t*>(block) + header.offset[index];
}

std::uint8_t* section_ptr(void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<std::uint8_t*>(block) + header.offset[index];
}

void verify_pool(InCursor& r, const PagedKVPool& pool, std::uint32_t stored_planes,
                 const char* label) {
    if (stored_planes != pool.plane_count()) {
        throw std::logic_error(std::string(label) + " plane count mismatch");
    }
    for (std::uint32_t i = 0; i < stored_planes; ++i) {
        check_fingerprint(r, pool.plane(i), pool.plane_order(), label);
    }
}

void verify_cyclic(const HeaderView& header, const CyclicKVCache& cache) {
    if (header.cyclic_layers != cache.layer_count() ||
        header.cyclic_capacity != cache.capacity() ||
        header.cyclic_padded != cache.padded_capacity() ||
        header.cyclic_kv_heads != cache.num_kv_heads() ||
        header.cyclic_head_dim != cache.head_dim() ||
        header.cyclic_lane_capacity != cache.lane_capacity() ||
        header.cyclic_lane_bytes != cache.lane_host_bytes()) {
        throw std::logic_error("RAM entry cyclic geometry mismatch");
    }
}

RamRestoredHost host_from_header(const void* block, const HeaderView& header) {
    RamRestoredHost out;
    out.execution_frontier      = header.execution_frontier;
    out.ledger_frontier         = header.ledger_frontier;
    out.rope_delta              = header.rope_delta;
    out.text_kv_valid           = header.text_kv_valid;
    out.mtp_kv_valid            = header.mtp_kv_valid;
    out.dflash_context_frontier = header.dflash_context_frontier;
    out.tail_hidden_valid       = header.tail_hidden_valid;
    out.rewrite_valid           = header.rewrite_valid;
    out.rewrite_kind            = header.rewrite_kind;
    out.rewrite_frontier        = header.rewrite_frontier;
    out.backend_image_present   = header.backend_mapped_pages > 0;
    const auto* ledger = section_ptr(block, header, 0);
    if (header.length[0] != header.ledger_frontier * sizeof(TokenId)) {
        throw std::logic_error("RAM entry ledger size mismatch");
    }
    out.ledger.resize(header.ledger_frontier);
    if (!out.ledger.empty()) {
        std::memcpy(out.ledger.data(), ledger, header.length[0]);
    }
    const auto* identity = section_ptr(block, header, 1);
    out.identity.unpack(identity, static_cast<std::size_t>(header.length[1]));
    return out;
}

} // namespace

KVRamCache::KVRamCache(std::size_t capacity_bytes) : arena_(capacity_bytes) {}

KVRamCache::~KVRamCache() {
    reap_retired(true);
    std::vector<std::uint64_t> ids(fifo_.begin(), fifo_.end());
    for (std::uint64_t id : ids) {
        auto it = records_.find(id);
        if (it == records_.end()) { continue; }
        if (it->second.copies_start != nullptr) {
            (void)cudaEventDestroy(it->second.copies_start);
            it->second.copies_start = nullptr;
        }
        if (it->second.copies_done != nullptr) {
            (void)cudaEventSynchronize(it->second.copies_done);
            (void)cudaEventDestroy(it->second.copies_done);
            it->second.copies_done = nullptr;
        }
    }
    records_.clear();
    fifo_.clear();
}

KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) {
    const auto it = records_.find(entry_id);
    if (it == records_.end()) { throw std::logic_error("RAM cache entry id is unknown"); }
    return it->second;
}

const KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) const {
    const auto it = records_.find(entry_id);
    if (it == records_.end()) { throw std::logic_error("RAM cache entry id is unknown"); }
    return it->second;
}

void KVRamCache::destroy_record(std::uint64_t entry_id, bool count_eviction) {
    auto it = records_.find(entry_id);
    if (it == records_.end()) { return; }
    orphaned_save_seconds_ += harvest_record(it->second);
    wait_copies(it->second);
    if (it->second.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(it->second.copies_start));
        it->second.copies_start = nullptr;
    }
    if (it->second.copies_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(it->second.copies_done));
        it->second.copies_done = nullptr;
    }
    it->second.copies_timed = false;
    if (it->second.block != nullptr) { arena_.free(it->second.block); }
    records_.erase(it);
    fifo_.erase(std::remove(fifo_.begin(), fifo_.end(), entry_id), fifo_.end());
    if (count_eviction) { ++evictions_; }
    bump_version();
}

void KVRamCache::begin_copies(Record& record, cudaStream_t stream) {
    if (record.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.copies_start));
        record.copies_start = nullptr;
    }
    CUDA_CHECK(cudaEventCreate(&record.copies_start));
    CUDA_CHECK(cudaEventRecord(record.copies_start, stream));
    record.copies_timed = false;
}

void KVRamCache::record_copies(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) { CUDA_CHECK(cudaEventCreate(&record.copies_done)); }
    CUDA_CHECK(cudaEventRecord(record.copies_done, stream));
    record.copies_timed = record.copies_start != nullptr;
}

double KVRamCache::harvest_record(Record& record) {
    if (!record.copies_timed || record.copies_start == nullptr || record.copies_done == nullptr) {
        return 0;
    }
    wait_copies(record);
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, record.copies_start, record.copies_done));
    record.copies_timed = false;
    CUDA_CHECK(cudaEventDestroy(record.copies_start));
    record.copies_start = nullptr;
    return static_cast<double>(milliseconds) / 1000.0;
}

KvRamCopySeconds KVRamCache::harvest_copy_seconds() {
    KvRamCopySeconds out;
    out.save += orphaned_save_seconds_;
    save_seconds_ += orphaned_save_seconds_;
    orphaned_save_seconds_ = 0;
    for (std::uint64_t id : pending_save_ids_) {
        const auto it = records_.find(id);
        if (it == records_.end()) { continue; }
        const double seconds = harvest_record(it->second);
        out.save += seconds;
        save_seconds_ += seconds;
    }
    pending_save_ids_.clear();
    if (pending_load_id_) {
        const auto it = records_.find(*pending_load_id_);
        if (it != records_.end()) {
            const double seconds = harvest_record(it->second);
            out.load += seconds;
            load_seconds_ += seconds;
        }
        pending_load_id_.reset();
    }
    return out;
}

void KVRamCache::wait_copies(Record& record) {
    if (record.copies_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(record.copies_done)); }
}

void KVRamCache::wait_copies_on_stream(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) { return; }
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, record.copies_done, 0));
        return;
    }
    wait_copies(record);
}

void KVRamCache::retire_record(Record& record) {
    RetiredCopy item;
    item.block       = record.block;
    item.copies_done = record.copies_done;
    record.block     = nullptr;
    record.copies_done = nullptr;
    retired_.push_back(item);
}

void KVRamCache::reap_retired(bool block) {
    std::size_t keep = 0;
    for (RetiredCopy& item : retired_) {
        if (item.copies_done != nullptr) {
            if (!block) {
                const cudaError_t ready = cudaEventQuery(item.copies_done);
                if (ready == cudaErrorNotReady) {
                    retired_[keep++] = item;
                    continue;
                }
                CUDA_CHECK(ready);
            } else {
                CUDA_CHECK(cudaEventSynchronize(item.copies_done));
            }
            CUDA_CHECK(cudaEventDestroy(item.copies_done));
            item.copies_done = nullptr;
        }
        if (item.block != nullptr) {
            arena_.free(item.block);
            item.block = nullptr;
        }
    }
    retired_.resize(keep);
}

bool KVRamCache::lineage_is_hot(std::uint64_t origin_hash) const {
    const auto it = lineage_hits_.find(origin_hash);
    return it != lineage_hits_.end() && it->second > 0;
}

void KVRamCache::note_lineage_hit(std::uint64_t origin_hash) {
    const auto [it, inserted] = lineage_hits_.try_emplace(origin_hash, 0);
    if (inserted) {
        lineage_order_.push_back(origin_hash);
        while (lineage_order_.size() > kLineageCacheCap) {
            lineage_hits_.erase(lineage_order_.front());
            lineage_order_.pop_front();
        }
    }
    if (it->second < std::numeric_limits<std::uint32_t>::max()) { ++it->second; }
}

void KVRamCache::evict_unpinned() {
    // Class is durable across a VRAM spill. Classifier records and unproven non-main records
    // churn first, demonstrated ordinary agent state next, and main conversation state last.
    // Claims remain absolute pins; the final main pass preserves capture forward progress.
    for (int rank = 0; rank != 3; ++rank) {
        for (std::uint64_t id : fifo_) {
            const auto it = records_.find(id);
            if (it == records_.end() || it->second.claims != 0) { continue; }
            const Record& record = it->second;
            const int record_rank = record.owner_class == runtime::RequestClass::Main
                                        ? 2
                                        : (record.owner_class == runtime::RequestClass::Classifier ||
                                                   !record.protected_tier
                                               ? 0
                                               : 1);
            if (record_rank == rank) {
                destroy_record(id, true);
                return;
            }
        }
    }
}

void KVRamCache::claim(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims != 0 && !record.multi_claim) {
        throw std::logic_error("RAM cache entry is already claimed");
    }
    ++record.claims;
    bump_version();
}

void KVRamCache::release(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims == 0) { throw std::logic_error("RAM cache entry is not claimed"); }
    --record.claims;
    bump_version();
}

void KVRamCache::consume(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims == 0) { throw std::logic_error("RAM cache consume requires a claimed entry"); }
    ++restores_;
    note_lineage_hit(record.origin_hash);
    --record.claims;
    if (record.multi_claim) {
        // Another sibling may still want to restore from this same entry -- drop this claim but
        // keep the record, which stays matchable throughout. It ages out through ordinary
        // FIFO/tiered eviction like any other record once nothing claims it further.
        record.protected_tier = true;
        bump_version();
        return;
    }
    retire_record(record);
    records_.erase(entry_id);
    fifo_.erase(std::remove(fifo_.begin(), fifo_.end(), entry_id), fifo_.end());
    bump_version();
    reap_retired(false);
}

std::optional<RamMatch> KVRamCache::plan_match(const PreparedPromptData& prompt,
                                               std::span<const PrefixHash128> hash_chain) {
    std::optional<RamMatch> best;
    for (std::uint64_t id : fifo_) {
        const Record& record = require(id);
        // A claimed exclusive record is spoken for and vanishes on its claimant's restore. A
        // multi_claim record stays matchable while claimed: several siblings of one burst
        // legitimately restore from the same snapshot, and hiding it from the ones that arrive
        // while a first claim is outstanding is what used to make each of them capture its own
        // duplicate of the same lane.
        if (record.claims != 0 && !record.multi_claim) { continue; }
        RamMatch candidate;
        candidate.entry_id = id;
        const bool frontier_hash =
            record.execution_frontier > 0 && record.execution_frontier < hash_chain.size() &&
            hash_chain[record.execution_frontier] == record.hash_f;
        const bool checkpoint_hash =
            record.hash_c_valid && record.checkpoint_frontier > 0 &&
            record.checkpoint_frontier < hash_chain.size() &&
            hash_chain[record.checkpoint_frontier] == record.hash_c;
        if (!frontier_hash && !checkpoint_hash) { continue; }

        const HeaderView header = read_header(record.block, record.bytes);
        const RamRestoredHost host = host_from_header(record.block, header);
        if (frontier_hash) {
            ++exact_comparisons_;
            if (prefix_matches(prompt, host.ledger, host.identity, record.execution_frontier)) {
                candidate.reuse      = PrefixReusePath::AppendAtFrontier;
                candidate.reuse_base = record.execution_frontier;
            }
        }
        if (candidate.reuse_base == 0 && checkpoint_hash) {
            ++exact_comparisons_;
            if (prefix_matches(prompt, host.ledger, host.identity, record.checkpoint_frontier)) {
                candidate.reuse      = record.checkpoint_path;
                candidate.reuse_base = record.checkpoint_frontier;
            }
        }
        if (candidate.reuse_base == 0) { continue; }
        if (!best || candidate.reuse_base > best->reuse_base) { best = candidate; }
    }
    return best;
}

RamRestoredHost KVRamCache::load_host(std::uint64_t entry_id) const {
    const Record& record = require(entry_id);
    return host_from_header(record.block, read_header(record.block, record.bytes));
}

KvRamSnapshot KVRamCache::snapshot() const noexcept {
    std::size_t used = 0;
    for (const auto& entry : records_) { used += entry.second.bytes; }
    return KvRamSnapshot{
        .capacity_bytes = arena_.capacity(),
        .used_bytes     = used,
        .entry_count    = records_.size(),
        .captures       = captures_,
        .restores       = restores_,
        .evictions      = evictions_,
        .drops          = drops_,
        .save_seconds   = save_seconds_,
        .load_seconds   = load_seconds_,
    };
}

bool KVRamCache::capture(const RamCaptureSource& source) {
    if (source.identity == nullptr || source.text == nullptr || source.text_pool == nullptr) {
        throw std::invalid_argument("RAM capture source is incomplete");
    }
    if (source.text_pages > source.text->mapped_page_count()) {
        throw std::logic_error("RAM capture text extent exceeds mapped pages");
    }
    if ((source.backend == nullptr) != (source.backend_pool == nullptr) ||
        (source.backend == nullptr && source.backend_pages != 0)) {
        throw std::invalid_argument("RAM capture backend source is inconsistent");
    }
    if (source.backend != nullptr && source.backend_pages > source.backend->mapped_page_count()) {
        throw std::logic_error("RAM capture backend extent exceeds mapped pages");
    }
    if (source.ledger.size() != source.ledger_frontier ||
        source.ledger_frontier != source.execution_frontier + 1) {
        throw std::logic_error("RAM capture ledger frontier is inconsistent");
    }

    HeaderView header;
    header.execution_frontier      = source.execution_frontier;
    header.ledger_frontier         = source.ledger_frontier;
    header.rope_delta              = source.rope_delta;
    header.text_kv_valid           = source.text_kv_valid;
    header.mtp_kv_valid            = source.mtp_kv_valid;
    header.dflash_context_frontier = source.dflash_context_frontier;
    header.tail_hidden_valid       = source.tail_hidden_valid;
    header.rewrite_valid           = source.rewrite_valid;
    header.rewrite_kind            = source.rewrite_kind;
    header.hash_c_valid            = source.hash_c_valid;
    header.rewrite_frontier        = source.rewrite_frontier;
    header.text_mapped_pages       = source.text_pages;
    header.backend_mapped_pages    = source.backend_pages;
    header.text_plane_count        = static_cast<std::uint32_t>(source.text_pool->plane_count());
    header.backend_plane_count     =
        source.backend_pool ? static_cast<std::uint32_t>(source.backend_pool->plane_count()) : 0;
    header.hash_f                  = source.hash_f;
    header.hash_c                  = source.hash_c;
    header.has_gdn                 = source.gdn != nullptr;
    header.has_dflash              = source.dflash_local != nullptr;
    if (source.dflash_local != nullptr) {
        header.cyclic_layers       = source.dflash_local->layer_count();
        header.cyclic_capacity     = source.dflash_local->capacity();
        header.cyclic_padded       = source.dflash_local->padded_capacity();
        header.cyclic_kv_heads     = source.dflash_local->num_kv_heads();
        header.cyclic_head_dim     = source.dflash_local->head_dim();
        header.cyclic_lane_capacity = source.dflash_local->lane_capacity();
        header.cyclic_lane_bytes   = source.dflash_local->lane_host_bytes();
    }
    if (source.tail_hidden != nullptr) { header.tail_hidden_bytes = source.tail_hidden->bytes(); }
    if (source.gdn != nullptr) {
        header.gdn_conv_bytes      = source.gdn->conv_host_image_bytes();
        header.gdn_recurrent_bytes = source.gdn->recurrent_host_image_bytes();
    }
    // The exact-key side store lives outside the paged pool and is indexed by slot row, so a
    // record that omits it would leave a restored sequence reading the row's previous tenant.
    const bool text_residual =
        source.text_cache != nullptr && source.text_cache->residual_enabled();
    const bool backend_residual =
        source.backend != nullptr && source.backend_cache != nullptr &&
        source.backend_cache->residual_enabled();
    if ((text_residual || backend_residual) && source.residual_row < 0) {
        throw std::logic_error("RAM capture needs the residual slot row");
    }
    if (text_residual) {
        header.text_residual_slot_bytes = source.text_cache->residual_slot_host_bytes();
        header.ring_valid_slot_bytes    = source.text_cache->ring_valid_slot_host_bytes();
    }
    if (backend_residual) {
        header.backend_residual_slot_bytes = source.backend_cache->residual_slot_host_bytes();
        header.ring_valid_slot_bytes       = source.backend_cache->ring_valid_slot_host_bytes();
    }

    std::array<std::size_t, kSectionCount> lengths{};
    std::array<std::size_t, kSectionCount> aligns{};
    lengths[0] = source.ledger.size() * sizeof(TokenId);
    lengths[1] = source.identity->packed_bytes();
    lengths[2] = paged_kv_host_image_bytes(*source.text_pool, header.text_mapped_pages);
    lengths[3] = source.backend_pool
                     ? paged_kv_host_image_bytes(*source.backend_pool, header.backend_mapped_pages)
                     : 0;
    lengths[4] = source.gdn ? source.gdn->conv_host_image_bytes() : 0;
    lengths[5] = source.gdn && source.rewrite_valid ? source.gdn->conv_host_image_bytes() : 0;
    lengths[6] = source.gdn ? source.gdn->recurrent_host_image_bytes() : 0;
    lengths[7] = source.gdn && source.rewrite_valid ? source.gdn->recurrent_host_image_bytes() : 0;
    lengths[8] = source.tail_hidden ? source.tail_hidden->bytes() : 0;
    lengths[9] = source.rewrite_valid && source.rewrite_checkpoint_hidden
                     ? source.rewrite_checkpoint_hidden->bytes()
                     : 0;
    lengths[10] = source.dflash_local ? source.dflash_local->lane_host_bytes() : 0;
    lengths[11] = source.dflash_checkpoint && source.rewrite_valid
                      ? source.dflash_checkpoint->lane_host_bytes()
                      : 0;
    lengths[12] = text_residual ? header.text_residual_slot_bytes : 0;
    lengths[13] = lengths[12];
    lengths[14] = text_residual ? header.ring_valid_slot_bytes : 0;
    lengths[15] = backend_residual ? header.backend_residual_slot_bytes : 0;
    lengths[16] = lengths[15];
    lengths[17] = backend_residual ? header.ring_valid_slot_bytes : 0;
    aligns[0] = kHostAlign;
    aligns[1] = kHostAlign;
    for (std::size_t i = 2; i < kSectionCount; ++i) { aligns[i] = kDeviceAlign; }

    const std::size_t header_bytes =
        header_bytes_for(header.text_plane_count, header.backend_plane_count);
    std::size_t cursor = header_bytes;
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        if (lengths[i] == 0) { continue; }
        cursor          = align_up(cursor, aligns[i]);
        header.offset[i] = cursor;
        header.length[i] = lengths[i];
        cursor += lengths[i];
    }
    header.entry_bytes  = align_up(cursor, kDeviceAlign);
    header.header_bytes = header_bytes;

    if (header.entry_bytes > arena_.capacity()) {
        ++drops_;
        bump_version();
        return false;
    }

    reap_retired(false);
    void* block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    if (block == nullptr) {
        reap_retired(true);
        block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    }
    while (block == nullptr) {
        const std::size_t before = records_.size();
        evict_unpinned();
        if (records_.size() == before) {
            ++drops_;
            bump_version();
            return false;
        }
        block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    }

    bool copies_launched = false;
    std::uint64_t live_id  = 0;
    cudaEvent_t copies_start = nullptr;
    try {
        auto* raw = static_cast<std::uint8_t*>(block);
        std::memset(raw, 0, header_bytes);
        Cursor w{raw, raw + header_bytes};
        write_fixed_header(w, header);
        for (std::uint32_t i = 0; i < header.text_plane_count; ++i) {
            write_fingerprint(w, source.text_pool->plane(i), source.text_pool->plane_order());
        }
        if (source.backend_pool != nullptr) {
            for (std::uint32_t i = 0; i < header.backend_plane_count; ++i) {
                write_fingerprint(w, source.backend_pool->plane(i),
                                  source.backend_pool->plane_order());
            }
        }

        if (lengths[0] != 0) {
            std::memcpy(raw + header.offset[0], source.ledger.data(), lengths[0]);
        }
        if (lengths[1] != 0) { source.identity->pack(raw + header.offset[1]); }
        const auto start_device_copies = [&] {
            if (copies_start != nullptr) { return; }
            CUDA_CHECK(cudaEventCreate(&copies_start));
            CUDA_CHECK(cudaEventRecord(copies_start, source.stream));
        };
        if (lengths[2] != 0) {
            start_device_copies();
            pack_paged_kv_allocation_to_host(*source.text, *source.text_pool,
                                             header.text_mapped_pages, raw + header.offset[2],
                                             source.stream);
            copies_launched = true;
        }
        if (source.backend != nullptr && lengths[3] != 0) {
            start_device_copies();
            pack_paged_kv_allocation_to_host(*source.backend, *source.backend_pool,
                                             header.backend_mapped_pages, raw + header.offset[3],
                                             source.stream);
            copies_launched = true;
        }
        if (source.gdn != nullptr) {
            start_device_copies();
            source.gdn->pack_slot_to_host(source.gdn_current_slot, raw + header.offset[4],
                                          raw + header.offset[6], source.stream);
            copies_launched = true;
            if (source.rewrite_valid) {
                source.gdn->pack_slot_to_host(source.gdn_checkpoint_slot, raw + header.offset[5],
                                              raw + header.offset[7], source.stream);
            }
        }
        if (source.tail_hidden != nullptr && lengths[8] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[8], source.tail_hidden->data, lengths[8],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }
        if (source.rewrite_checkpoint_hidden != nullptr && lengths[9] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[9],
                                       source.rewrite_checkpoint_hidden->data, lengths[9],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }
        if (source.dflash_local != nullptr && lengths[10] != 0) {
            start_device_copies();
            source.dflash_local->copy_lane_to_host(source.dflash_lane, raw + header.offset[10],
                                                   source.stream);
            copies_launched = true;
        }
        if (source.dflash_checkpoint != nullptr && lengths[11] != 0) {
            start_device_copies();
            source.dflash_checkpoint->copy_lane_to_host(source.dflash_lane, raw + header.offset[11],
                                                        source.stream);
            copies_launched = true;
        }
        if (lengths[12] != 0) {
            start_device_copies();
            source.text_cache->pack_residual_slot_to_host(
                source.residual_row, raw + header.offset[12], raw + header.offset[13],
                raw + header.offset[14], source.stream);
            copies_launched = true;
        }
        if (lengths[15] != 0) {
            start_device_copies();
            source.backend_cache->pack_residual_slot_to_host(
                source.residual_row, raw + header.offset[15], raw + header.offset[16],
                raw + header.offset[17], source.stream);
            copies_launched = true;
        }

        if (next_id_ == 0) { throw std::logic_error("RAM cache entry id overflow"); }
        Record record;
        record.id                  = next_id_++;
        record.hash_f              = source.hash_f;
        record.hash_c              = source.hash_c;
        record.hash_c_valid        = source.hash_c_valid;
        record.execution_frontier  = source.execution_frontier;
        record.checkpoint_frontier = source.rewrite_frontier;
        record.checkpoint_valid    = source.rewrite_valid;
        record.checkpoint_path     = source.rewrite_kind == RewriteCheckpointKind::TurnClosure
                                         ? PrefixReusePath::RestoreTurnCheckpoint
                                         : PrefixReusePath::RestoreResponseCheckpoint;
        record.block               = block;
        record.bytes               = header.entry_bytes;
        record.origin_hash         = hash_origin(source.ledger);
        record.protected_tier      = lineage_is_hot(record.origin_hash);
        record.multi_claim         = source.multi_claim;
        record.owner_class         = source.owner_class;
        record.copies_start        = copies_start;
        copies_start               = nullptr;
        const auto [it, inserted]  = records_.emplace(record.id, record);
        if (!inserted) { throw std::logic_error("RAM cache entry id already exists"); }
        live_id = record.id;
        fifo_.push_back(record.id);
        record_copies(it->second, source.stream);
        pending_save_ids_.push_back(record.id);
        ++captures_;
        bump_version();
        return true;
    } catch (...) {
        if (copies_launched) {
            if (source.stream != nullptr) {
                (void)cudaStreamSynchronize(source.stream);
            } else {
                (void)cudaDeviceSynchronize();
            }
        }
        if (copies_start != nullptr) { (void)cudaEventDestroy(copies_start); }
        if (live_id != 0) {
            destroy_record(live_id, false);
        } else {
            arena_.free(block);
        }
        throw;
    }
}

RamRestoredHost KVRamCache::unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target) {
    Record& record            = require(entry_id);
    wait_copies_on_stream(record, target.stream);
    const HeaderView header   = read_header(record.block, record.bytes);
    auto* raw                 = static_cast<std::uint8_t*>(record.block);
    const auto* fingerprint   = raw + kFixedHeader;
    InCursor fp{fingerprint, raw + header.header_bytes};
    if (target.text == nullptr || target.text_pool == nullptr) {
        throw std::invalid_argument("RAM restore target is incomplete");
    }
    verify_pool(fp, *target.text_pool, header.text_plane_count, "text KV");
    if (header.backend_plane_count != 0) {
        if (target.backend == nullptr || target.backend_pool == nullptr) {
            throw std::logic_error("RAM restore is missing the backend pool");
        }
        verify_pool(fp, *target.backend_pool, header.backend_plane_count, "backend KV");
    }
    if (header.has_dflash) {
        if (target.dflash_local == nullptr) {
            throw std::logic_error("RAM restore is missing DFlash cyclic state");
        }
        verify_cyclic(header, *target.dflash_local);
        if (header.length[11] != 0) {
            if (target.dflash_checkpoint == nullptr) {
                throw std::logic_error("RAM restore is missing DFlash checkpoint cyclic state");
            }
            verify_cyclic(header, *target.dflash_checkpoint);
        }
    }
    if (header.has_gdn) {
        if (target.gdn == nullptr) { throw std::logic_error("RAM restore is missing GDN state"); }
        if (header.gdn_conv_bytes != target.gdn->conv_host_image_bytes() ||
            header.gdn_recurrent_bytes != target.gdn->recurrent_host_image_bytes()) {
            throw std::logic_error("RAM entry GDN geometry mismatch");
        }
    }
    if (target.tail_hidden != nullptr &&
        header.tail_hidden_bytes != target.tail_hidden->bytes()) {
        throw std::logic_error("RAM entry hidden geometry mismatch");
    }
    // A cache that keeps an exact-key side store must be handed the record's own copy of it. If
    // the record carries none, the destination row would keep whatever the previous tenant of
    // that row left behind and the decode kernel would attend to it, so refuse rather than
    // restore a sequence the record only partly describes.
    const bool text_residual =
        target.text_cache != nullptr && target.text_cache->residual_enabled();
    const bool backend_residual = target.backend != nullptr && target.backend_cache != nullptr &&
                                  target.backend_cache->residual_enabled();
    if ((text_residual && header.length[12] == 0) ||
        (backend_residual && header.length[15] == 0)) {
        throw std::logic_error("RAM entry lacks the exact-key side store this cache requires");
    }
    if ((header.length[12] != 0 && !text_residual) ||
        (header.length[15] != 0 && !backend_residual)) {
        throw std::logic_error("RAM entry carries a side store this cache does not use");
    }
    if ((text_residual || backend_residual) && target.residual_row < 0) {
        throw std::logic_error("RAM restore needs the residual slot row");
    }
    if (text_residual &&
        (header.text_residual_slot_bytes != target.text_cache->residual_slot_host_bytes() ||
         header.ring_valid_slot_bytes != target.text_cache->ring_valid_slot_host_bytes())) {
        throw std::logic_error("RAM entry residual geometry mismatch");
    }
    if (backend_residual &&
        (header.backend_residual_slot_bytes != target.backend_cache->residual_slot_host_bytes() ||
         header.ring_valid_slot_bytes != target.backend_cache->ring_valid_slot_host_bytes())) {
        throw std::logic_error("RAM entry backend residual geometry mismatch");
    }

    begin_copies(record, target.stream);
    unpack_paged_kv_allocation_from_host(*target.text, *target.text_pool, raw + header.offset[2],
                                         header.text_mapped_pages, target.text_dst_pages,
                                         target.stream);
    if (target.backend != nullptr && header.length[3] != 0) {
        unpack_paged_kv_allocation_from_host(*target.backend, *target.backend_pool,
                                             raw + header.offset[3], header.backend_mapped_pages,
                                             target.backend_dst_pages, target.stream);
    }
    if (target.gdn != nullptr && header.length[4] != 0) {
        target.gdn->unpack_slot_from_host(target.gdn_current_slot, raw + header.offset[4],
                                          raw + header.offset[6], target.stream);
        if (header.length[5] != 0) {
            target.gdn->unpack_slot_from_host(target.gdn_checkpoint_slot, raw + header.offset[5],
                                              raw + header.offset[7], target.stream);
        }
    }
    if (target.tail_hidden != nullptr && header.length[8] != 0) {
        CUDA_CHECK(cudaMemcpyAsync(target.tail_hidden->data, raw + header.offset[8],
                                   static_cast<std::size_t>(header.length[8]),
                                   cudaMemcpyHostToDevice, target.stream));
    }
    if (target.rewrite_checkpoint_hidden != nullptr && header.length[9] != 0) {
        if (header.length[9] != target.rewrite_checkpoint_hidden->bytes()) {
            throw std::logic_error("RAM entry rewrite-checkpoint hidden geometry mismatch");
        }
        CUDA_CHECK(cudaMemcpyAsync(target.rewrite_checkpoint_hidden->data, raw + header.offset[9],
                                   static_cast<std::size_t>(header.length[9]),
                                   cudaMemcpyHostToDevice, target.stream));
    }
    if (target.dflash_local != nullptr && header.length[10] != 0) {
        target.dflash_local->copy_lane_from_host(raw + header.offset[10], target.dflash_lane,
                                                 target.stream);
    }
    if (target.dflash_checkpoint != nullptr && header.length[11] != 0) {
        target.dflash_checkpoint->copy_lane_from_host(raw + header.offset[11], target.dflash_lane,
                                                      target.stream);
    }
    if (header.length[12] != 0) {
        target.text_cache->unpack_residual_slot_from_host(
            target.residual_row, raw + header.offset[12], raw + header.offset[13],
            raw + header.offset[14], target.stream);
    }
    if (header.length[15] != 0) {
        target.backend_cache->unpack_residual_slot_from_host(
            target.residual_row, raw + header.offset[15], raw + header.offset[16],
            raw + header.offset[17], target.stream);
    }
    record_copies(record, target.stream);
    pending_load_id_ = entry_id;
    return host_from_header(record.block, header);
}

void KVRamCache::test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte) {
    Record& record            = require(entry_id);
    const HeaderView header   = read_header(record.block, record.bytes);
    auto* identity_bytes      = section_ptr(record.block, header, 1);
    ResidentPrefixIdentity identity;
    identity.unpack(identity_bytes, static_cast<std::size_t>(header.length[1]));
    identity.test_tamper_content_digest(0, byte);
    identity.pack(identity_bytes);
}

} // namespace ninfer::targets::qwen3_6::detail
