#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/decoder_state.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

void fill_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        unsigned char seed) {
    const auto pages                    = allocation.page_ids();
    const ninfer::PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
                std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                          host.begin() + static_cast<std::ptrdiff_t>(begin + tensor.nb[3]), value);
            } else {
                const std::size_t bpp =
                    static_cast<std::size_t>(tensor.ne[3]) * static_cast<std::size_t>(tensor.nb[2]);
                for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
                    const std::size_t begin = static_cast<std::size_t>(
                        head * tensor.nb[3] + pages[i] * tensor.nb[2]);
                    std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                              host.begin() + static_cast<std::ptrdiff_t>(begin + tensor.nb[2]),
                              static_cast<unsigned char>(value + static_cast<unsigned>(head)));
                }
                (void)bpp;
            }
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int expect_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                         unsigned char seed, const char* label) {
    const auto pages                    = allocation.page_ids();
    const ninfer::PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
                for (std::int64_t byte = 0; byte < tensor.nb[3]; ++byte) {
                    if (host[begin + static_cast<std::size_t>(byte)] != value) {
                        std::cerr << label << " PageMajor logical page " << i << " plane " << plane
                                  << " mismatch\n";
                        return 1;
                    }
                }
            } else {
                for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
                    const unsigned char expected =
                        static_cast<unsigned char>(value + static_cast<unsigned>(head));
                    const std::size_t begin = static_cast<std::size_t>(
                        head * tensor.nb[3] + pages[i] * tensor.nb[2]);
                    for (std::int64_t byte = 0; byte < tensor.nb[2]; ++byte) {
                        if (host[begin + static_cast<std::size_t>(byte)] != expected) {
                            std::cerr << label << " HeadMajor logical page " << i << " head "
                                      << head << " mismatch\n";
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int expect_host_page_major_layout(const unsigned char* image, const ninfer::PagedKVPool& pool,
                                  std::uint32_t page_count, unsigned char seed, const char* label) {
    std::size_t offset = 0;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        const std::size_t bpp        = static_cast<std::size_t>(tensor.nb[3]);
        for (std::uint32_t i = 0; i < page_count; ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + i + 1U);
            for (std::size_t byte = 0; byte < bpp; ++byte) {
                if (image[offset + i * bpp + byte] != value) {
                    std::cerr << label << " host PageMajor layout mismatch\n";
                    return 1;
                }
            }
        }
        offset += static_cast<std::size_t>(page_count) * bpp;
    }
    return 0;
}

int expect_host_head_major_layout(const unsigned char* image, const ninfer::PagedKVPool& pool,
                                  std::uint32_t page_count, unsigned char seed, const char* label) {
    const ninfer::Tensor& tensor = pool.plane(0);
    const std::size_t bpp =
        static_cast<std::size_t>(tensor.ne[3]) * static_cast<std::size_t>(tensor.nb[2]);
    for (std::uint32_t i = 0; i < page_count; ++i) {
        const unsigned char value = static_cast<unsigned char>(seed + i + 1U);
        for (std::int32_t head = 0; head < tensor.ne[3]; ++head) {
            const unsigned char expected =
                static_cast<unsigned char>(value + static_cast<unsigned>(head));
            const std::size_t begin = static_cast<std::size_t>(i) * bpp +
                                      static_cast<std::size_t>(head) * tensor.nb[2];
            for (std::int64_t byte = 0; byte < tensor.nb[2]; ++byte) {
                if (image[begin + static_cast<std::size_t>(byte)] != expected) {
                    std::cerr << label << " host HeadMajor page " << i << " is not i*bpp packed\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int round_trip_pool(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool,
                    std::uint32_t entitlement, std::uint32_t mapped, unsigned char seed,
                    const char* label) {
    auto source = pool.reserve(entitlement);
    source.materialize_pages(mapped);
    fill_logical_pages(pool, source, seed);
    const std::uint32_t captured = source.mapped_page_count();
    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, captured);
    ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
    void* image = host.try_alloc(image_bytes, 256);
    if (image == nullptr) {
        std::cerr << label << " host image allocation failed\n";
        return 1;
    }
    ninfer::pack_paged_kv_allocation_to_host(source, pool, captured, image, ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    int failures = 0;
    if (pool.plane_order() == ninfer::PagedKVPlaneOrder::PageMajor) {
        failures += expect_host_page_major_layout(static_cast<const unsigned char*>(image), pool,
                                                  captured, seed, label);
    } else {
        failures += expect_host_head_major_layout(static_cast<const unsigned char*>(image), pool,
                                                  captured, seed, label);
    }
    source.release();

    auto destination = pool.reserve(entitlement);
    destination.materialize_pages(mapped);
    ninfer::unpack_paged_kv_allocation_from_host(destination, pool, image, captured, mapped,
                                                 ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    failures += expect_logical_pages(pool, destination, seed, label);
    destination.release();
    return failures;
}

ninfer::targets::qwen3_6::PreparedPromptData text_prompt(std::vector<ninfer::TokenId> tokens) {
    ninfer::targets::qwen3_6::PreparedPromptData prompt;
    prompt.token_ids   = std::move(tokens);
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(3 * prompt.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i) {
            prompt.positions[static_cast<std::size_t>(axis) * prompt.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    return prompt;
}

int capture_text_entry(
    ninfer::targets::qwen3_6::detail::KVRamCache& cache, ninfer::PagedKVPool& pool,
    ninfer::PagedKVAllocation& alloc, const ninfer::targets::qwen3_6::PreparedPromptData& prompt,
    cudaStream_t stream, std::uint32_t checkpoint_frontier = 0,
    std::uint32_t capture_pages = std::numeric_limits<std::uint32_t>::max(),
    bool multi_claim = false, ninfer::RequestClass owner_class = ninfer::RequestClass::Agents,
    ninfer::targets::qwen3_6::detail::RamCaptureKind capture_kind =
        ninfer::targets::qwen3_6::detail::RamCaptureKind::Terminal,
    ninfer::targets::qwen3_6::detail::RamCapturePolicy policy =
        ninfer::targets::qwen3_6::detail::RamCapturePolicy::AllowEviction,
    std::size_t* expected_bytes = nullptr) {
    ninfer::targets::qwen3_6::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    ninfer::targets::qwen3_6::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    ninfer::targets::qwen3_6::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid      = source.execution_frontier;
    source.tail_hidden_valid  = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = ninfer::targets::qwen3_6::detail::prefix_hash_at(
        retained.token_ids, identity, source.execution_frontier);
    if (checkpoint_frontier != 0) {
        source.rewrite_valid     = true;
        source.rewrite_kind      = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure;
        source.rewrite_frontier  = checkpoint_frontier;
        source.hash_c_valid      = true;
        source.hash_c            = ninfer::targets::qwen3_6::detail::prefix_hash_at(
            retained.token_ids, identity, checkpoint_frontier);
    }
    source.text      = &alloc;
    source.text_pool = &pool;
    source.text_pages = capture_pages == std::numeric_limits<std::uint32_t>::max()
                            ? alloc.mapped_page_count()
                            : capture_pages;
    source.stream    = stream;
    source.multi_claim = multi_claim;
    source.owner_class = owner_class;
    source.capture_kind = capture_kind;
    if (expected_bytes != nullptr) { *expected_bytes = cache.capture_bytes(source); }
    return cache.capture(source, policy) ? 0 : 1;
}

int expect_logical_page(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        std::size_t logical_index, unsigned char seed, const char* label);

// The hyperquant exact-key side store is indexed by slot row, not by sequence, so a record that
// travels between rows has to carry its own row's image; restoring into a row that still holds a
// previous tenant's keys is what made a restored request answer with another request's context.
// This pins the row indexing in both directions: every layer of the captured row must land in the
// destination row, and no other row may be touched.
int test_residual_slot_round_trip(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;

    q36::DecoderStateSpec spec;
    spec.full_attention_layers     = 2;
    spec.mtp_layers                = 0;
    spec.capacity                  = 128;
    spec.kv_heads                  = 4;
    spec.attention_head_dim        = 256;
    spec.kv_dtype                  = ninfer::DType::U8;
    spec.kv_quant_group            = q36::kKvHqQuantGroup;
    spec.enable_mtp                = false;
    spec.kv_table_rows             = 3;
    spec.text_physical_page_groups = 4;
    // plan_decoder_state rejects a zero-layer linear-attention pool, so this has to describe a
    // real (if tiny) one even though the residual side store under test is independent of it.
    spec.linear_attention          = {.layers         = 2,
                                      .conv_channels  = 8,
                                      .conv_width     = 4,
                                      .value_heads    = 2,
                                      .value_head_dim = 4,
                                      .key_head_dim   = 3,
                                      .slot_count     = 4};

    ninfer::LayoutBuilder builder;
    const q36::DecoderStateLayout layout = q36::plan_decoder_state(builder, spec);
    ninfer::DeviceArena arena(builder.finish(256));
    q36::PagedKVCache cache({arena.base(), arena.capacity()}, layout.text_kv);
    if (!cache.residual_enabled()) {
        std::cerr << "residual side store is not enabled for the hq layout\n";
        return 1;
    }

    const std::size_t slot_bytes = cache.residual_slot_host_bytes();
    const std::size_t ring_bytes = cache.ring_valid_slot_host_bytes();
    if (slot_bytes == 0 || ring_bytes == 0) {
        std::cerr << "residual slot image sizes are empty\n";
        return 1;
    }

    // Give every row a distinct fill so a mis-indexed copy cannot be mistaken for a correct one.
    const auto fill_row = [&](std::int32_t row, unsigned char byte) {
        std::vector<unsigned char> k(slot_bytes, byte);
        std::vector<unsigned char> v(slot_bytes, static_cast<unsigned char>(byte + 1));
        std::vector<unsigned char> ring(ring_bytes, static_cast<unsigned char>(byte + 2));
        cache.unpack_residual_slot_from_host(row, k.data(), v.data(), ring.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    };
    fill_row(0, 0x10);
    fill_row(1, 0x40);
    fill_row(2, 0x70);

    std::vector<unsigned char> k_image(slot_bytes);
    std::vector<unsigned char> v_image(slot_bytes);
    std::vector<unsigned char> ring_image(ring_bytes);
    cache.pack_residual_slot_to_host(1, k_image.data(), v_image.data(), ring_image.data(),
                                     ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    if (k_image != std::vector<unsigned char>(slot_bytes, 0x40) ||
        v_image != std::vector<unsigned char>(slot_bytes, 0x41) ||
        ring_image != std::vector<unsigned char>(ring_bytes, 0x42)) {
        ++failures;
        std::cerr << "residual slot pack read the wrong row\n";
    }

    // Restore row 1's image into row 2, the case a host-RAM restore performs.
    cache.unpack_residual_slot_from_host(2, k_image.data(), v_image.data(), ring_image.data(),
                                         ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto read_row = [&](std::int32_t row) {
        std::vector<unsigned char> k(slot_bytes);
        std::vector<unsigned char> v(slot_bytes);
        std::vector<unsigned char> ring(ring_bytes);
        cache.pack_residual_slot_to_host(row, k.data(), v.data(), ring.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        return std::tuple{k, v, ring};
    };
    const auto [k2, v2, ring2] = read_row(2);
    if (k2 != k_image || v2 != v_image || ring2 != ring_image) {
        ++failures;
        std::cerr << "residual slot unpack did not reach the destination row\n";
    }
    const auto [k0, v0, ring0] = read_row(0);
    if (k0 != std::vector<unsigned char>(slot_bytes, 0x10) ||
        v0 != std::vector<unsigned char>(slot_bytes, 0x11) ||
        ring0 != std::vector<unsigned char>(ring_bytes, 0x12)) {
        ++failures;
        std::cerr << "residual slot unpack disturbed an unrelated row\n";
    }

    // The exact-key side store lives outside the paged pool, so an admission preflight that sizes
    // only the pool under-quotes an hq record by two slot images plus the ring-valid image. Size
    // and capture the same source and require the arena to land on the quoted number.
    {
        auto alloc = cache.pool().reserve(1);
        alloc.materialize_pages(1, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

        const auto prompt                = text_prompt({11, 12, 13});
        q36::PreparedPromptData retained = prompt;
        retained.token_ids.push_back(0);
        retained.token_types.push_back(0);
        const std::size_t tokens = retained.token_ids.size();
        retained.positions.resize(3 * tokens);
        for (int axis = 0; axis < 3; ++axis) {
            for (std::size_t i = 0; i < tokens; ++i) {
                retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                    static_cast<std::int32_t>(i);
            }
        }
        q36::detail::ResidentPrefixIdentity identity;
        identity.assign(retained);

        q36::detail::RamCaptureSource source;
        source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
        source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
        source.text_kv_valid      = source.execution_frontier;
        source.ledger             = retained.token_ids;
        source.identity           = &identity;
        source.hash_f = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                    source.execution_frontier);
        source.text       = &alloc;
        source.text_pool  = &cache.pool();
        source.text_pages = alloc.mapped_page_count();
        source.stream     = ctx.stream;

        q36::detail::KVRamCache sized(8ULL << 20);
        const std::size_t pool_only = sized.capture_bytes(source);
        source.text_cache           = &cache;
        source.residual_row         = 1;
        const std::size_t with_side_store = sized.capture_bytes(source);
        if (with_side_store < pool_only + 2 * slot_bytes + ring_bytes) {
            ++failures;
            std::cerr << "hq residual side store is missing from the capture footprint\n";
        }
        if (!sized.capture(source)) {
            ++failures;
            std::cerr << "hq residual capture failed\n";
        } else if (sized.snapshot().used_bytes != with_side_store) {
            ++failures;
            std::cerr << "hq residual preflight size disagrees with the captured record\n";
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        alloc.release();
    }
    return failures;
}

int test_kv_ram_index(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    const auto chain_a  = q36::detail::prefix_hash_chain(prompt_a);
    const auto chain_b  = q36::detail::prefix_hash_chain(prompt_b);
    const auto chain_c  = q36::detail::prefix_hash_chain(prompt_c);

    q36::detail::KVRamCache probe(8ULL << 20);
    std::size_t expected_entry_bytes = 0;
    if (capture_text_entry(probe, pool, alloc, prompt_a, ctx.stream, 0,
                           std::numeric_limits<std::uint32_t>::max(), false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::Terminal,
                           q36::detail::RamCapturePolicy::AllowEviction,
                           &expected_entry_bytes) != 0) {
        std::cerr << "RAM probe capture failed\n";
        alloc.release();
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    if (entry_bytes == 0 || entry_bytes != expected_entry_bytes) {
        std::cerr << "RAM preflight size disagrees with the captured record\n";
        alloc.release();
        return 1;
    }

    q36::detail::KVRamCache oversize(4096);
    if (capture_text_entry(oversize, pool, alloc, prompt_a, ctx.stream) == 0 ||
        oversize.snapshot().drops == 0) {
        std::cerr << "oversize capture did not drop\n";
        ++failures;
    }

    q36::detail::KVRamCache pinned(entry_bytes);
    if (capture_text_entry(pinned, pool, alloc, prompt_a, ctx.stream) != 0) {
        std::cerr << "single-entry RAM capture failed\n";
        alloc.release();
        return failures + 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto pinned_match = pinned.plan_match(prompt_a, chain_a);
    if (!pinned_match) {
        std::cerr << "single-entry RAM capture did not match\n";
        ++failures;
    } else {
        pinned.claim(pinned_match->entry_id);
        if (capture_text_entry(pinned, pool, alloc, prompt_b, ctx.stream) == 0 ||
            pinned.snapshot().drops == 0 || pinned.snapshot().entry_count != 1) {
            std::cerr << "claimed RAM entry did not survive a capture storm\n";
            ++failures;
        }
        if (pinned.plan_match(prompt_a, chain_a)) {
            std::cerr << "plan_match returned a claimed RAM entry during a capture storm\n";
            ++failures;
        }
        pinned.release(pinned_match->entry_id);
        if (!pinned.plan_match(prompt_a, chain_a)) {
            std::cerr << "release after a capture storm did not restore the match\n";
            ++failures;
        }
        pinned.claim(pinned_match->entry_id);
        pinned.consume(pinned_match->entry_id);
        if (pinned.plan_match(prompt_a, chain_a) || pinned.snapshot().restores != 1) {
            std::cerr << "consume did not remove the RAM entry\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache cache(4ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(cache, pool, alloc, prompt_b, ctx.stream) != 0) {
        std::cerr << "RAM index capture failed\n";
        alloc.release();
        return failures + 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto match_a = cache.plan_match(prompt_a, chain_a);
    const auto match_b = cache.plan_match(prompt_b, chain_b);
    if (!match_a || !match_b || match_a->entry_id == match_b->entry_id ||
        match_a->reuse_base != prompt_a.token_ids.size() ||
        match_b->reuse_base != prompt_b.token_ids.size()) {
        std::cerr << "RAM index failed to match captured prompts\n";
        ++failures;
    }

    auto changed = prompt_a;
    changed.token_ids.back() += 1;
    const auto before_exact = cache.exact_comparisons();
    const auto miss_hash    = cache.plan_match(changed, q36::detail::prefix_hash_chain(changed));
    if (miss_hash || cache.exact_comparisons() != before_exact) {
        std::cerr << "hash-prefilter miss performed an exact comparison\n";
        ++failures;
    }

    q36::PreparedPromptData vision = prompt_a;
    q36::VisionItem item;
    item.modality    = q36::PromptModality::Image;
    item.grid        = {.temporal = 1, .height = 2, .width = 4};
    item.patch_count = 8;
    item.token_spans = {{.begin = 0, .count = 4}};
    item.content_digest.fill(1);
    vision.token_types.assign(4, static_cast<std::uint8_t>(q36::PromptModality::Image));
    vision.vision_items.push_back(item);
    q36::detail::KVRamCache vision_cache(4ULL << 20);
    if (capture_text_entry(vision_cache, pool, alloc, vision, ctx.stream) != 0) {
        std::cerr << "vision RAM capture failed\n";
        alloc.release();
        return failures + 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto vision_match =
        vision_cache.plan_match(vision, q36::detail::prefix_hash_chain(vision));
    if (!vision_match) {
        std::cerr << "vision RAM entry did not match itself\n";
        ++failures;
    } else {
        vision_cache.test_tamper_identity_digest(vision_match->entry_id, 9);
        const auto before_tamper = vision_cache.exact_comparisons();
        const auto tampered =
            vision_cache.plan_match(vision, q36::detail::prefix_hash_chain(vision));
        if (tampered || vision_cache.exact_comparisons() <= before_tamper) {
            std::cerr << "exact-gate digest tamper did not miss via prefix_matches\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache two(entry_bytes * 2 + 256);
    if (capture_text_entry(two, pool, alloc, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(two, pool, alloc, prompt_b, ctx.stream) != 0) {
        std::cerr << "two-entry RAM capture failed\n";
        ++failures;
    } else {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const auto keep = two.plan_match(prompt_a, chain_a);
        if (!keep) {
            std::cerr << "two-entry cache lost prompt A\n";
            ++failures;
        } else {
            const auto version_before_claim = two.index_version();
            two.claim(keep->entry_id);
            if (two.index_version() == version_before_claim) {
                std::cerr << "claim did not bump the RAM index version\n";
                ++failures;
            }
            if (two.plan_match(prompt_a, chain_a)) {
                std::cerr << "plan_match returned a claimed RAM entry\n";
                ++failures;
            }
            (void)capture_text_entry(two, pool, alloc, prompt_c, ctx.stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (two.plan_match(prompt_a, chain_a) || two.plan_match(prompt_b, chain_b) ||
                two.snapshot().entry_count != 2) {
                std::cerr << "pinned FIFO eviction removed the claimed entry\n";
                ++failures;
            }
            (void)chain_c;
            two.release(keep->entry_id);
            if (!two.plan_match(prompt_a, chain_a)) {
                std::cerr << "release did not make the claimed entry matchable\n";
                ++failures;
            }
        }
    }

    q36::detail::KVRamCache fifo(4ULL << 20);
    if (capture_text_entry(fifo, pool, alloc, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(fifo, pool, alloc, prompt_a, ctx.stream) != 0) {
        std::cerr << "duplicate RAM capture failed\n";
        ++failures;
    } else {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const auto oldest = fifo.plan_match(prompt_a, chain_a);
        if (!oldest || oldest->reuse_base != prompt_a.token_ids.size()) {
            std::cerr << "duplicate RAM captures did not match the captured frontier\n";
            ++failures;
        }
    }

    auto longer = prompt_a;
    longer.token_ids.push_back(99);
    longer.token_types.push_back(0);
    longer.positions.resize(3 * longer.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < longer.token_ids.size(); ++i) {
            longer.positions[static_cast<std::size_t>(axis) * longer.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    const auto longer_match =
        fifo.plan_match(longer, q36::detail::prefix_hash_chain(longer));
    if (!longer_match || longer_match->reuse_base != prompt_a.token_ids.size()) {
        std::cerr << "RAM index did not match a longer prompt at the captured frontier\n";
        ++failures;
    }

    q36::detail::KVRamCache lengths(8ULL << 20);
    if (capture_text_entry(lengths, pool, alloc, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(lengths, pool, alloc, longer, ctx.stream) != 0) {
        std::cerr << "different-length RAM capture failed\n";
        ++failures;
    } else {
        const auto longest = lengths.plan_match(longer, q36::detail::prefix_hash_chain(longer));
        if (!longest || longest->reuse_base != longer.token_ids.size()) {
            std::cerr << "RAM index did not prefer the longer captured frontier\n";
            ++failures;
        }
    }

    auto checkpoint_prompt = text_prompt({10, 11, 12, 13});
    q36::detail::KVRamCache checkpoint(8ULL << 20);
    if (capture_text_entry(checkpoint, pool, alloc, checkpoint_prompt, ctx.stream, 2) != 0) {
        std::cerr << "checkpoint RAM capture failed\n";
        ++failures;
    } else {
        const auto short_prompt = text_prompt({10, 11});
        const auto restored =
            checkpoint.plan_match(short_prompt, q36::detail::prefix_hash_chain(short_prompt));
        if (!restored || restored->reuse_base != 2 ||
            restored->reuse != ninfer::PrefixReusePath::RestoreTurnCheckpoint) {
            std::cerr << "RAM index did not match the rewrite checkpoint prefix\n";
            ++failures;
        }
    }

    q36::detail::KVRamCache tight(entry_bytes + 256);
    if (capture_text_entry(tight, pool, alloc, prompt_a, ctx.stream) != 0) {
        std::cerr << "tight FIFO first capture failed\n";
        ++failures;
    } else {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const auto evictions_before = tight.snapshot().evictions;
        if (capture_text_entry(tight, pool, alloc, prompt_b, ctx.stream) != 0) {
            std::cerr << "tight FIFO second capture failed\n";
            ++failures;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            if (tight.plan_match(prompt_a, chain_a) || !tight.plan_match(prompt_b, chain_b) ||
                tight.snapshot().evictions <= evictions_before) {
                std::cerr << "unpinned FIFO eviction did not drop the oldest entry\n";
                ++failures;
            }
        }
    }

    alloc.release();
    return failures;
}

int test_capture_extent_and_eviction_policy(ninfer::DeviceContext& ctx,
                                            ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 73);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto prompt_a = text_prompt({101, 102, 103, 104});
    const auto prompt_b = text_prompt({201, 202, 203, 204});
    const auto prompt_c = text_prompt({301, 302, 303, 304});
    const auto chain_a  = q36::detail::prefix_hash_chain(prompt_a);
    const auto chain_b  = q36::detail::prefix_hash_chain(prompt_b);
    const auto chain_c  = q36::detail::prefix_hash_chain(prompt_c);

    q36::detail::KVRamCache full_probe(8ULL << 20);
    q36::detail::KVRamCache prefix_probe(8ULL << 20);
    if (capture_text_entry(full_probe, pool, source, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(prefix_probe, pool, source, prompt_a, ctx.stream, 0, 1) != 0) {
        return fail("explicit-page capture probe failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t full_bytes   = full_probe.snapshot().used_bytes;
    const std::size_t prefix_bytes = prefix_probe.snapshot().used_bytes;
    if (prefix_bytes >= full_bytes ||
        full_bytes - prefix_bytes < paged_kv_host_image_bytes(pool, 1)) {
        std::cerr << "explicit-page capture did not reduce the RAM image: full=" << full_bytes
                  << " prefix=" << prefix_bytes << '\n';
        ++failures;
    }

    const auto prefix_match = prefix_probe.plan_match(prompt_a, chain_a);
    auto destination        = pool.reserve(2);
    destination.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, destination, 19);
    if (!prefix_match) {
        ++failures;
        std::cerr << "explicit-page capture did not index\n";
    } else {
        q36::detail::RamRestoreTarget target;
        target.text           = &destination;
        target.text_pool      = &pool;
        target.text_dst_pages = 1;
        target.stream         = ctx.stream;
        prefix_probe.claim(prefix_match->entry_id);
        (void)prefix_probe.unpack_device(prefix_match->entry_id, target);
        prefix_probe.consume(prefix_match->entry_id);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        failures += expect_logical_page(pool, destination, 0, 73,
                                        "explicit-page restore captured page");
        failures += expect_logical_page(pool, destination, 1, 19,
                                        "explicit-page restore untouched tail");
    }
    destination.release();

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return failures;
}

int test_persistent_promotion_and_class_eviction(ninfer::DeviceContext& ctx,
                                                 ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto source   = pool.reserve(1);
    source.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, source, 83);
    const auto prompt_a = text_prompt({401, 402, 403, 404});
    const auto prompt_b = text_prompt({501, 502, 503, 504});
    const auto prompt_c = text_prompt({601, 602, 603, 604});
    const auto chain_a  = q36::detail::prefix_hash_chain(prompt_a);
    const auto chain_b  = q36::detail::prefix_hash_chain(prompt_b);
    const auto chain_c  = q36::detail::prefix_hash_chain(prompt_c);
    q36::detail::KVRamCache size_probe(8ULL << 20);
    if (capture_text_entry(size_probe, pool, source, prompt_a, ctx.stream) != 0) {
        return fail("eviction policy size probe failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t prefix_bytes = size_probe.snapshot().used_bytes;

    // A successful persistent consume promotes the live record immediately. Under pressure the
    // newer but still-probationary ordinary record must therefore leave first.
    q36::detail::KVRamCache promoted(prefix_bytes * 2 + 256);
    if (capture_text_entry(promoted, pool, source, prompt_a, ctx.stream, 0, 1, true) != 0 ||
        capture_text_entry(promoted, pool, source, prompt_b, ctx.stream, 0, 1) != 0) {
        ++failures;
        std::cerr << "promotion policy setup capture failed\n";
    } else {
        const auto hit = promoted.plan_match(prompt_a, chain_a);
        if (!hit) {
            ++failures;
            std::cerr << "persistent promotion source did not match\n";
        } else {
            promoted.claim(hit->entry_id);
            promoted.consume(hit->entry_id);
            (void)capture_text_entry(promoted, pool, source, prompt_c, ctx.stream, 0, 1);
            if (!promoted.plan_match(prompt_a, chain_a) ||
                promoted.plan_match(prompt_b, chain_b) ||
                !promoted.plan_match(prompt_c, chain_c)) {
                ++failures;
                std::cerr << "persistent consume did not protect the live record\n";
            }
        }
    }

    // Main class survives ordinary pressure even when it is the oldest record. Classifier is the
    // first victim; if no lower class is available, the final pass can still evict main.
    q36::detail::KVRamCache classes(prefix_bytes * 2 + 256);
    if (capture_text_entry(classes, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Main) != 0 ||
        capture_text_entry(classes, pool, source, prompt_b, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Classifier) != 0 ||
        capture_text_entry(classes, pool, source, prompt_c, ctx.stream, 0, 1) != 0) {
        ++failures;
        std::cerr << "class eviction setup capture failed\n";
    } else if (!classes.plan_match(prompt_a, chain_a) || classes.plan_match(prompt_b, chain_b) ||
               !classes.plan_match(prompt_c, chain_c)) {
        ++failures;
        std::cerr << "class-aware eviction did not preserve main over classifier\n";
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return failures;
}

// The dynamic-boundary capture (the LCP-driven checkpoint chosen by admission's boundary policy)
// has no demotion path once consume() promotes it, so its live count must be capped independently
// of ordinary byte-capacity pressure -- otherwise every burst leaves behind a permanently
// protected record. This pins: (1) a third capture evicts the coldest one even with room to spare,
// (2) a touched (recently hit) record survives over an untouched one of the same age, (3) a cap
// with every existing record claimed fails the capture cleanly instead of forcing an eviction.
int test_dynamic_boundary_cap(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto source   = pool.reserve(1);
    source.materialize_pages(1, ctx.stream);
    fill_logical_pages(pool, source, 71);
    const auto prompt_a = text_prompt({701, 702, 703, 704});
    const auto prompt_b = text_prompt({801, 802, 803, 804});
    const auto prompt_c = text_prompt({901, 902, 903, 904});
    const auto chain_a  = q36::detail::prefix_hash_chain(prompt_a);
    const auto chain_b  = q36::detail::prefix_hash_chain(prompt_b);
    const auto chain_c  = q36::detail::prefix_hash_chain(prompt_c);

    q36::detail::KVRamCache size_probe(8ULL << 20);
    if (capture_text_entry(size_probe, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
        0) {
        return fail("dynamic-boundary cap size probe failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t prefix_bytes = size_probe.snapshot().used_bytes;

    // (1) Ample room for all three; the cap (not byte pressure) must still evict the oldest.
    q36::detail::KVRamCache capped(prefix_bytes * 4 + 256);
    if (capture_text_entry(capped, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0 ||
        capture_text_entry(capped, pool, source, prompt_b, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0) {
        ++failures;
        std::cerr << "dynamic-boundary cap setup capture failed\n";
    } else {
        if (capture_text_entry(capped, pool, source, prompt_c, ctx.stream, 0, 1, false,
                               ninfer::RequestClass::Agents,
                               q36::detail::RamCaptureKind::DynamicBoundary) != 0) {
            ++failures;
            std::cerr << "third dynamic-boundary capture unexpectedly dropped\n";
        } else if (capped.plan_match(prompt_a, chain_a) || !capped.plan_match(prompt_b, chain_b) ||
                   !capped.plan_match(prompt_c, chain_c)) {
            ++failures;
            std::cerr << "dynamic-boundary cap did not evict the oldest record\n";
        }
    }

    // Admission-selected boundaries use PreserveExisting: the same cap must now reject the third
    // entry even while an ordinary cache capture would be allowed to replace the coldest one.
    q36::detail::KVRamCache preserved(prefix_bytes * 4 + 256);
    if (capture_text_entry(preserved, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::DynamicBoundary) != 0 ||
        capture_text_entry(preserved, pool, source, prompt_b, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::DynamicBoundary) != 0 ||
        capture_text_entry(preserved, pool, source, prompt_c, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::DynamicBoundary,
                           q36::detail::RamCapturePolicy::PreserveExisting) == 0 ||
        !preserved.plan_match(prompt_a, chain_a) || !preserved.plan_match(prompt_b, chain_b) ||
        preserved.plan_match(prompt_c, chain_c) || preserved.snapshot().evictions != 0) {
        ++failures;
        std::cerr << "non-evicting dynamic-boundary admission displaced a live record\n";
    }

    // Byte pressure follows the same contract independently of the DynamicBoundary count cap.
    q36::detail::KVRamCache byte_pressure(prefix_bytes * 2 + 256);
    if (capture_text_entry(byte_pressure, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::ActiveSibling) != 0 ||
        capture_text_entry(byte_pressure, pool, source, prompt_b, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::ActiveSibling) != 0 ||
        capture_text_entry(byte_pressure, pool, source, prompt_c, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents,
                           q36::detail::RamCaptureKind::ActiveSibling,
                           q36::detail::RamCapturePolicy::PreserveExisting) == 0 ||
        !byte_pressure.plan_match(prompt_a, chain_a) ||
        !byte_pressure.plan_match(prompt_b, chain_b) ||
        byte_pressure.plan_match(prompt_c, chain_c) || byte_pressure.snapshot().evictions != 0) {
        ++failures;
        std::cerr << "non-evicting byte-pressure admission displaced a live record\n";
    }

    // (2) A, B multi_claim so a consume() hit touches without erasing. Touch A, then a third
    // capture must evict B (colder), not A.
    q36::detail::KVRamCache touched(prefix_bytes * 4 + 256);
    if (capture_text_entry(touched, pool, source, prompt_a, ctx.stream, 0, 1, true,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0 ||
        capture_text_entry(touched, pool, source, prompt_b, ctx.stream, 0, 1, true,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0) {
        ++failures;
        std::cerr << "dynamic-boundary touch setup capture failed\n";
    } else {
        const auto hit = touched.plan_match(prompt_a, chain_a);
        if (!hit) {
            ++failures;
            std::cerr << "dynamic-boundary touch source did not match\n";
        } else {
            touched.claim(hit->entry_id);
            touched.consume(hit->entry_id);
            if (capture_text_entry(touched, pool, source, prompt_c, ctx.stream, 0, 1, false,
                                   ninfer::RequestClass::Agents,
                                   q36::detail::RamCaptureKind::DynamicBoundary) != 0) {
                ++failures;
                std::cerr << "post-touch dynamic-boundary capture unexpectedly dropped\n";
            } else if (!touched.plan_match(prompt_a, chain_a) ||
                       touched.plan_match(prompt_b, chain_b) ||
                       !touched.plan_match(prompt_c, chain_c)) {
                ++failures;
                std::cerr << "touch-on-hit did not protect the more recently used record\n";
            }
        }
    }

    // (3) Cap met and both existing records claimed: the capture must fail cleanly, not force an
    // eviction of state still in use.
    q36::detail::KVRamCache claimed(prefix_bytes * 4 + 256);
    if (capture_text_entry(claimed, pool, source, prompt_a, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0 ||
        capture_text_entry(claimed, pool, source, prompt_b, ctx.stream, 0, 1, false,
                           ninfer::RequestClass::Agents, q36::detail::RamCaptureKind::DynamicBoundary) !=
            0) {
        ++failures;
        std::cerr << "dynamic-boundary claimed-cap setup capture failed\n";
    } else {
        const auto match_a = claimed.plan_match(prompt_a, chain_a);
        const auto match_b = claimed.plan_match(prompt_b, chain_b);
        if (!match_a || !match_b) {
            ++failures;
            std::cerr << "dynamic-boundary claimed-cap setup did not match both records\n";
        } else {
            claimed.claim(match_a->entry_id);
            claimed.claim(match_b->entry_id);
            const std::size_t before_drops = claimed.snapshot().drops;
            if (capture_text_entry(claimed, pool, source, prompt_c, ctx.stream, 0, 1, false,
                                   ninfer::RequestClass::Agents,
                                   q36::detail::RamCaptureKind::DynamicBoundary) == 0 ||
                claimed.snapshot().drops == before_drops ||
                claimed.snapshot().entry_count != 2) {
                ++failures;
                std::cerr << "fully claimed dynamic-boundary cap forced an eviction instead of "
                             "dropping the new capture\n";
            }
            claimed.release(match_a->entry_id);
            claimed.release(match_b->entry_id);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return failures;
}

int test_unpack_consume_and_drop(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    int failures  = 0;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 13);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const auto prompt = text_prompt({40, 41, 42, 43});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.stream) != 0) {
        std::cerr << "unpack-consume capture failed\n";
        source.release();
        return 1;
    }
    const auto saved = cache.harvest_copy_seconds();
    if (saved.save < 0.0 || saved.load != 0.0 || cache.snapshot().save_seconds != saved.save) {
        std::cerr << "capture copy elapsed was not harvested as save\n";
        source.release();
        return 1;
    }
    source.release();

    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        std::cerr << "unpack-consume capture did not index\n";
        return 1;
    }

    auto full = pool.reserve(2);
    full.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget full_target;
    full_target.text           = &full;
    full_target.text_pool      = &pool;
    full_target.text_dst_pages = 2;
    full_target.stream         = ctx.stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, full_target);
    const auto loaded = cache.harvest_copy_seconds();
    if (loaded.save != 0.0 || loaded.load < 0.0 || cache.snapshot().load_seconds != loaded.load) {
        std::cerr << "restore copy elapsed was not harvested as load\n";
        ++failures;
    }
    cache.consume(match->entry_id);
    if (cache.snapshot().entry_count != 0 || cache.snapshot().restores != 1 ||
        cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt))) {
        std::cerr << "consume did not drop the RAM entry from the index\n";
        ++failures;
    }
    failures += expect_logical_pages(pool, full, 13, "unpack then consume");
    full.release();

    auto again = pool.reserve(2);
    again.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, again, 14);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    q36::detail::KVRamCache prefix_cache(8ULL << 20);
    if (capture_text_entry(prefix_cache, pool, again, prompt, ctx.stream) != 0) {
        std::cerr << "prefix-unpack capture failed\n";
        again.release();
        return failures + 1;
    }
    again.release();
    const auto prefix_match =
        prefix_cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!prefix_match) {
        std::cerr << "prefix-unpack capture did not index\n";
        return failures + 1;
    }
    auto prefix_dest = pool.reserve(2);
    prefix_dest.materialize_pages(1, ctx.stream);
    q36::detail::RamRestoreTarget prefix_target;
    prefix_target.text           = &prefix_dest;
    prefix_target.text_pool      = &pool;
    prefix_target.text_dst_pages = 1;
    prefix_target.stream         = ctx.stream;
    prefix_cache.claim(prefix_match->entry_id);
    (void)prefix_cache.unpack_device(prefix_match->entry_id, prefix_target);
    prefix_cache.consume(prefix_match->entry_id);
    failures += expect_logical_pages(pool, prefix_dest, 14, "checkpoint-sized unpack prefix");
    prefix_dest.release();

    q36::detail::KVRamCache tiny(256);
    auto tiny_source = pool.reserve(2);
    tiny_source.materialize_pages(2, ctx.stream);
    const auto drops_before = tiny.snapshot().drops;
    if (capture_text_entry(tiny, pool, tiny_source, prompt, ctx.stream) == 0 ||
        tiny.snapshot().drops <= drops_before || tiny.snapshot().entry_count != 0) {
        std::cerr << "over-capacity capture did not drop\n";
        ++failures;
    }
    tiny_source.release();
    return failures;
}

int expect_logical_page(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        std::size_t logical_index, unsigned char seed, const char* label) {
    const auto pages = allocation.page_ids();
    if (logical_index >= pages.size()) {
        std::cerr << label << " logical page index out of range\n";
        return 1;
    }
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        const unsigned char value = static_cast<unsigned char>(
            seed + plane * 17U + static_cast<unsigned>(logical_index) + 1U);
        const std::size_t begin =
            static_cast<std::size_t>(pages[logical_index] * tensor.nb[3]);
        for (std::int64_t byte = 0; byte < tensor.nb[3]; ++byte) {
            if (host[begin + static_cast<std::size_t>(byte)] != value) {
                std::cerr << label << " logical page " << logical_index << " plane " << plane
                          << " mismatch\n";
                return 1;
            }
        }
    }
    return 0;
}

int test_frontier_beats_checkpoint(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt = text_prompt({10, 11, 12, 13});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt, ctx.stream, 2) != 0) {
        alloc.release();
        return fail("frontier-beats-checkpoint capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    alloc.release();
    if (!match || match->reuse != ninfer::PrefixReusePath::AppendAtFrontier ||
        match->reuse_base != prompt.token_ids.size()) {
        std::cerr << "frontier match lost to the shorter checkpoint\n";
        return 1;
    }
    return 0;
}

int test_asymmetric_fifo(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    auto small          = pool.reserve(1);
    small.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, small, prompt_a, ctx.stream) != 0) {
        small.release();
        return fail("asymmetric FIFO probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t small_bytes = probe.snapshot().used_bytes;
    auto large                    = pool.reserve(2);
    large.materialize_pages(2, ctx.stream);
    q36::detail::KVRamCache large_probe(8ULL << 20);
    if (capture_text_entry(large_probe, pool, large, prompt_c, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO large probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t large_bytes = large_probe.snapshot().used_bytes;
    if (large_bytes <= small_bytes) {
        small.release();
        large.release();
        return fail("two-page capture was not larger than one-page capture");
    }
    q36::detail::KVRamCache cache(small_bytes * 2 + 256);
    if (capture_text_entry(cache, pool, small, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_b, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO small captures failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto evictions_before = cache.snapshot().evictions;
    const auto drops_before     = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, large, prompt_c, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("asymmetric FIFO large capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    int failures = 0;
    if (cache.snapshot().evictions < evictions_before + 2 || cache.snapshot().drops != drops_before ||
        cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c))) {
        std::cerr << "asymmetric FIFO did not evict both small entries: evictions="
                  << cache.snapshot().evictions - evictions_before
                  << " drops=" << cache.snapshot().drops - drops_before
                  << " entries=" << cache.snapshot().entry_count << '\n';
        ++failures;
    }
    small.release();
    large.release();
    return failures;
}

int test_fifo_consume_middle(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    auto alloc          = pool.reserve(1);
    alloc.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.stream) != 0) {
        alloc.release();
        return fail("FIFO middle first capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t bytes_a = cache.snapshot().used_bytes;
    if (capture_text_entry(cache, pool, alloc, prompt_b, ctx.stream) != 0) {
        alloc.release();
        return fail("FIFO middle second capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t bytes_ab = cache.snapshot().used_bytes;
    if (capture_text_entry(cache, pool, alloc, prompt_c, ctx.stream) != 0) {
        alloc.release();
        return fail("FIFO middle third capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t bytes_abc = cache.snapshot().used_bytes;
    const std::size_t bytes_c   = bytes_abc - bytes_ab;
    const auto match_b          = cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b) {
        alloc.release();
        return fail("FIFO middle did not match B");
    }
    cache.claim(match_b->entry_id);
    cache.consume(match_b->entry_id);
    int failures = 0;
    if (cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c)) ||
        cache.snapshot().entry_count != 2 || cache.snapshot().restores != 1 ||
        cache.snapshot().used_bytes != bytes_a + bytes_c) {
        std::cerr << "FIFO consume did not drop the middle host resident: entries="
                  << cache.snapshot().entry_count << " used=" << cache.snapshot().used_bytes
                  << " expected=" << bytes_a + bytes_c << '\n';
        ++failures;
    }
    alloc.release();
    return failures;
}

int test_fifo_evict_after_middle_consume(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    const auto prompt_a = text_prompt({10, 11, 12, 13});
    const auto prompt_b = text_prompt({20, 21, 22, 23});
    const auto prompt_c = text_prompt({30, 31, 32, 33});
    const auto prompt_d = text_prompt({40, 41, 42, 43});
    auto small          = pool.reserve(1);
    small.materialize_pages(1, ctx.stream);
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, small, prompt_a, ctx.stream) != 0) {
        small.release();
        return fail("FIFO hole probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t small_bytes = probe.snapshot().used_bytes;
    auto large                    = pool.reserve(2);
    large.materialize_pages(2, ctx.stream);
    q36::detail::KVRamCache large_probe(8ULL << 20);
    if (capture_text_entry(large_probe, pool, large, prompt_d, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole large probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t large_bytes = large_probe.snapshot().used_bytes;
    if (large_bytes <= small_bytes || large_bytes > small_bytes * 2) {
        small.release();
        large.release();
        return fail("FIFO hole large capture does not force exactly one eviction after a hole");
    }
    q36::detail::KVRamCache cache(small_bytes * 3 + 256);
    if (capture_text_entry(cache, pool, small, prompt_a, ctx.stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_b, ctx.stream) != 0 ||
        capture_text_entry(cache, pool, small, prompt_c, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole small captures failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto match_b = cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b));
    if (!match_b) {
        small.release();
        large.release();
        return fail("FIFO hole did not match B");
    }
    cache.claim(match_b->entry_id);
    cache.consume(match_b->entry_id);
    const auto evictions_before = cache.snapshot().evictions;
    const auto drops_before     = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, large, prompt_d, ctx.stream) != 0) {
        small.release();
        large.release();
        return fail("FIFO hole large capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    int failures = 0;
    if (cache.snapshot().evictions != evictions_before + 1 || cache.snapshot().drops != drops_before ||
        cache.snapshot().entry_count != 2 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b)) ||
        !cache.plan_match(prompt_c, q36::detail::prefix_hash_chain(prompt_c)) ||
        !cache.plan_match(prompt_d, q36::detail::prefix_hash_chain(prompt_d))) {
        std::cerr << "FIFO hole did not evict the oldest remaining resident: evictions="
                  << cache.snapshot().evictions - evictions_before
                  << " drops=" << cache.snapshot().drops - drops_before
                  << " entries=" << cache.snapshot().entry_count << '\n';
        ++failures;
    }
    small.release();
    large.release();
    return failures;
}

int test_prefix_unpack_preserves_tail(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 31);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto prompt = text_prompt({40, 41, 42, 43});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.stream) != 0) {
        source.release();
        return fail("prefix-tail capture failed");
    }
    source.release();
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("prefix-tail capture did not index"); }
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, dest, 99);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 1;
    target.stream         = ctx.stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    int failures = 0;
    failures += expect_logical_page(pool, dest, 0, 31, "prefix unpack page 0");
    failures += expect_logical_page(pool, dest, 1, 99, "prefix unpack must not write dest page 1");
    dest.release();
    return failures;
}

int test_consume_reaps_for_next_capture(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 61);
    const auto prompt_a = text_prompt({60, 61, 62, 63});
    const auto prompt_b = text_prompt({70, 71, 72, 73});
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, source, prompt_a, ctx.stream) != 0) {
        source.release();
        return fail("consume-reap probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    q36::detail::KVRamCache cache(entry_bytes + 256);
    if (capture_text_entry(cache, pool, source, prompt_a, ctx.stream) != 0) {
        source.release();
        return fail("consume-reap first capture failed");
    }
    const auto match = cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!match) {
        source.release();
        return fail("consume-reap first capture did not index");
    }
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    const auto drops_before     = cache.snapshot().drops;
    const auto evictions_before = cache.snapshot().evictions;
    if (capture_text_entry(cache, pool, source, prompt_b, ctx.stream) != 0) {
        source.release();
        dest.release();
        return fail("consume-reap second capture failed");
    }
    int failures = 0;
    if (cache.snapshot().drops != drops_before || cache.snapshot().evictions != evictions_before ||
        cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a)) ||
        !cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b))) {
        std::cerr << "consume did not return host budget to the next capture\n";
        ++failures;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    failures += expect_logical_pages(pool, dest, 61, "consume-reap dest after next capture");
    source.release();
    dest.release();
    return failures;
}

int test_event_overlap_unpack(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 44);
    const auto prompt = text_prompt({50, 51, 52, 53});
    q36::detail::KVRamCache cache(8ULL << 20);
    if (capture_text_entry(cache, pool, source, prompt, ctx.stream) != 0) {
        source.release();
        return fail("overlap capture failed");
    }
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        source.release();
        return fail("overlap capture did not index before D2H completion");
    }
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    q36::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = 2;
    target.stream         = ctx.stream;
    cache.claim(match->entry_id);
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const int failures = expect_logical_pages(pool, dest, 44, "event-overlapped unpack");
    dest.release();
    return failures;
}

int test_irregular_page_major_runs(ninfer::DeviceContext& ctx) {
    auto plan = plan_paged_cache(8, 8, 2,
                                 {{ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::I8, 64, 2},
                                  {ninfer::DType::FP16, 1, 2},
                                  {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, plan.layout);
    std::vector<ninfer::PagedKVAllocation> held;
    for (int i = 0; i < 8; ++i) {
        auto page = pool.reserve(1);
        page.materialize_pages(1, ctx.stream);
        held.push_back(std::move(page));
    }
    held[0].release();
    held[2].release();
    held[3].release();
    held[5].release();
    held[7].release();
    auto source = pool.reserve(5);
    source.materialize_pages(5, ctx.stream);
    const auto ids = source.page_ids();
    bool mixed     = false;
    if (ids.size() >= 3) {
        const std::int32_t delta = ids[1] - ids[0];
        for (std::size_t i = 2; i < ids.size(); ++i) {
            if (ids[i] - ids[i - 1] != delta) {
                mixed = true;
                break;
            }
        }
    }
    if (!mixed) {
        std::cerr << "irregular PageMajor test did not produce mixed strides:";
        for (std::int32_t id : ids) { std::cerr << ' ' << id; }
        std::cerr << '\n';
        source.release();
        return 1;
    }
    fill_logical_pages(pool, source, 73);
    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, 5);
    ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
    void* image = host.try_alloc(image_bytes, 256);
    ninfer::pack_paged_kv_allocation_to_host(source, pool, 5, image, ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    source.release();
    auto dest = pool.reserve(5);
    dest.materialize_pages(5, ctx.stream);
    ninfer::unpack_paged_kv_allocation_from_host(dest, pool, image, 5, 5, ctx.stream);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const int failures = expect_logical_pages(pool, dest, 73, "irregular PageMajor runs");
    dest.release();
    return failures;
}

int test_restore_throw_then_replay(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 81);
    const auto prompt = text_prompt({80, 81, 82, 83});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xb1);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xb2);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});
    q36::detail::RamCaptureSource cap;
    cap.execution_frontier        = static_cast<std::uint32_t>(prompt.token_ids.size());
    cap.ledger_frontier           = static_cast<std::uint32_t>(tokens);
    cap.text_kv_valid             = cap.execution_frontier;
    cap.tail_hidden_valid         = true;
    cap.rewrite_valid             = true;
    cap.rewrite_kind              = q36::RewriteCheckpointKind::ResponseReplay;
    cap.rewrite_frontier          = 2;
    cap.hash_c_valid              = true;
    cap.ledger                    = retained.token_ids;
    cap.identity                  = &identity;
    cap.hash_f                    = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                                cap.execution_frontier);
    cap.hash_c                    = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    cap.text                      = &source;
    cap.text_pool                 = &pool;
    cap.text_pages                = source.mapped_page_count();
    cap.tail_hidden               = &hidden;
    cap.rewrite_checkpoint_hidden = &rewrite;
    cap.stream                    = ctx.stream;
    q36::detail::KVRamCache cache(8ULL << 20);
    if (!cache.capture(cap)) {
        source.release();
        return fail("throw-after-H2D capture failed");
    }
    source.release();
    auto dest = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_bad_buf(64);
    rewrite_bad_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor rewrite_bad(rewrite_bad_buf.p, ninfer::DType::U8, {64});
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) {
        dest.release();
        return fail("throw-after-H2D capture did not match");
    }
    q36::detail::RamRestoreTarget bad;
    bad.text                      = &dest;
    bad.text_pool                 = &pool;
    bad.text_dst_pages            = 2;
    bad.tail_hidden               = &hidden_out;
    bad.rewrite_checkpoint_hidden = &rewrite_bad;
    bad.stream                    = ctx.stream;
    cache.claim(match->entry_id);
    bool threw = false;
    try {
        (void)cache.unpack_device(match->entry_id, bad);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    cache.release(match->entry_id);
    dest.release();
    if (!threw) { return fail("mismatched rewrite hidden did not throw after KV H2D"); }

    auto dest2 = pool.reserve(2);
    dest2.materialize_pages(2, ctx.stream);
    ninfer::DeviceBuffer rewrite_ok_buf(128);
    rewrite_ok_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor rewrite_ok(rewrite_ok_buf.p, ninfer::DType::U8, {128});
    hidden_out_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    q36::detail::RamRestoreTarget ok;
    ok.text                      = &dest2;
    ok.text_pool                 = &pool;
    ok.text_dst_pages            = 2;
    ok.tail_hidden               = &hidden_out;
    ok.rewrite_checkpoint_hidden = &rewrite_ok;
    ok.stream                    = ctx.stream;
    cache.claim(match->entry_id);
    const auto host = cache.unpack_device(match->entry_id, ok);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    int failures = 0;
    if (!host.rewrite_valid || host.rewrite_kind != q36::RewriteCheckpointKind::ResponseReplay ||
        host.rewrite_frontier != 2) {
        std::cerr << "response-checkpoint host metadata mismatch after failed unpack\n";
        ++failures;
    }
    failures += expect_logical_pages(pool, dest2, 81, "replay after thrown unpack");
    dest2.release();
    return failures;
}

int test_destructor_with_inflight_copies(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto source   = pool.reserve(2);
    source.materialize_pages(2, ctx.stream);
    fill_logical_pages(pool, source, 91);
    const auto prompt = text_prompt({90, 91, 92, 93});
    auto dest         = pool.reserve(2);
    dest.materialize_pages(2, ctx.stream);
    {
        q36::detail::KVRamCache cache(8ULL << 20);
        if (capture_text_entry(cache, pool, source, prompt, ctx.stream) != 0) {
            source.release();
            dest.release();
            return fail("destructor-inflight capture failed");
        }
        const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
        if (!match) {
            source.release();
            dest.release();
            return fail("destructor-inflight capture did not match");
        }
        q36::detail::RamRestoreTarget target;
        target.text           = &dest;
        target.text_pool      = &pool;
        target.text_dst_pages = 2;
        target.stream         = ctx.stream;
        cache.claim(match->entry_id);
        (void)cache.unpack_device(match->entry_id, target);
        cache.consume(match->entry_id);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const int failures = expect_logical_pages(pool, dest, 91, "destructor drained in-flight H2D");
    source.release();
    dest.release();
    return failures;
}

int test_spill_drop_keeps_indexed_source(ninfer::DeviceContext& ctx, ninfer::PagedKVPool& pool) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto alloc    = pool.reserve(2);
    alloc.materialize_pages(1, ctx.stream);
    const auto prompt_a = text_prompt({11, 12, 13, 14});
    const auto prompt_b = text_prompt({21, 22, 23, 24});
    q36::detail::KVRamCache probe(8ULL << 20);
    if (capture_text_entry(probe, pool, alloc, prompt_a, ctx.stream) != 0) {
        alloc.release();
        return fail("spill-drop probe capture failed");
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const std::size_t entry_bytes = probe.snapshot().used_bytes;
    q36::detail::KVRamCache cache(entry_bytes + 256);
    if (capture_text_entry(cache, pool, alloc, prompt_a, ctx.stream) != 0) {
        alloc.release();
        return fail("spill-drop first capture failed");
    }
    const auto first = cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a));
    if (!first) {
        alloc.release();
        return fail("spill-drop first capture did not index");
    }
    cache.claim(first->entry_id);
    const auto drops_before = cache.snapshot().drops;
    if (capture_text_entry(cache, pool, alloc, prompt_b, ctx.stream) == 0) {
        alloc.release();
        return fail("pinned one-entry budget captured a second bundle");
    }
    int failures = 0;
    if (cache.snapshot().drops != drops_before + 1 || cache.snapshot().entry_count != 1 ||
        cache.plan_match(prompt_b, q36::detail::prefix_hash_chain(prompt_b))) {
        std::cerr << "spill drop indexed the dropped victim\n";
        ++failures;
    }
    cache.release(first->entry_id);
    if (!cache.plan_match(prompt_a, q36::detail::prefix_hash_chain(prompt_a))) {
        std::cerr << "spill drop lost the pinned source\n";
        ++failures;
    }
    alloc.release();
    return failures;
}

int test_full_state_image(ninfer::DeviceContext& ctx) {
    namespace q36 = ninfer::targets::qwen3_6;
    auto text_plan =
        plan_paged_cache(4, 4, 2,
                         {{ninfer::DType::I8, 16, 2},
                          {ninfer::DType::I8, 16, 2},
                          {ninfer::DType::FP16, 1, 2},
                          {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena text_arena(text_plan.bytes);
    ninfer::PagedKVPool text_pool({text_arena.base(), text_arena.capacity()}, text_plan.layout);
    auto backend_plan =
        plan_paged_cache(4, 4, 1, {{ninfer::DType::I8, 8, 1}, {ninfer::DType::I8, 8, 1}});
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 3,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout = ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 16, 2, 8, 2);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache dflash_local({cyclic_arena.base(), cyclic_arena.capacity()},
                                       cyclic_layout);
    ninfer::LayoutBuilder cyclic_ckpt_builder;
    const auto cyclic_ckpt_layout =
        ninfer::plan_cyclic_kv_cache(cyclic_ckpt_builder, 2, 16, 2, 8, 2);
    ninfer::DeviceArena cyclic_ckpt_arena(cyclic_ckpt_builder.finish(256));
    ninfer::CyclicKVCache dflash_ckpt({cyclic_ckpt_arena.base(), cyclic_ckpt_arena.capacity()},
                                      cyclic_ckpt_layout);

    auto text = text_pool.reserve(2);
    text.materialize_pages(2, ctx.stream);
    fill_logical_pages(text_pool, text, 7);
    auto backend = backend_pool.reserve(2);
    backend.materialize_pages(1, ctx.stream);
    fill_logical_pages(backend_pool, backend, 8);

    std::vector<unsigned char> conv_cur(gdn.conv_slot(0, 0).bytes());
    std::vector<unsigned char> conv_ckpt(gdn.conv_slot(0, 1).bytes());
    for (std::size_t i = 0; i < conv_cur.size(); ++i) {
        conv_cur[i]  = static_cast<unsigned char>(i + 1);
        conv_ckpt[i] = static_cast<unsigned char>(i + 9);
    }
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(0, 0).data, conv_cur.data(), conv_cur.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(2, 0).data, conv_cur.data(), conv_cur.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gdn.conv_slot(0, 1).data, conv_ckpt.data(), conv_ckpt.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> rec_cur(gdn.recurrent_slot(0, 0).bytes(), 0x21);
    std::vector<unsigned char> rec_ckpt(gdn.recurrent_slot(0, 1).bytes(), 0x22);
    CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(1, 0).data, rec_cur.data(), rec_cur.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gdn.recurrent_slot(1, 1).data, rec_ckpt.data(), rec_ckpt.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());

    ninfer::CyclicKVCacheLayerView local_view = dflash_local.layer_view(0);
    std::vector<unsigned char> k_local(local_view.k.slice(3, 0, 1).bytes(), 0x3c);
    std::vector<unsigned char> v_local(local_view.v.slice(3, 0, 1).bytes(), 0x3d);
    CUDA_CHECK(cudaMemcpy(local_view.k.slice(3, 0, 1).data, k_local.data(), k_local.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(local_view.v.slice(3, 0, 1).data, v_local.data(), v_local.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::CyclicKVCacheLayerView ckpt_view = dflash_ckpt.layer_view(0);
    std::vector<unsigned char> k_ckpt(ckpt_view.k.slice(3, 0, 1).bytes(), 0x4c);
    std::vector<unsigned char> v_ckpt(ckpt_view.v.slice(3, 0, 1).bytes(), 0x4d);
    CUDA_CHECK(cudaMemcpy(ckpt_view.k.slice(3, 0, 1).data, k_ckpt.data(), k_ckpt.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ckpt_view.v.slice(3, 0, 1).data, v_ckpt.data(), v_ckpt.size(),
                           cudaMemcpyHostToDevice));
                           CUDA_CHECK(cudaDeviceSynchronize());

    ninfer::DeviceBuffer hidden_buf(128);
    hidden_buf.fill(0xa1);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_buf(128);
    rewrite_buf.fill(0xa2);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {128});

    const auto prompt = text_prompt({1, 2, 3, 4});
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    q36::detail::ResidentPrefixIdentity identity;
    identity.assign(retained);
    q36::detail::RamCaptureSource source;
    source.execution_frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier    = static_cast<std::uint32_t>(tokens);
    source.rope_delta         = 7;
    source.text_kv_valid      = source.execution_frontier;
    source.mtp_kv_valid       = 3;
    source.tail_hidden_valid  = true;
    source.rewrite_valid      = true;
    source.rewrite_kind       = q36::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier   = 2;
    source.hash_c_valid       = true;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = q36::detail::prefix_hash_at(retained.token_ids, identity,
                                                            source.execution_frontier);
    source.hash_c             = q36::detail::prefix_hash_at(retained.token_ids, identity, 2);
    source.text               = &text;
    source.text_pool          = &text_pool;
    source.text_pages         = text.mapped_page_count();
    source.backend            = &backend;
    source.backend_pool       = &backend_pool;
    source.backend_pages      = backend.mapped_page_count();
    source.gdn                = &gdn;
    source.gdn_current_slot   = 0;
    source.gdn_checkpoint_slot = 1;
    source.tail_hidden        = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.dflash_local       = &dflash_local;
    source.dflash_checkpoint  = &dflash_ckpt;
    source.dflash_lane        = 0;
    source.stream             = ctx.stream;
    q36::detail::KVRamCache cache(16ULL << 20);
    const std::size_t expected_capture_bytes = cache.capture_bytes(source);
    if (!cache.capture(source)) { return fail("full-state capture failed"); }
    if (cache.snapshot().used_bytes != expected_capture_bytes) {
        return fail("full-state preflight size disagrees with the captured record");
    }

    auto text_dest = text_pool.reserve(2);
    text_dest.materialize_pages(2, ctx.stream);
    auto backend_dest = backend_pool.reserve(2);
    backend_dest.materialize_pages(1, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(128);
    hidden_out_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {128});
    ninfer::DeviceBuffer rewrite_out_buf(128);
    rewrite_out_buf.fill(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Tensor rewrite_out(rewrite_out_buf.p, ninfer::DType::U8, {128});
    const auto match = cache.plan_match(prompt, q36::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("full-state capture did not match"); }
    q36::detail::RamRestoreTarget target;
    target.text                    = &text_dest;
    target.text_pool               = &text_pool;
    target.text_dst_pages          = 2;
    target.backend                 = &backend_dest;
    target.backend_pool            = &backend_pool;
    target.backend_dst_pages       = 1;
    target.gdn                     = &gdn;
    target.gdn_current_slot        = 2;
    target.gdn_checkpoint_slot     = 3;
    target.tail_hidden             = &hidden_out;
    target.rewrite_checkpoint_hidden = &rewrite_out;
    target.dflash_local            = &dflash_local;
    target.dflash_checkpoint       = &dflash_ckpt;
    target.dflash_lane             = 1;
    target.stream                  = ctx.stream;
    cache.claim(match->entry_id);
    const q36::detail::RamRestoredHost host = cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    int failures = 0;
    if (host.rope_delta != 7 || host.mtp_kv_valid != 3 || !host.backend_image_present ||
        !host.rewrite_valid || host.rewrite_frontier != 2 ||
        host.ledger.size() != tokens) {
        std::cerr << "full-state host metadata mismatch\n";
        ++failures;
    }
    failures += expect_logical_pages(text_pool, text_dest, 7, "full-state text KV");
    failures += expect_logical_pages(backend_pool, backend_dest, 8, "full-state backend KV");
    std::vector<unsigned char> conv_out(conv_cur.size());
    CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 2).data, conv_out.size(),
                           cudaMemcpyDeviceToHost));
    if (conv_out != conv_cur) {
        std::cerr << "full-state GDN conv current did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 3).data, conv_out.size(),
                           cudaMemcpyDeviceToHost));
    if (conv_out != conv_ckpt) {
        std::cerr << "full-state GDN conv checkpoint did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> rec_out(rec_cur.size());
    CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 2).data, rec_out.size(),
                           cudaMemcpyDeviceToHost));
    if (rec_out != rec_cur) {
        std::cerr << "full-state GDN recurrent current did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 3).data, rec_out.size(),
                           cudaMemcpyDeviceToHost));
    if (rec_out != rec_ckpt) {
        std::cerr << "full-state GDN recurrent checkpoint did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> hidden_host(128);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(128, 0xa1)) {
        std::cerr << "full-state tail hidden did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), rewrite_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(128, 0xa2)) {
        std::cerr << "full-state rewrite hidden did not round-trip\n";
        ++failures;
    }
    std::vector<unsigned char> k_out(k_local.size());
    CUDA_CHECK(cudaMemcpy(k_out.data(), dflash_local.layer_view(0).k.slice(3, 1, 1).data,
                           k_out.size(), cudaMemcpyDeviceToHost));
    std::vector<unsigned char> v_out(v_local.size());
    CUDA_CHECK(cudaMemcpy(v_out.data(), dflash_local.layer_view(0).v.slice(3, 1, 1).data,
                           v_out.size(), cudaMemcpyDeviceToHost));
    if (k_out != k_local || v_out != v_local) {
        std::cerr << "full-state DFlash local lane did not round-trip\n";
        ++failures;
    }
    CUDA_CHECK(cudaMemcpy(k_out.data(), dflash_ckpt.layer_view(0).k.slice(3, 1, 1).data, k_out.size(),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_out.data(), dflash_ckpt.layer_view(0).v.slice(3, 1, 1).data, v_out.size(),
                           cudaMemcpyDeviceToHost));
    if (k_out != k_ckpt || v_out != v_ckpt) {
        std::cerr << "full-state DFlash checkpoint lane did not round-trip\n";
        ++failures;
    }
    text.release();
    backend.release();
    text_dest.release();
    backend_dest.release();
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);

    auto keep = paged_pool.reserve(3);
    keep.materialize_pages(3);
    failures += round_trip_pool(ctx, paged_pool, 6, 6, 3, "INT8 fragmented PageMajor");
    auto scrambled = paged_pool.reserve(3);
    scrambled.materialize_pages(3);
    // Physical IDs are assigned from the free list; force a logical order that is not physical
    // order by packing the live mapping after a fragmented take. The round-trip above already
    // used non-contiguous IDs {0,1,2,6,7,8}-style after the reserved 3-page keep.
    keep.release();
    scrambled.release();

    auto consecutive_physical = paged_pool.reserve(4);
    consecutive_physical.materialize_pages(4);
    fill_logical_pages(paged_pool, consecutive_physical, 9);
    {
        auto ids = consecutive_physical.page_ids();
        if (ids.size() >= 3 && ids[1] + 1 == ids[2] && ids[0] + 1 != ids[1]) {
            // logical 1..2 is a physical run that is not a prefix of the logical sequence
        }
        const std::size_t image_bytes =
            ninfer::paged_kv_host_image_bytes(paged_pool, consecutive_physical.mapped_page_count());
        ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
        void* image = host.try_alloc(image_bytes, 256);
        ninfer::pack_paged_kv_allocation_to_host(consecutive_physical, paged_pool, 4, image,
                                                 ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        consecutive_physical.release();
        auto restored = paged_pool.reserve(4);
        restored.materialize_pages(4);
        ninfer::unpack_paged_kv_allocation_from_host(restored, paged_pool, image, 4, 4, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        failures += expect_logical_pages(paged_pool, restored, 9, "no zero_pages sort");
        restored.release();
    }

    auto partial = paged_pool.reserve(3);
    partial.materialize_tokens(70);
    failures += expect_size(partial.mapped_page_count(), 2, "partial tail page count");
    fill_logical_pages(paged_pool, partial, 11);
    {
        const std::uint32_t captured = partial.mapped_page_count();
        const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(paged_pool, captured);
        ninfer::HostPinnedArena host(std::max<std::size_t>(image_bytes, 256));
        void* image = host.try_alloc(image_bytes, 256);
        ninfer::pack_paged_kv_allocation_to_host(partial, paged_pool, captured, image, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        partial.release();
        auto restored = paged_pool.reserve(3);
        restored.materialize_pages(2);
        ninfer::unpack_paged_kv_allocation_from_host(restored, paged_pool, image, captured, 2,
                                                     ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        failures += expect_logical_pages(paged_pool, restored, 11, "partial tail page");
        restored.release();
    }

    auto head_major_plan = plan_paged_cache(8, 8, 1, {{ninfer::DType::BF16, 16, 4}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += round_trip_pool(ctx, head_major_pool, 3, 3, 21, "HeadMajor Hkv>1 count>1");

    auto bf16_plan = plan_paged_cache(8, 8, 2,
                                      {{ninfer::DType::BF16, 32, 2}, {ninfer::DType::BF16, 32, 2}});
    ninfer::DeviceArena bf16_arena(bf16_plan.bytes);
    ninfer::PagedKVPool bf16_pool({bf16_arena.base(), bf16_arena.capacity()}, bf16_plan.layout);
    failures += round_trip_pool(ctx, bf16_pool, 3, 3, 17, "BF16 PageMajor");

    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 2,
                      .conv_channels  = 8,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 4,
                      .key_head_dim   = 3,
                      .slot_count     = 4,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    {
        const ninfer::Tensor conv = gdn.conv_slot(0, 0);
        std::vector<unsigned char> pattern(conv.bytes());
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            pattern[i] = static_cast<unsigned char>(i * 3 + 1);
        }
        CUDA_CHECK(cudaMemcpy(conv.data, pattern.data(), pattern.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        const ninfer::Tensor rec = gdn.recurrent_slot(1, 0);
        std::vector<unsigned char> rec_pattern(rec.bytes(), 0xab);
        CUDA_CHECK(
            cudaMemcpy(rec.data, rec_pattern.data(), rec_pattern.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> conv_host(gdn.conv_host_image_bytes());
        std::vector<unsigned char> rec_host(gdn.recurrent_host_image_bytes());
        gdn.pack_slot_to_host(0, conv_host.data(), rec_host.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        gdn.unpack_slot_from_host(2, conv_host.data(), rec_host.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        std::vector<unsigned char> conv_out(conv.bytes());
        CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(0, 2).data, conv_out.size(),
                              cudaMemcpyDeviceToHost));
        if (conv_out != pattern) {
            ++failures;
            std::cerr << "GDN conv host round-trip across slots failed\n";
        }
        std::vector<unsigned char> rec_out(rec.bytes());
        CUDA_CHECK(cudaMemcpy(rec_out.data(), gdn.recurrent_slot(1, 2).data, rec_out.size(),
                              cudaMemcpyDeviceToHost));
        if (rec_out != rec_pattern) {
            ++failures;
            std::cerr << "GDN recurrent host round-trip across slots failed\n";
        }
    }

    ninfer::LayoutBuilder cyclic_builder;
    const auto cyclic_layout =
        ninfer::plan_cyclic_kv_cache(cyclic_builder, 2, 32, 2, 8, 3);
    ninfer::DeviceArena cyclic_arena(cyclic_builder.finish(256));
    ninfer::CyclicKVCache cyclic({cyclic_arena.base(), cyclic_arena.capacity()}, cyclic_layout);
    {
        ninfer::CyclicKVCacheLayerView layer = cyclic.layer_view(0);
        std::vector<unsigned char> k_pattern(layer.k.slice(3, 0, 1).bytes(), 0x3c);
        CUDA_CHECK(cudaMemcpy(layer.k.slice(3, 0, 1).data, k_pattern.data(), k_pattern.size(),
                              cudaMemcpyHostToDevice));
                              CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> v_pattern(layer.v.slice(3, 0, 1).bytes(), 0x4d);
        CUDA_CHECK(cudaMemcpy(layer.v.slice(3, 0, 1).data, v_pattern.data(), v_pattern.size(),
                              cudaMemcpyHostToDevice));
                              CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> host(cyclic.lane_host_bytes());
        cyclic.copy_lane_to_host(0, host.data(), ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        cyclic.copy_lane_from_host(host.data(), 2, ctx.stream);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        std::vector<unsigned char> k_out(k_pattern.size());
        CUDA_CHECK(cudaMemcpy(k_out.data(), cyclic.layer_view(0).k.slice(3, 2, 1).data, k_out.size(),
                              cudaMemcpyDeviceToHost));
        std::vector<unsigned char> v_out(v_pattern.size());
        CUDA_CHECK(cudaMemcpy(v_out.data(), cyclic.layer_view(0).v.slice(3, 2, 1).data, v_out.size(),
                              cudaMemcpyDeviceToHost));
        if (k_out != k_pattern || v_out != v_pattern) {
            ++failures;
            std::cerr << "DFlash cyclic host round-trip across lanes failed\n";
        }
    }

    failures += test_kv_ram_index(ctx, paged_pool);
    failures += test_capture_extent_and_eviction_policy(ctx, paged_pool);
    failures += test_persistent_promotion_and_class_eviction(ctx, paged_pool);
    failures += test_dynamic_boundary_cap(ctx, paged_pool);
    failures += test_unpack_consume_and_drop(ctx, paged_pool);
    failures += test_frontier_beats_checkpoint(ctx, paged_pool);
    failures += test_asymmetric_fifo(ctx, paged_pool);
    failures += test_fifo_consume_middle(ctx, paged_pool);
    failures += test_fifo_evict_after_middle_consume(ctx, paged_pool);
    failures += test_prefix_unpack_preserves_tail(ctx, paged_pool);
    failures += test_consume_reaps_for_next_capture(ctx, paged_pool);
    failures += test_event_overlap_unpack(ctx, paged_pool);
    failures += test_irregular_page_major_runs(ctx);
    failures += test_restore_throw_then_replay(ctx, paged_pool);
    failures += test_destructor_with_inflight_copies(ctx, paged_pool);
    failures += test_spill_drop_keeps_indexed_source(ctx, paged_pool);
    failures += test_full_state_image(ctx);
    // Guarded: a test that throws during its own setup would otherwise abort the process with no
    // diagnostic at all, which reads exactly like a clean pass to anything checking only for
    // absence of failure output.
    try {
        failures += test_residual_slot_round_trip(ctx);
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "residual slot round-trip threw: " << error.what() << '\n';
    }

    return failures == 0 ? 0 : fail("kv ram cache core test failed");
}
