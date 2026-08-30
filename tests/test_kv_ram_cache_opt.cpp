#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace {

constexpr int kWarmup = 2;
constexpr int kIters  = 8;
constexpr double kMinFractionOfPinnedMemcpy = 0.50;

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

ninfer::targets::qwen3_6::PreparedPromptData text_prompt(std::vector<ninfer::TokenId> tokens) {
    ninfer::targets::qwen3_6::PreparedPromptData prompt;
    prompt.token_ids = std::move(tokens);
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

void fill_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                        unsigned char seed) {
    const auto pages = allocation.page_ids();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
            std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                      host.begin() + static_cast<std::ptrdiff_t>(begin + tensor.nb[3]), value);
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
}

int expect_logical_pages(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                         unsigned char seed) {
    const auto pages = allocation.page_ids();
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        std::vector<unsigned char> host(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            const std::size_t begin = static_cast<std::size_t>(pages[i] * tensor.nb[3]);
            for (std::int64_t byte = 0; byte < tensor.nb[3]; ++byte) {
                if (host[begin + static_cast<std::size_t>(byte)] != value) {
                    std::cerr << "opt restore logical page " << i << " plane " << plane
                              << " mismatch\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int capture_entry(ninfer::targets::qwen3_6::detail::KVRamCache& cache, ninfer::PagedKVPool& pool,
                  ninfer::PagedKVAllocation& alloc,
                  const ninfer::targets::qwen3_6::PreparedPromptData& prompt, cudaStream_t stream,
                  ninfer::LinearAttentionStatePool* gdn, const ninfer::Tensor* hidden) {
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
    source.tail_hidden_valid  = hidden != nullptr;
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = ninfer::targets::qwen3_6::detail::prefix_hash_at(
        retained.token_ids, identity, source.execution_frontier);
    source.text             = &alloc;
    source.text_pool        = &pool;
    source.text_pages       = alloc.mapped_page_count();
    source.gdn              = gdn;
    source.gdn_current_slot = gdn != nullptr ? 0 : -1;
    source.tail_hidden      = hidden;
    source.stream           = stream;
    return cache.capture(source) ? 0 : 1;
}

double gbs(std::size_t bytes, double ms, int copies) {
    return (static_cast<double>(copies) * static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
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

    ninfer::DeviceContext ctx(0);
    constexpr std::uint32_t kPages = 128;
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = kPages * 2,
                                                       .logical_page_capacity = kPages,
                                                       .table_rows            = 2,
                                                       .plane_order           = ninfer::PagedKVPlaneOrder::PageMajor,
                                                       .planes                = {{ninfer::DType::I8, 64, 2},
                                                                                  {ninfer::DType::I8, 64, 2},
                                                                                  {ninfer::DType::FP16, 1, 2},
                                                                                  {ninfer::DType::FP16, 1, 2}}});
    ninfer::DeviceArena arena(builder.finish(256));
    ninfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout);

    std::vector<ninfer::PagedKVAllocation> holes;
    holes.reserve(kPages);
    std::vector<ninfer::PagedKVAllocation> held;
    held.reserve(kPages);
    for (std::uint32_t i = 0; i < kPages; ++i) {
        auto even_page = pool.reserve(1);
        even_page.materialize_pages(1, ctx.stream);
        auto odd_page = pool.reserve(1);
        odd_page.materialize_pages(1, ctx.stream);
        holes.push_back(std::move(even_page));
        held.push_back(std::move(odd_page));
    }
    for (auto& hole : holes) { hole.release(); }
    auto source = pool.reserve(kPages);
    source.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, source, 21);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto ids = source.page_ids();
    bool fragmented = false;
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (ids[i] != ids[i - 1] + 1) {
            fragmented = true;
            break;
        }
    }
    if (!fragmented) { return fail("opt suite did not build a fragmented PageMajor mapping"); }

    const std::size_t image_bytes = ninfer::paged_kv_host_image_bytes(pool, kPages);
    ninfer::HostPinnedArena pinned(std::max(image_bytes, std::size_t{256}));
    void* pinned_block = pinned.try_alloc(image_bytes, 256);
    if (pinned_block == nullptr) { return fail("pinned memcpy buffer alloc failed"); }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    ninfer::DeviceBuffer bulk(image_bytes);
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(pinned_block, bulk.p, image_bytes, cudaMemcpyDeviceToHost,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(bulk.p, pinned_block, image_bytes, cudaMemcpyHostToDevice,
                                   ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(pinned_block, bulk.p, image_bytes, cudaMemcpyDeviceToHost,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(bulk.p, pinned_block, image_bytes, cudaMemcpyHostToDevice,
                                   ctx.stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double memcpy_ms  = elapsed_ms(start, stop) / kIters;
    const double memcpy_gbs = gbs(image_bytes, memcpy_ms, 2);

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        ninfer::pack_paged_kv_allocation_to_host(source, pool, kPages, pinned_block, ctx.stream);
        ninfer::unpack_paged_kv_allocation_from_host(source, pool, pinned_block, kPages, kPages,
                                                     ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double pack_ms  = elapsed_ms(start, stop) / kIters;
    const double pack_gbs = gbs(image_bytes, pack_ms, 2);
    std::cerr << "kv_ram_opt pages=" << kPages << " image_bytes=" << image_bytes
              << " fragmented=" << fragmented << " pack_unpack=" << pack_gbs
              << " GB/s memcpy=" << memcpy_gbs << " GB/s\n";
    if (pack_gbs < kMinFractionOfPinnedMemcpy * memcpy_gbs) {
        std::cerr << "paged KV host pack/unpack is below " << kMinFractionOfPinnedMemcpy
                  << " of pinned memcpy (" << pack_gbs << " vs " << memcpy_gbs << " GB/s)\n";
        return 1;
    }

    for (auto& page : held) { page.release(); }
    source.release();
    auto contiguous = pool.reserve(kPages);
    contiguous.materialize_pages(kPages, ctx.stream);
    fill_logical_pages(pool, contiguous, 23);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        ninfer::pack_paged_kv_allocation_to_host(contiguous, pool, kPages, pinned_block, ctx.stream);
        ninfer::unpack_paged_kv_allocation_from_host(contiguous, pool, pinned_block, kPages, kPages,
                                                     ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double contig_gbs = gbs(image_bytes, elapsed_ms(start, stop) / kIters, 2);
    std::cerr << "kv_ram_opt contiguous_pack_unpack=" << contig_gbs << " GB/s\n";
    if (contig_gbs < kMinFractionOfPinnedMemcpy * memcpy_gbs) {
        return fail("contiguous PageMajor pack/unpack fell below pinned memcpy floor");
    }

    const auto prompt = text_prompt({1, 2, 3, 4});
    ninfer::targets::qwen3_6::detail::KVRamCache cache(
        std::max(image_bytes * 4, std::size_t{1} << 20));
    if (capture_entry(cache, pool, contiguous, prompt, ctx.stream, nullptr, nullptr) != 0) {
        return fail("opt capture failed");
    }
    auto dest = pool.reserve(kPages);
    dest.materialize_pages(kPages, ctx.stream);
    const auto match =
        cache.plan_match(prompt, ninfer::targets::qwen3_6::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("opt capture did not index before copy completion"); }
    ninfer::targets::qwen3_6::detail::RamRestoreTarget target;
    target.text           = &dest;
    target.text_pool      = &pool;
    target.text_dst_pages = kPages;
    target.stream         = ctx.stream;
    cache.claim(match->entry_id);
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double restore_ms  = elapsed_ms(start, stop);
    const double restore_gbs = gbs(image_bytes, restore_ms, 1);
    std::cerr << "kv_ram_opt restore=" << restore_gbs << " GB/s (" << restore_ms << " ms)\n";
    if (restore_gbs < kMinFractionOfPinnedMemcpy * memcpy_gbs / 2.0) {
        std::cerr << "RAM restore is far below pinned memcpy (" << restore_gbs << " vs "
                  << memcpy_gbs << " GB/s)\n";
        return 1;
    }
    if (expect_logical_pages(pool, dest, 23) != 0) {
        return fail("event-overlapped restore corrupted logical pages");
    }
    dest.release();

    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(
        gdn_builder, {.layers         = 8,
                      .conv_channels  = 16,
                      .conv_width     = 4,
                      .value_heads    = 2,
                      .value_head_dim = 8,
                      .key_head_dim   = 4,
                      .slot_count     = 2,
                      .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    const std::size_t gdn_bytes = gdn.conv_host_image_bytes() + gdn.recurrent_host_image_bytes();
    ninfer::HostPinnedArena gdn_pinned(std::max(gdn_bytes, std::size_t{256}));
    void* gdn_host = gdn_pinned.try_alloc(gdn_bytes, 256);
    if (gdn_host == nullptr) { return fail("GDN pinned image alloc failed"); }
    std::vector<unsigned char> conv_pattern(gdn.conv_slot(0, 0).bytes());
    for (std::size_t i = 0; i < conv_pattern.size(); ++i) {
        conv_pattern[i] = static_cast<unsigned char>(i * 5 + 1);
    }
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        CUDA_CHECK(cudaMemcpy(gdn.conv_slot(layer, 0).data, conv_pattern.data(), conv_pattern.size(),
                               cudaMemcpyHostToDevice));
    }
    ninfer::DeviceBuffer hidden_buf(256);
    hidden_buf.fill(0x5a);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {256});
    fill_logical_pages(pool, contiguous, 24);
    ninfer::targets::qwen3_6::detail::KVRamCache gdn_cache(8ULL << 20);
    if (capture_entry(gdn_cache, pool, contiguous, prompt, ctx.stream, &gdn, &hidden) != 0) {
        return fail("GDN capture failed");
    }
    auto gdn_dest = pool.reserve(kPages);
    gdn_dest.materialize_pages(kPages, ctx.stream);
    const auto gdn_match =
        gdn_cache.plan_match(prompt, ninfer::targets::qwen3_6::detail::prefix_hash_chain(prompt));
    if (!gdn_match) { return fail("GDN capture did not match"); }
    ninfer::DeviceBuffer hidden_out_buf(256);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {256});
    ninfer::targets::qwen3_6::detail::RamRestoreTarget gdn_target;
    gdn_target.text             = &gdn_dest;
    gdn_target.text_pool        = &pool;
    gdn_target.text_dst_pages   = kPages;
    gdn_target.gdn              = &gdn;
    gdn_target.gdn_current_slot = 1;
    gdn_target.tail_hidden      = &hidden_out;
    gdn_target.stream           = ctx.stream;
    gdn_cache.claim(gdn_match->entry_id);
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    (void)gdn_cache.unpack_device(gdn_match->entry_id, gdn_target);
    gdn_cache.consume(gdn_match->entry_id);
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    std::cerr << "kv_ram_opt gdn8_restore=" << elapsed_ms(start, stop) << " ms\n";
    if (expect_logical_pages(pool, gdn_dest, 24) != 0) { return fail("GDN restore KV mismatch"); }
    std::vector<unsigned char> conv_out(conv_pattern.size());
    CUDA_CHECK(cudaMemcpy(conv_out.data(), gdn.conv_slot(7, 1).data, conv_out.size(),
                           cudaMemcpyDeviceToHost));
    if (conv_out != conv_pattern) { return fail("8-layer GDN conv did not round-trip"); }
    std::vector<unsigned char> hidden_host(256);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(256, 0x5a)) {
        return fail("tail hidden did not round-trip through RAM");
    }

    auto* conv_dst = static_cast<unsigned char*>(gdn_host);
    auto* rec_dst  = conv_dst + gdn.conv_host_image_bytes();
    for (int i = 0; i < kWarmup; ++i) {
        gdn.pack_slot_to_host(1, conv_dst, rec_dst, ctx.stream);
        gdn.unpack_slot_from_host(0, conv_dst, rec_dst, ctx.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        gdn.pack_slot_to_host(1, conv_dst, rec_dst, ctx.stream);
        gdn.unpack_slot_from_host(0, conv_dst, rec_dst, ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double gdn_gbs = gbs(gdn_bytes, elapsed_ms(start, stop) / kIters, 2);
    std::cerr << "kv_ram_opt gdn8_pack_unpack=" << gdn_gbs << " GB/s bytes=" << gdn_bytes << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    contiguous.release();
    gdn_dest.release();
    std::cout << "ok\n";
    return 0;
}
