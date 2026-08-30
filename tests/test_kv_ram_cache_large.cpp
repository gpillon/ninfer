#include "core/arena.h"
#include "core/device.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace {

constexpr int kWarmup = 1;
constexpr int kIters  = 3;
constexpr double kMinFractionOfPinnedMemcpy = 0.50;
constexpr std::uint32_t kKvPages            = 48;
constexpr std::size_t kMinKvImageBytes      = 90ULL * 1024ULL * 1024ULL;

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

double gbs(std::size_t bytes, double ms, int copies) {
    return (static_cast<double>(copies) * static_cast<double>(bytes) / 1.0e9) / (ms / 1000.0);
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

std::vector<ninfer::PagedKVPlaneSpec> int8_text_planes() {
    std::vector<ninfer::PagedKVPlaneSpec> planes;
    planes.reserve(64);
    for (int layer = 0; layer < 16; ++layer) {
        planes.push_back({ninfer::DType::I8, 256, 4, 256});
        planes.push_back({ninfer::DType::I8, 256, 4, 256});
        planes.push_back({ninfer::DType::FP16, 4, 4, 256});
        planes.push_back({ninfer::DType::FP16, 4, 4, 256});
    }
    return planes;
}

ninfer::LinearAttentionStatePoolSpec gdn_27b_spec(std::int32_t slots) {
    return {.layers         = 48,
            .conv_channels  = 10240,
            .conv_width     = 3,
            .value_heads    = 48,
            .value_head_dim = 128,
            .key_head_dim   = 128,
            .slot_count     = slots,
            .conv_dtype     = ninfer::DType::BF16};
}

void fill_slot(ninfer::LinearAttentionStatePool& gdn, std::int32_t slot, unsigned char seed) {
    for (std::uint32_t layer = 0; layer < gdn.layer_count(); ++layer) {
        const ninfer::Tensor conv = gdn.conv_slot(layer, slot);
        const ninfer::Tensor rec  = gdn.recurrent_slot(layer, slot);
        CUDA_CHECK(cudaMemset(conv.data, seed + static_cast<unsigned char>(layer), conv.bytes()));
        CUDA_CHECK(cudaMemset(rec.data, seed + 17U + static_cast<unsigned char>(layer), rec.bytes()));
    }
}

int expect_slot(ninfer::LinearAttentionStatePool& gdn, std::int32_t slot, unsigned char seed,
                const char* label) {
    const std::uint32_t layers[] = {0U, gdn.layer_count() / 2U, gdn.layer_count() - 1U};
    for (std::uint32_t layer : layers) {
        const ninfer::Tensor conv = gdn.conv_slot(layer, slot);
        const ninfer::Tensor rec  = gdn.recurrent_slot(layer, slot);
        std::vector<unsigned char> conv_host(conv.bytes());
        std::vector<unsigned char> rec_host(rec.bytes());
        CUDA_CHECK(cudaMemcpy(conv_host.data(), conv.data, conv.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rec_host.data(), rec.data, rec.bytes(), cudaMemcpyDeviceToHost));
        const unsigned char conv_byte = seed + static_cast<unsigned char>(layer);
        const unsigned char rec_byte  = seed + 17U + static_cast<unsigned char>(layer);
        for (unsigned char b : conv_host) {
            if (b != conv_byte) {
                std::cerr << label << " GDN conv layer " << layer << " mismatch\n";
                return 1;
            }
        }
        for (unsigned char b : rec_host) {
            if (b != rec_byte) {
                std::cerr << label << " GDN recurrent layer " << layer << " mismatch\n";
                return 1;
            }
        }
    }
    return 0;
}

void fill_page_sentinels(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                         unsigned char seed) {
    const auto pages = allocation.page_ids();
    if (pages.empty()) { return; }
    const std::size_t last = pages.size() - 1;
    const std::size_t plane_ids[] = {0, pool.plane_count() / 2, pool.plane_count() - 1};
    for (std::size_t plane : plane_ids) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        const std::size_t bpp        = static_cast<std::size_t>(tensor.nb[3]);
        auto* base                   = static_cast<unsigned char*>(tensor.data);
        for (std::size_t i : {std::size_t{0}, last}) {
            std::vector<unsigned char> page(
                bpp, static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U));
            CUDA_CHECK(cudaMemcpy(base + static_cast<std::int64_t>(pages[i]) * tensor.nb[3],
                                  page.data(), bpp, cudaMemcpyHostToDevice));
        }
    }
}

int expect_page_sentinels(ninfer::PagedKVPool& pool, const ninfer::PagedKVAllocation& allocation,
                          unsigned char seed, const char* label) {
    const auto pages = allocation.page_ids();
    if (pages.empty()) { return 0; }
    const std::size_t last = pages.size() - 1;
    const std::size_t plane_ids[] = {0, pool.plane_count() / 2, pool.plane_count() - 1};
    for (std::size_t plane : plane_ids) {
        const ninfer::Tensor& tensor = pool.plane(plane);
        const std::size_t bpp        = static_cast<std::size_t>(tensor.nb[3]);
        auto* base                   = static_cast<unsigned char*>(tensor.data);
        for (std::size_t i : {std::size_t{0}, last}) {
            std::vector<unsigned char> page(bpp);
            CUDA_CHECK(cudaMemcpy(page.data(),
                                  base + static_cast<std::int64_t>(pages[i]) * tensor.nb[3], bpp,
                                  cudaMemcpyDeviceToHost));
            const unsigned char value =
                static_cast<unsigned char>(seed + plane * 17U + static_cast<unsigned>(i) + 1U);
            for (unsigned char b : page) {
                if (b != value) {
                    std::cerr << label << " sentinel plane " << plane << " page " << i
                              << " mismatch\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int capture_bundle(ninfer::targets::qwen3_6::detail::KVRamCache& cache, ninfer::PagedKVPool& pool,
                   ninfer::PagedKVAllocation& alloc, ninfer::LinearAttentionStatePool& gdn,
                   const ninfer::Tensor& hidden, const ninfer::Tensor& rewrite, cudaStream_t stream) {
    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
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
    source.execution_frontier      = static_cast<std::uint32_t>(prompt.token_ids.size());
    source.ledger_frontier         = static_cast<std::uint32_t>(tokens);
    source.text_kv_valid           = source.execution_frontier;
    source.tail_hidden_valid       = true;
    source.rewrite_valid           = true;
    source.rewrite_kind            = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure;
    source.rewrite_frontier        = 4;
    source.hash_c_valid            = true;
    source.ledger                  = retained.token_ids;
    source.identity                = &identity;
    source.hash_f                  = ninfer::targets::qwen3_6::detail::prefix_hash_at(
        retained.token_ids, identity, source.execution_frontier);
    source.hash_c = ninfer::targets::qwen3_6::detail::prefix_hash_at(retained.token_ids, identity, 4);
    source.text                 = &alloc;
    source.text_pool            = &pool;
    source.text_pages           = alloc.mapped_page_count();
    source.gdn                  = &gdn;
    source.gdn_current_slot     = 0;
    source.gdn_checkpoint_slot  = 1;
    source.tail_hidden          = &hidden;
    source.rewrite_checkpoint_hidden = &rewrite;
    source.stream               = stream;
    return cache.capture(source) ? 0 : 1;
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
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    ninfer::LayoutBuilder gdn_builder;
    const auto gdn_layout = ninfer::plan_linear_attention_state_pool(gdn_builder, gdn_27b_spec(2));
    ninfer::DeviceArena gdn_arena(gdn_builder.finish(256));
    ninfer::LinearAttentionStatePool gdn({gdn_arena.base(), gdn_arena.capacity()}, gdn_layout);
    const std::size_t gdn_slot_bytes =
        gdn.conv_host_image_bytes() + gdn.recurrent_host_image_bytes();
    if (gdn_slot_bytes < 140ULL * 1024ULL * 1024ULL) {
        return fail("27B GDN slot image is smaller than the documented recurrent state");
    }
    ninfer::HostPinnedArena gdn_pinned(std::max(gdn_slot_bytes, std::size_t{256}));
    void* gdn_host = gdn_pinned.try_alloc(gdn_slot_bytes, 256);
    if (gdn_host == nullptr) { return fail("27B GDN pinned alloc failed"); }
    auto* conv_dst = static_cast<unsigned char*>(gdn_host);
    auto* rec_dst  = conv_dst + gdn.conv_host_image_bytes();

    fill_slot(gdn, 0, 0x31);
    ninfer::DeviceBuffer gdn_bulk(gdn_slot_bytes);
    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(gdn_host, gdn_bulk.p, gdn_slot_bytes, cudaMemcpyDeviceToHost,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(gdn_bulk.p, gdn_host, gdn_slot_bytes, cudaMemcpyHostToDevice,
                                   ctx.stream));
        gdn.pack_slot_to_host(0, conv_dst, rec_dst, ctx.stream);
        gdn.unpack_slot_from_host(1, conv_dst, rec_dst, ctx.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(gdn_host, gdn_bulk.p, gdn_slot_bytes, cudaMemcpyDeviceToHost,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(gdn_bulk.p, gdn_host, gdn_slot_bytes, cudaMemcpyHostToDevice,
                                   ctx.stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double gdn_memcpy_ms  = elapsed_ms(start, stop) / kIters;
    const double gdn_memcpy_gbs = gbs(gdn_slot_bytes, gdn_memcpy_ms, 2);

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        gdn.pack_slot_to_host(0, conv_dst, rec_dst, ctx.stream);
        gdn.unpack_slot_from_host(1, conv_dst, rec_dst, ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double gdn_pack_ms  = elapsed_ms(start, stop) / kIters;
    const double gdn_pack_gbs = gbs(gdn_slot_bytes, gdn_pack_ms, 2);
    std::cerr << "kv_ram_large gdn27_slot_bytes=" << gdn_slot_bytes
              << " pack_unpack=" << gdn_pack_gbs << " GB/s (" << gdn_pack_ms
              << " ms) memcpy=" << gdn_memcpy_gbs << " GB/s (" << gdn_memcpy_ms << " ms)\n";
    if (gdn_pack_gbs < kMinFractionOfPinnedMemcpy * gdn_memcpy_gbs) {
        return fail("27B GDN pack/unpack fell below pinned memcpy floor");
    }
    if (expect_slot(gdn, 1, 0x31, "27B GDN unpack") != 0) { return 1; }

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) { gdn.copy_slot(0, 1, ctx.stream); }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double gdn_d2d_ms = elapsed_ms(start, stop) / kIters;
    std::cerr << "kv_ram_large gdn27_copy_slot=" << gdn_d2d_ms << " ms ("
              << gbs(gdn_slot_bytes, gdn_d2d_ms, 1) << " GB/s D2D)\n";

    ninfer::LayoutBuilder kv_builder;
    auto kv_layout = ninfer::plan_paged_kv_pool(kv_builder, {.page_group_count      = kKvPages * 2,
                                                             .logical_page_capacity = kKvPages,
                                                             .table_rows            = 2,
                                                             .plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                                                             .planes      = int8_text_planes()});
    ninfer::DeviceArena kv_arena(kv_builder.finish(256));
    ninfer::PagedKVPool kv_pool({kv_arena.base(), kv_arena.capacity()}, kv_layout);

    std::vector<ninfer::PagedKVAllocation> holes;
    std::vector<ninfer::PagedKVAllocation> held;
    holes.reserve(kKvPages);
    held.reserve(kKvPages);
    for (std::uint32_t i = 0; i < kKvPages; ++i) {
        auto even_page = kv_pool.reserve(1);
        even_page.materialize_pages(1, ctx.stream);
        auto odd_page = kv_pool.reserve(1);
        odd_page.materialize_pages(1, ctx.stream);
        holes.push_back(std::move(even_page));
        held.push_back(std::move(odd_page));
    }
    for (auto& hole : holes) { hole.release(); }
    auto fragmented = kv_pool.reserve(kKvPages);
    fragmented.materialize_pages(kKvPages, ctx.stream);
    const auto frag_ids = fragmented.page_ids();
    bool is_fragmented  = false;
    for (std::size_t i = 1; i < frag_ids.size(); ++i) {
        if (frag_ids[i] != frag_ids[i - 1] + 1) {
            is_fragmented = true;
            break;
        }
    }
    if (!is_fragmented) { return fail("large suite did not build a fragmented 64-plane mapping"); }
    fill_page_sentinels(kv_pool, fragmented, 41);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const std::size_t kv_bytes = ninfer::paged_kv_host_image_bytes(kv_pool, kKvPages);
    if (kv_bytes < kMinKvImageBytes) {
        std::cerr << "64-plane KV image is " << kv_bytes << " bytes, expected >= " << kMinKvImageBytes
                  << '\n';
        return 1;
    }
    ninfer::HostPinnedArena kv_pinned(std::max(kv_bytes, std::size_t{256}));
    void* kv_host = kv_pinned.try_alloc(kv_bytes, 256);
    if (kv_host == nullptr) { return fail("100 MiB KV pinned alloc failed"); }
    ninfer::DeviceBuffer kv_bulk(kv_bytes);

    for (int i = 0; i < kWarmup; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(kv_host, kv_bulk.p, kv_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(kv_bulk.p, kv_host, kv_bytes, cudaMemcpyHostToDevice, ctx.stream));
        ninfer::pack_paged_kv_allocation_to_host(fragmented, kv_pool, kKvPages, kv_host, ctx.stream);
        ninfer::unpack_paged_kv_allocation_from_host(fragmented, kv_pool, kv_host, kKvPages, kKvPages,
                                                     ctx.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(kv_host, kv_bulk.p, kv_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(kv_bulk.p, kv_host, kv_bytes, cudaMemcpyHostToDevice, ctx.stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double kv_memcpy_ms  = elapsed_ms(start, stop) / kIters;
    const double kv_memcpy_gbs = gbs(kv_bytes, kv_memcpy_ms, 2);

    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        ninfer::pack_paged_kv_allocation_to_host(fragmented, kv_pool, kKvPages, kv_host, ctx.stream);
        ninfer::unpack_paged_kv_allocation_from_host(fragmented, kv_pool, kv_host, kKvPages, kKvPages,
                                                     ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double kv_frag_ms  = elapsed_ms(start, stop) / kIters;
    const double kv_frag_gbs = gbs(kv_bytes, kv_frag_ms, 2);
    std::cerr << "kv_ram_large kv_bytes=" << kv_bytes << " planes=" << kv_pool.plane_count()
              << " fragmented_pack_unpack=" << kv_frag_gbs << " GB/s (" << kv_frag_ms
              << " ms) memcpy=" << kv_memcpy_gbs << " GB/s (" << kv_memcpy_ms << " ms)\n";
    if (kv_frag_gbs < kMinFractionOfPinnedMemcpy * kv_memcpy_gbs) {
        return fail("fragmented 100 MiB KV pack/unpack fell below pinned memcpy floor");
    }
    if (expect_page_sentinels(kv_pool, fragmented, 41, "fragmented 100 MiB KV") != 0) { return 1; }

    for (auto& page : held) { page.release(); }
    fragmented.release();
    auto contiguous = kv_pool.reserve(kKvPages);
    contiguous.materialize_pages(kKvPages, ctx.stream);
    fill_page_sentinels(kv_pool, contiguous, 42);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    for (int i = 0; i < kIters; ++i) {
        ninfer::pack_paged_kv_allocation_to_host(contiguous, kv_pool, kKvPages, kv_host, ctx.stream);
        ninfer::unpack_paged_kv_allocation_from_host(contiguous, kv_pool, kv_host, kKvPages, kKvPages,
                                                     ctx.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double kv_contig_ms  = elapsed_ms(start, stop) / kIters;
    const double kv_contig_gbs = gbs(kv_bytes, kv_contig_ms, 2);
    std::cerr << "kv_ram_large contiguous_pack_unpack=" << kv_contig_gbs << " GB/s (" << kv_contig_ms
              << " ms)\n";
    if (kv_contig_gbs < kMinFractionOfPinnedMemcpy * kv_memcpy_gbs) {
        return fail("contiguous 100 MiB KV pack/unpack fell below pinned memcpy floor");
    }

    fill_slot(gdn, 0, 0x44);
    fill_slot(gdn, 1, 0x45);
    ninfer::DeviceBuffer hidden_buf(10240);
    hidden_buf.fill(0xa1);
    ninfer::Tensor hidden(hidden_buf.p, ninfer::DType::U8, {10240});
    ninfer::DeviceBuffer rewrite_buf(10240);
    rewrite_buf.fill(0xa2);
    ninfer::Tensor rewrite(rewrite_buf.p, ninfer::DType::U8, {10240});
    ninfer::targets::qwen3_6::detail::KVRamCache cache(1024ULL * 1024ULL * 1024ULL);
    if (capture_bundle(cache, kv_pool, contiguous, gdn, hidden, rewrite, ctx.stream) != 0) {
        return fail("large SequenceState capture failed");
    }
    gdn.zero_slot(0, ctx.stream);
    gdn.zero_slot(1, ctx.stream);
    auto dest = kv_pool.reserve(kKvPages);
    dest.materialize_pages(kKvPages, ctx.stream);
    ninfer::DeviceBuffer hidden_out_buf(10240);
    hidden_out_buf.fill(0);
    ninfer::Tensor hidden_out(hidden_out_buf.p, ninfer::DType::U8, {10240});
    ninfer::DeviceBuffer rewrite_out_buf(10240);
    rewrite_out_buf.fill(0);
    ninfer::Tensor rewrite_out(rewrite_out_buf.p, ninfer::DType::U8, {10240});
    const auto prompt = text_prompt({1, 2, 3, 4, 5, 6, 7, 8});
    const auto match =
        cache.plan_match(prompt, ninfer::targets::qwen3_6::detail::prefix_hash_chain(prompt));
    if (!match) { return fail("large capture did not match"); }
    ninfer::targets::qwen3_6::detail::RamRestoreTarget target;
    target.text                      = &dest;
    target.text_pool                 = &kv_pool;
    target.text_dst_pages            = kKvPages;
    target.gdn                       = &gdn;
    target.gdn_current_slot          = 0;
    target.gdn_checkpoint_slot       = 1;
    target.tail_hidden               = &hidden_out;
    target.rewrite_checkpoint_hidden = &rewrite_out;
    target.stream                    = ctx.stream;
    cache.claim(match->entry_id);
    CUDA_CHECK(cudaEventRecord(start, ctx.stream));
    (void)cache.unpack_device(match->entry_id, target);
    cache.consume(match->entry_id);
    CUDA_CHECK(cudaEventRecord(stop, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const double restore_ms   = elapsed_ms(start, stop);
    const std::size_t payload = kv_bytes + 2 * gdn_slot_bytes;
    std::cerr << "kv_ram_large restore_two_gdn_slots_plus_kv=" << gbs(payload, restore_ms, 1)
              << " GB/s (" << restore_ms << " ms) payload_bytes=" << payload << '\n';
    if (expect_page_sentinels(kv_pool, dest, 42, "large restore KV") != 0) { return 1; }
    if (expect_slot(gdn, 0, 0x44, "large restore GDN current") != 0) { return 1; }
    if (expect_slot(gdn, 1, 0x45, "large restore GDN checkpoint") != 0) { return 1; }
    std::vector<unsigned char> hidden_host(10240);
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), hidden_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(10240, 0xa1)) {
        return fail("large restore tail hidden mismatch");
    }
    CUDA_CHECK(cudaMemcpy(hidden_host.data(), rewrite_out.data, hidden_host.size(),
                           cudaMemcpyDeviceToHost));
    if (hidden_host != std::vector<unsigned char>(10240, 0xa2)) {
        return fail("large restore rewrite hidden mismatch");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    contiguous.release();
    dest.release();
    std::cout << "ok\n";
    return 0;
}
