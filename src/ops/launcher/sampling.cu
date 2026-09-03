// Implements: include/ninfer/ops/sampling.h
// Match: validated contiguous BF16/I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

void sample_batch_launch_impl(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                              const SamplingConfig* configs, const Tensor& logical_positions,
                              std::int32_t purpose, const Tensor* hidden, const Tensor* lanes,
                              Tensor* continuation_hidden, DeviceSpan workspace,
                              cudaStream_t stream) {
    const bool scatter = hidden != nullptr;

    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    if (!layout.multiblock) {
        if (scatter) {
            throw std::invalid_argument("sample_and_scatter_hidden requires multiblock sampling");
        }
        sample_row_kernel<<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const dim3 partial_grid(static_cast<unsigned int>(partial_blocks),
                            static_cast<unsigned int>(batch));
    sampling_partial_topk_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), configs, token_domain, physical_rows,
        scratch, scatter ? static_cast<const uint4*>(hidden->data) : nullptr,
        scatter ? static_cast<const std::int32_t*>(lanes->data) : nullptr,
        scatter ? static_cast<uint4*>(continuation_hidden->data) : nullptr,
        scatter ? hidden->ne[0] / 8 : 0);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    sample_batch_launch_impl(logits, out, token_domain, configs, logical_positions, purpose,
                             nullptr, nullptr, nullptr, workspace, stream);
}

void sample_batch_scatter_hidden_launch(
    const Tensor& logits, Tensor& out, std::int32_t token_domain, const SamplingConfig* configs,
    const Tensor& logical_positions, std::int32_t purpose, const Tensor& hidden,
    const Tensor& lanes, Tensor& continuation_hidden, DeviceSpan workspace, cudaStream_t stream) {
    sample_batch_launch_impl(logits, out, token_domain, configs, logical_positions, purpose,
                             &hidden, &lanes, &continuation_hidden, workspace, stream);
}

} // namespace ninfer::ops::detail
