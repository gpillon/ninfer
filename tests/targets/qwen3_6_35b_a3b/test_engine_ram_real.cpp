#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

// 35B site-2 RAM restore. DFlash cyclic payload is covered by the family unit image in
// ninfer_kv_ram_cache_test, not by D17's planner gate.

namespace {

constexpr std::size_t kRamHitBytes = 1024ULL * 1024ULL * 1024ULL;

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> tokens_a() { return {248045, 846, 198, 5834, 248046, 198}; }
std::vector<ninfer::TokenId> tokens_c() { return {248045, 846, 198, 9906, 248046, 198}; }

std::vector<ninfer::TokenId> resume_prefix(const std::vector<ninfer::TokenId>& keep,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> prefix = keep;
    if (!generated.empty()) {
        prefix.insert(prefix.end(), generated.begin(), generated.end() - 1);
    }
    return prefix;
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency       = 1;
    options.prefill_chunk         = 1024;
    options.kv_ram_capacity_bytes = kRamHitBytes;
    options.kv_cache              = ninfer::KvCacheStorage::Int8Group64;
    options.enable_vision         = false;
    options.use_cuda_graph        = true;
    return options;
}

int exercise_site2(ninfer::Engine& engine) {
    const auto keep                          = tokens_a();
    const ninfer::GenerationResult first     = engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        std::cerr << "35B RAM source request did not generate eight tokens\n";
        return 1;
    }
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (other.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "35B RAM evictor did not FullReset\n";
        return 1;
    }
    if (engine.runtime_stats().kv_ram_captures <= captures_before) {
        std::cerr << "35B RAM evictor did not increment kv_ram_captures\n";
        return 1;
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const auto restores_before                 = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        hit.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        hit.reused_prompt_tokens != history.size()) {
        std::cerr << "35B RAM hit failed: source=" << static_cast<int>(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        std::cerr << "35B RAM hit did not increment kv_ram_restores\n";
        return 1;
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        std::cerr << "35B RAM reuse changed greedy output\n";
        return 1;
    }
    const ninfer::GenerationResult denied =
        engine.generate(engine.prepare_tokens(history), greedy(2, false));
    if (denied.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        denied.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "35B allow_prefix_reuse=false reused a RAM entry\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_35B_A3B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_6_35B_A3B_WEIGHTS is not set\n";
        return 77;
    }
    ninfer::Engine engine(engine_options(artifact));
    if (engine.memory_summary().kv_ram_capacity_bytes != kRamHitBytes) {
        std::cerr << "35B KV RAM capacity was not applied\n";
        return 1;
    }
    if (const int result = exercise_site2(engine); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
