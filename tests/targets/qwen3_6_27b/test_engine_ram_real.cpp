#include "ninfer/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kRamHitBytes  = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kRamDropBytes = 1024ULL * 1024ULL;

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<ninfer::TokenId> tokens_a() { return {248045, 846, 198, 5834, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_b(std::size_t count) {
    return std::vector<ninfer::TokenId>(count, 198);
}

std::vector<ninfer::TokenId> tokens_c() { return {248045, 846, 198, 9906, 248046, 198}; }

std::vector<ninfer::TokenId> tokens_d() { return {400, 401, 402, 403, 404, 405}; }

std::vector<ninfer::TokenId> tokens_e() { return {500, 501, 502, 503, 504, 505}; }

std::vector<ninfer::TokenId> concat(std::vector<ninfer::TokenId> prefix,
                                    const std::vector<ninfer::TokenId>& suffix) {
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return prefix;
}

std::vector<ninfer::TokenId> resume_prefix(const std::vector<ninfer::TokenId>& keep,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> prefix = keep;
    if (!generated.empty()) {
        prefix.insert(prefix.end(), generated.begin(), generated.end() - 1);
    }
    return prefix;
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

std::vector<ninfer::TokenId> pad_tokens(std::vector<ninfer::TokenId> seed, std::size_t count) {
    if (seed.size() > count) { seed.resize(count); }
    seed.resize(count, 198);
    return seed;
}

int expect_ram_hit(const ninfer::GenerationResult& result, std::uint32_t history_tokens,
                   const char* label) {
    if (result.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        std::cerr << label << " reuse_source is "
                  << static_cast<int>(result.prefix_reuse_source) << ", expected HostRam\n";
        return 1;
    }
    if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " RAM hit used FullReset\n";
        return 1;
    }
    if (result.reused_prompt_tokens != history_tokens) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected "
                  << history_tokens << '\n';
        return 1;
    }
    return 0;
}

ninfer::EngineOptions ordinary_options(const char* artifact, std::uint32_t max_concurrency,
                                       std::uint32_t max_context, std::size_t ram_bytes,
                                       std::uint32_t prefill_chunk = 1024) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = max_context;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency       = max_concurrency;
    options.prefill_chunk         = prefill_chunk;
    options.kv_ram_capacity_bytes = ram_bytes;
    options.kv_cache              = ninfer::KvCacheStorage::Int8Group64;
    options.enable_vision         = false;
    options.use_cuda_graph        = true;
    return options;
}

ninfer::EngineOptions pooled_c3_options(const char* artifact) {
    ninfer::EngineOptions options = ordinary_options(artifact, 3, 512, kRamHitBytes, 128);
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(640);
    options.pending_timeout_ms    = 120000;
    return options;
}

ninfer::EngineOptions mtp_options(const char* artifact) {
    ninfer::EngineOptions options     = ordinary_options(artifact, 1, 4096, kRamHitBytes);
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

int verify_ram_tier(const ninfer::Engine& engine, std::size_t ram_bytes) {
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.kv_ram_capacity_bytes != ram_bytes) {
        std::cerr << "KV RAM capacity is " << memory.kv_ram_capacity_bytes << ", expected "
                  << ram_bytes << '\n';
        return 1;
    }
    return 0;
}

int expect_suffix_hit(const ninfer::GenerationResult& result, ninfer::PrefixReuseSource source,
                      std::uint32_t prefix_tokens, std::uint32_t prompt_tokens, const char* label) {
    if (prefix_tokens >= prompt_tokens) {
        std::cerr << label << " suffix fixture is not longer than the reused prefix\n";
        return 1;
    }
    if (result.prompt.prompt_tokens != prompt_tokens) {
        std::cerr << label << " prompt_tokens is " << result.prompt.prompt_tokens << ", expected "
                  << prompt_tokens << '\n';
        return 1;
    }
    if (result.prefix_reuse_source != source) {
        std::cerr << label << " reuse_source is " << static_cast<int>(result.prefix_reuse_source)
                  << ", expected " << static_cast<int>(source) << '\n';
        return 1;
    }
    if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " suffix hit used FullReset\n";
        return 1;
    }
    if (result.reused_prompt_tokens != prefix_tokens) {
        std::cerr << label << " reused " << result.reused_prompt_tokens << ", expected prefix "
                  << prefix_tokens << " of " << prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int expect_exact_hit(const ninfer::GenerationResult& result, std::uint32_t prompt_tokens,
                     const char* label) {
    if (result.prompt.prompt_tokens != prompt_tokens || result.reused_prompt_tokens != prompt_tokens) {
        std::cerr << label << " exact hit reused " << result.reused_prompt_tokens << " of prompt "
                  << result.prompt.prompt_tokens << ", expected " << prompt_tokens << '\n';
        return 1;
    }
    if (result.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        result.prefix_reuse_source == ninfer::PrefixReuseSource::None) {
        std::cerr << label << " exact hit used FullReset/None: source="
                  << static_cast<int>(result.prefix_reuse_source)
                  << " path=" << static_cast<int>(result.prefix_reuse_path) << '\n';
        return 1;
    }
    return 0;
}

bool wait_scheduler(ninfer::Engine& engine, std::uint32_t* max_prefilling,
                    const auto& predicate, const char* label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    while (std::chrono::steady_clock::now() < deadline) {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        *max_prefilling = std::max(*max_prefilling, stats.prefilling_requests);
        if (stats.prefilling_requests > 1) {
            std::cerr << label << " observed prefilling_requests=" << stats.prefilling_requests
                      << '\n';
            return false;
        }
        if (predicate(stats)) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const ninfer::RuntimeStats stats = engine.runtime_stats();
    std::cerr << label << " timed out: running=" << stats.running_requests
              << " prefilling=" << stats.prefilling_requests
              << " decode_ready=" << stats.decode_ready_requests
              << " waiting=" << stats.waiting_requests << '\n';
    return false;
}

int capture_then_hit(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& keep,
                     const std::vector<ninfer::TokenId>& evictor, const char* label) {
    const ninfer::GenerationResult first = engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("source request did not generate eight tokens");
    }
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare_tokens(evictor), greedy(4, false));
    if (other.generated_token_ids.size() != 4 ||
        other.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << label << " evictor did not FullReset\n";
        return 1;
    }
    if (engine.runtime_stats().kv_ram_captures <= captures_before) {
        std::cerr << label << " evictor did not increment kv_ram_captures\n";
        return 1;
    }

    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const auto restores_before                 = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc = expect_ram_hit(hit, static_cast<std::uint32_t>(history.size()), label);
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        std::cerr << label << " kv_ram_restores is " << engine.runtime_stats().kv_ram_restores
                  << ", expected " << restores_before + 1 << '\n';
        return 1;
    }
    const auto restores_after_hit              = engine.runtime_stats().kv_ram_restores;
    const std::vector<ninfer::TokenId> after_hit =
        resume_prefix(history, hit.generated_token_ids);
    const ninfer::GenerationResult resident =
        engine.generate(engine.prepare_tokens(after_hit), greedy(2, true));
    if (resident.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        resident.reused_prompt_tokens != after_hit.size() ||
        engine.runtime_stats().kv_ram_restores != restores_after_hit) {
        std::cerr << label << " consumed RAM hit did not leave VRAM as the remaining copy: source="
                  << static_cast<int>(resident.prefix_reuse_source) << " reused "
                  << resident.reused_prompt_tokens << '\n';
        return 1;
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        std::cerr << label << " RAM reuse changed greedy output\n";
        return 1;
    }
    return 0;
}

int exercise_exclusive_occupancy(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const auto keep_b = tokens_c();
    const auto keep_d = tokens_d();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(8, false));
    if (first_a.generated_token_ids.size() != 8) {
        return fail("exclusive occupancy source A did not generate eight tokens");
    }
    const auto captures_after_a = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(keep_b), greedy(4, false));
    if (first_b.generated_token_ids.size() != 4 ||
        first_b.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("exclusive occupancy evictor did not FullReset");
    }
    if (engine.runtime_stats().kv_ram_captures <= captures_after_a) {
        return fail("exclusive occupancy evictor did not capture A");
    }
    const ninfer::MemorySummary after_capture = engine.memory_summary();
    if (after_capture.kv_ram_entry_count != 1 || after_capture.kv_ram_used_bytes == 0) {
        std::cerr << "exclusive occupancy after capture: entries="
                  << after_capture.kv_ram_entry_count
                  << " used=" << after_capture.kv_ram_used_bytes << '\n';
        return 1;
    }

    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const auto captures_before_hit               = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult hit_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(4, true));
    if (const int rc = expect_ram_hit(hit_a, static_cast<std::uint32_t>(history_a.size()),
                                      "exclusive occupancy RAM hit");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1 ||
        engine.runtime_stats().kv_ram_captures != captures_before_hit + 1) {
        return fail("exclusive occupancy RAM hit did not consume A and capture the victim");
    }
    const ninfer::MemorySummary after_hit = engine.memory_summary();
    if (after_hit.kv_ram_entry_count != 1 || after_hit.kv_ram_used_bytes == 0) {
        std::cerr << "exclusive occupancy after RAM hit is not victim-only: entries="
                  << after_hit.kv_ram_entry_count << " used=" << after_hit.kv_ram_used_bytes
                  << '\n';
        return 1;
    }
    const std::vector<ninfer::TokenId> after_hit_a =
        resume_prefix(history_a, hit_a.generated_token_ids);
    const auto restores_after_hit = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult resident_a =
        engine.generate(engine.prepare_tokens(after_hit_a), greedy(2, true));
    if (resident_a.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        resident_a.reused_prompt_tokens != after_hit_a.size() ||
        engine.runtime_stats().kv_ram_restores != restores_after_hit) {
        std::cerr << "exclusive occupancy continue after hit was not VRAM: source="
                  << static_cast<int>(resident_a.prefix_reuse_source) << '\n';
        return 1;
    }

    const auto captures_before_d = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult first_d =
        engine.generate(engine.prepare_tokens(keep_d), greedy(4, false));
    if (first_d.generated_token_ids.size() != 4 ||
        first_d.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("exclusive occupancy third chat did not FullReset");
    }
    if (engine.runtime_stats().kv_ram_captures <= captures_before_d) {
        return fail("exclusive occupancy third chat did not recapture the VRAM resident");
    }
    const ninfer::MemorySummary after_recapture = engine.memory_summary();
    if (after_recapture.kv_ram_entry_count != 2) {
        std::cerr << "exclusive occupancy after recapture: entries="
                  << after_recapture.kv_ram_entry_count << ", expected 2\n";
        return 1;
    }
    const std::vector<ninfer::TokenId> recapture_a =
        resume_prefix(after_hit_a, resident_a.generated_token_ids);
    const auto restores_before_recapture = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_again =
        engine.generate(engine.prepare_tokens(recapture_a), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_again, static_cast<std::uint32_t>(recapture_a.size()),
                                      "exclusive occupancy recapture");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before_recapture + 1) {
        return fail("exclusive occupancy recapture did not restore from host RAM");
    }
    return 0;
}

int exercise_site2(ninfer::Engine& engine) {
    return capture_then_hit(engine, tokens_a(), tokens_c(), "site 2");
}

int exercise_negative(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> keep{100, 101, 102, 103, 104, 105};
    const auto evictor = tokens_c();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(4, false));
    if (first.generated_token_ids.size() != 4) { return fail("negative source did not complete"); }
    (void)engine.generate(engine.prepare_tokens(evictor), greedy(2, false));
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const ninfer::GenerationResult denied =
        engine.generate(engine.prepare_tokens(history), greedy(2, false));
    if (denied.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        denied.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        denied.reused_prompt_tokens != 0) {
        return fail("allow_prefix_reuse=false reused a RAM entry");
    }
    return 0;
}

int exercise_checkpoint(ninfer::Engine& engine) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto input = [&]() {
        ninfer::PromptInput prompt;
        prompt.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        prompt.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return prompt;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input()), greedy(4, false));
    if (first.generated_token_ids.size() != 4) {
        return fail("checkpoint source request did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(2, false));
    ninfer::PromptInput replay = input();
    replay.options.preserve_thinking = false;
    const auto restores_before       = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit = engine.generate(engine.prepare(std::move(replay)), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        (hit.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint &&
         hit.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnCheckpoint) ||
        hit.reused_prompt_tokens == 0) {
        std::cerr << "RAM checkpoint restore failed: source="
                  << static_cast<int>(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("RAM checkpoint restore did not increment kv_ram_restores");
    }
    return 0;
}

int exercise_partial_prefill(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> keep{200, 201, 202, 203, 204, 205};
    const ninfer::GenerationResult first = engine.generate(engine.prepare_tokens(keep), greedy(4, false));
    if (first.generated_token_ids.size() != 4) {
        return fail("partial-prefill source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(2, false));

    std::vector<ninfer::TokenId> long_prompt = resume_prefix(keep, first.generated_token_ids);
    long_prompt.insert(long_prompt.end(), 400, 198);
    const auto restores_before = engine.runtime_stats().kv_ram_restores;
    std::atomic<bool> stop{false};
    ninfer::CancellationView cancel([&] {
        if (engine.runtime_stats().kv_ram_restores > restores_before) { stop.store(true); }
        return stop.load();
    });
    const ninfer::GenerationResult cancelled =
        engine.generate(engine.prepare_tokens(long_prompt), greedy(2, true), nullptr, cancel);
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        return fail("partial-prefill request was not cancelled");
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("partial-prefill cancel did not consume the RAM entry");
    }
    const ninfer::GenerationResult again =
        engine.generate(engine.prepare_tokens(long_prompt), greedy(2, true));
    if (again.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        again.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("matching prompt after consume did not FullReset");
    }
    return 0;
}

int exercise_site1(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 256, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep                          = tokens_a();
    const ninfer::GenerationResult first     = engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) { return fail("site 1 source did not complete"); }
    const auto captures_before               = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare_tokens(tokens_b(200)), greedy(8, false));
    if (other.generated_token_ids.size() != 8) { return fail("site 1 evictor did not complete"); }
    if (engine.runtime_stats().kv_ram_captures <= captures_before) {
        return fail("site 1 eviction did not capture the displaced chat");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const auto restores_before                 = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc = expect_ram_hit(hit, static_cast<std::uint32_t>(history.size()), "site 1");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("site 1 did not restore from RAM");
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        return fail("site 1 RAM reuse changed greedy output");
    }
    return 0;
}

int exercise_vram_wins(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("VRAM-wins source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    const ninfer::GenerationResult again =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (again.generated_token_ids.size() != 8) {
        return fail("VRAM-wins rebuild did not complete");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, again.generated_token_ids);
    const auto restores_before                 = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit.reused_prompt_tokens != history.size() ||
        hit.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().kv_ram_restores != restores_before) {
        std::cerr << "equal reuse did not keep VRAM: source="
                  << static_cast<int>(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << " restores="
                  << engine.runtime_stats().kv_ram_restores << '\n';
        return 1;
    }
    return 0;
}

int exercise_longer_ram_beats_vram(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("longer-RAM source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    const ninfer::GenerationResult shorter =
        engine.generate(engine.prepare_tokens(keep), greedy(2, false));
    if (shorter.generated_token_ids.size() != 2) {
        return fail("longer-RAM shorter VRAM rebuild did not complete");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const auto restores_before                 = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(4, true));
    if (const int rc = expect_ram_hit(hit, static_cast<std::uint32_t>(history.size()),
                                      "longer RAM beats shorter VRAM");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        std::cerr << "longer RAM did not restore: restores="
                  << engine.runtime_stats().kv_ram_restores << '\n';
        return 1;
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(history), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        return fail("longer-RAM reuse changed greedy output");
    }
    return 0;
}

int exercise_site3_victim(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const auto keep_b = tokens_c();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(keep_b), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || first_b.generated_token_ids.size() != 4) {
        return fail("site-3 sources did not complete");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const auto captures_before                   = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult ram_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_a, static_cast<std::uint32_t>(history_a.size()),
                                      "site 3 victim");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_captures != captures_before + 1) {
        return fail("site 3 RAM restore did not capture the covered dirty lane");
    }
    const std::vector<ninfer::TokenId> history_b = resume_prefix(keep_b, first_b.generated_token_ids);
    const ninfer::GenerationResult ram_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_b, static_cast<std::uint32_t>(history_b.size()),
                                      "site 3 covered chat");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_suffix_prefill(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("suffix-prefill source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const std::vector<ninfer::TokenId> continued = concat(history, {198, 198, 198, 198});
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(continued), greedy(4, true));
    if (const int rc =
            expect_suffix_hit(hit, ninfer::PrefixReuseSource::HostRam,
                              static_cast<std::uint32_t>(history.size()),
                              static_cast<std::uint32_t>(continued.size()), "suffix RAM restore");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("suffix RAM restore did not increment kv_ram_restores");
    }
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult next =
        engine.generate(engine.prepare_tokens(tokens_d()), greedy(4, false));
    if (next.generated_token_ids.size() != 4 ||
        next.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().kv_ram_captures <= captures_before) {
        return fail("next FullReset after suffix restore did not capture the restored lane");
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(continued), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        return fail("suffix RAM reuse changed greedy output");
    }
    return 0;
}

int exercise_ram_disabled(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, 0));
    if (const int rc = verify_ram_tier(engine, 0); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(4, false));
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(2, false));
    if (engine.runtime_stats().kv_ram_captures != 0) {
        return fail("RAM-disabled engine captured a retained bundle");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const ninfer::GenerationResult miss =
        engine.generate(engine.prepare_tokens(history), greedy(2, true));
    if (miss.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        miss.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("RAM-disabled engine reused a host entry");
    }
    return 0;
}

int exercise_queued_ram_hit(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("queued RAM source did not complete");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    ninfer::GenerationHandle evictor =
        engine.submit(engine.prepare_tokens(tokens_c()), greedy(8, false));
    ninfer::GenerationHandle queued =
        engine.submit(engine.prepare_tokens(history), greedy(2, true));
    const ninfer::GenerationResult other = evictor.wait();
    const ninfer::GenerationResult hit   = queued.wait();
    if (other.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("queued evictor did not FullReset");
    }
    if (const int rc = expect_ram_hit(hit, static_cast<std::uint32_t>(history.size()),
                                      "queued matcher");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_c2_keeps_first_vram(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || engine.runtime_stats().running_requests != 0) {
        return fail("C=2 sequential first chat did not complete alone");
    }
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (second.generated_token_ids.size() != 4 ||
        second.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().running_requests != 0) {
        return fail("C=2 sequential second chat did not FullReset alone");
    }
    if (engine.runtime_stats().kv_ram_captures != captures_before) {
        return fail("C=2 sequential second chat captured the first retained lane instead of the empty lane");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult exact_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_exact_hit(exact_a, static_cast<std::uint32_t>(history_a.size()),
                                        "C=2 sequential first-chat exact resume");
        rc != 0) {
        return rc;
    }
    if (exact_a.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        engine.runtime_stats().kv_ram_restores != restores_before ||
        engine.runtime_stats().running_requests != 0) {
        std::cerr << "C=2 sequential first-chat exact resume used source="
                  << static_cast<int>(exact_a.prefix_reuse_source) << '\n';
        return 1;
    }
    return 0;
}

int exercise_c2_continue_refreshes_recency(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || first_b.generated_token_ids.size() != 4 ||
        engine.runtime_stats().kv_ram_captures != 0) {
        return fail("C=2 continue-A setup did not keep A and B in VRAM");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const std::vector<ninfer::TokenId> history_b =
        resume_prefix(tokens_c(), first_b.generated_token_ids);
    const ninfer::GenerationResult exact_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_exact_hit(exact_a, static_cast<std::uint32_t>(history_a.size()),
                                        "C=2 continue-A exact resume");
        rc != 0) {
        return rc;
    }
    const std::vector<ninfer::TokenId> continued_a =
        concat(resume_prefix(history_a, exact_a.generated_token_ids), {198, 198, 198, 198});
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(continued_a), greedy(4, true));
    if (const int rc = expect_suffix_hit(
            hit, ninfer::PrefixReuseSource::VramResident,
            static_cast<std::uint32_t>(continued_a.size() - 4),
            static_cast<std::uint32_t>(continued_a.size()), "C=2 continue-A new message");
        rc != 0) {
        return rc;
    }
    const auto captures_before_third = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare_tokens(tokens_e()), greedy(2, false));
    if (third.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().kv_ram_captures != captures_before_third + 1) {
        return fail("C=2 continue-A third chat did not cover one dirty lane");
    }
    const std::vector<ninfer::TokenId> after_a =
        resume_prefix(continued_a, hit.generated_token_ids);
    const ninfer::GenerationResult still_a =
        engine.generate(engine.prepare_tokens(after_a), greedy(2, true));
    if (still_a.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=2 continue-A spilled the MRU chat: source="
                  << static_cast<int>(still_a.prefix_reuse_source) << '\n';
        return 1;
    }
    const auto restores_before_b = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_b, static_cast<std::uint32_t>(history_b.size()),
                                      "C=2 continue-A spilled older B");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before_b + 1) {
        return fail("C=2 continue-A third chat did not capture the older B lane");
    }
    return 0;
}

int exercise_c2_lru_covers_oldest(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || first_b.generated_token_ids.size() != 4 ||
        engine.runtime_stats().kv_ram_captures != 0) {
        return fail("C=2 LRU oldest setup did not keep A and B in VRAM");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const std::vector<ninfer::TokenId> history_b =
        resume_prefix(tokens_c(), first_b.generated_token_ids);
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare_tokens(tokens_e()), greedy(2, false));
    if (third.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().kv_ram_captures != captures_before + 1) {
        return fail("C=2 LRU oldest third chat did not cover one dirty lane");
    }
    const ninfer::GenerationResult still_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (still_b.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=2 LRU oldest spilled newer B: source="
                  << static_cast<int>(still_b.prefix_reuse_source) << '\n';
        return 1;
    }
    const auto restores_before = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc =
            expect_ram_hit(ram_a, static_cast<std::uint32_t>(history_a.size()), "C=2 LRU oldest A");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("C=2 LRU oldest did not capture A");
    }
    return 0;
}

int exercise_c2_ram_covers_lru_dirty(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(8, false));
    if (first_a.generated_token_ids.size() != 8) {
        return fail("C=2 RAM LRU source A did not complete");
    }
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    const ninfer::GenerationResult first_d =
        engine.generate(engine.prepare_tokens(tokens_d()), greedy(4, false));
    if (first_b.generated_token_ids.size() != 4 || first_d.generated_token_ids.size() != 4 ||
        engine.runtime_stats().kv_ram_captures == 0) {
        return fail("C=2 RAM LRU did not spill A behind B and D");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const std::vector<ninfer::TokenId> history_b =
        resume_prefix(tokens_c(), first_b.generated_token_ids);
    const std::vector<ninfer::TokenId> history_d =
        resume_prefix(tokens_d(), first_d.generated_token_ids);
    const auto restores_before = engine.runtime_stats().kv_ram_restores;
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult ram_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_a, static_cast<std::uint32_t>(history_a.size()),
                                      "C=2 RAM LRU restore A");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1 ||
        engine.runtime_stats().kv_ram_captures != captures_before + 1) {
        return fail("C=2 RAM LRU restore did not capture the dirty victim");
    }
    const ninfer::GenerationResult still_d =
        engine.generate(engine.prepare_tokens(history_d), greedy(2, true));
    if (still_d.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
        std::cerr << "C=2 RAM LRU spilled newer D: source="
                  << static_cast<int>(still_d.prefix_reuse_source) << '\n';
        return 1;
    }
    const auto restores_before_b = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_b, static_cast<std::uint32_t>(history_b.size()),
                                      "C=2 RAM LRU covered older B");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before_b + 1) {
        return fail("C=2 RAM LRU did not leave older B on the host");
    }
    return 0;
}

int exercise_c3_keeps_empty_lane(const char* artifact) {
    ninfer::Engine engine(pooled_c3_options(artifact));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a = pad_tokens(tokens_a(), 128);
    const auto keep_b = pad_tokens(tokens_c(), 128);
    const auto keep_c = pad_tokens(tokens_d(), 64);
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(8, false));
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(keep_b), greedy(8, false));
    if (first_a.generated_token_ids.size() != 8 || first_b.generated_token_ids.size() != 8 ||
        engine.runtime_stats().running_requests != 0) {
        return fail("C=3 sequential first two chats did not complete alone");
    }
    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare_tokens(keep_c), greedy(2, false));
    if (third.generated_token_ids.size() != 2 ||
        third.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        engine.runtime_stats().running_requests != 0) {
        return fail("C=3 sequential third chat did not FullReset alone");
    }
    if (engine.runtime_stats().kv_ram_captures != captures_before) {
        return fail("C=3 sequential third chat captured a retained lane instead of the empty lane");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const std::vector<ninfer::TokenId> history_b = resume_prefix(keep_b, first_b.generated_token_ids);
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit_a =
        engine.generate(engine.prepare_tokens(concat(history_a, {198, 198})), greedy(2, true));
    const ninfer::GenerationResult hit_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (hit_a.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        hit_b.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident ||
        engine.runtime_stats().kv_ram_restores != restores_before) {
        std::cerr << "C=3 sequential resume left VRAM: A source="
                  << static_cast<int>(hit_a.prefix_reuse_source) << " B source="
                  << static_cast<int>(hit_b.prefix_reuse_source)
                  << " restores=" << engine.runtime_stats().kv_ram_restores << '\n';
        return 1;
    }
    return 0;
}

int exercise_site3_and_pass1(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep_a                        = tokens_a();
    const auto keep_b                        = tokens_c();
    const ninfer::GenerationResult first_a   = engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    const ninfer::GenerationResult first_b   = engine.generate(engine.prepare_tokens(keep_b), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || first_b.generated_token_ids.size() != 4) {
        return fail("pass-1 sources did not complete");
    }
    const ninfer::GenerationResult other =
        engine.generate(engine.prepare_tokens(tokens_b(32)), greedy(2, false));
    if (other.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("pass-1 displacer did not FullReset a retained lane");
    }
    const std::vector<ninfer::TokenId> history_a =
        resume_prefix(keep_a, first_a.generated_token_ids);
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_hit =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_hit, static_cast<std::uint32_t>(history_a.size()),
                                      "site 3 / pass 1");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("site 3 did not restore from RAM");
    }
    const std::vector<ninfer::TokenId> history_b =
        resume_prefix(keep_b, first_b.generated_token_ids);
    const ninfer::GenerationResult other_hit =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (other_hit.reused_prompt_tokens != history_b.size() ||
        other_hit.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        (other_hit.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident &&
         other_hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam)) {
        std::cerr << "pass-1 other chat was not reused: source="
                  << static_cast<int>(other_hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(other_hit.prefix_reuse_path)
                  << " reused=" << other_hit.reused_prompt_tokens << '\n';
        return 1;
    }

    const auto overlap_captures = engine.runtime_stats().kv_ram_captures;
    const std::vector<ninfer::TokenId> retained_a =
        concat(history_a, ram_hit.generated_token_ids);
    const std::vector<ninfer::TokenId> retained_b =
        concat(history_b, other_hit.generated_token_ids);
    ninfer::GenerationHandle first_inflight =
        engine.submit(engine.prepare_tokens(tokens_d()), greedy(8, false));
    ninfer::GenerationHandle second_inflight =
        engine.submit(engine.prepare_tokens(tokens_e()), greedy(8, false));
    const ninfer::GenerationResult overlap_d = first_inflight.wait();
    const ninfer::GenerationResult overlap_e = second_inflight.wait();
    if (overlap_d.generated_token_ids.size() != 8 || overlap_e.generated_token_ids.size() != 8) {
        return fail("overlapping concurrency-2 submits did not complete");
    }
    if (engine.runtime_stats().kv_ram_captures < overlap_captures + 1) {
        std::cerr << "overlapping admits captured "
                  << engine.runtime_stats().kv_ram_captures - overlap_captures
                  << " entries, expected at least 1\n";
        return 1;
    }
    const auto overlap_restores = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult ram_a =
        engine.generate(engine.prepare_tokens(retained_a), greedy(2, true));
    const ninfer::GenerationResult ram_b =
        engine.generate(engine.prepare_tokens(retained_b), greedy(2, true));
    if (ram_a.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        ram_a.reused_prompt_tokens == 0 ||
        ram_a.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << "overlap A reuse_source is "
                  << static_cast<int>(ram_a.prefix_reuse_source) << " reused "
                  << ram_a.reused_prompt_tokens << '\n';
        return 1;
    }
    if (ram_b.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        ram_b.reused_prompt_tokens == 0 ||
        ram_b.prefix_reuse_path == ninfer::PrefixReusePath::FullReset) {
        std::cerr << "overlap B reuse_source is "
                  << static_cast<int>(ram_b.prefix_reuse_source) << " reused "
                  << ram_b.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != overlap_restores + 2) {
        return fail("overlapping concurrency-2 capture did not restore both chats from RAM");
    }
    return 0;
}

int exercise_duplicate_ram_submit(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 2, 256, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep                      = tokens_a();
    const ninfer::GenerationResult first = engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("duplicate-submit source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_b(200)), greedy(8, false));
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    ninfer::GenerationHandle first_hit =
        engine.submit(engine.prepare_tokens(history), greedy(2, true));
    ninfer::GenerationHandle second_hit =
        engine.submit(engine.prepare_tokens(history), greedy(2, true));
    const ninfer::GenerationResult ram_first  = first_hit.wait();
    const ninfer::GenerationResult ram_second = second_hit.wait();
    if (ram_first.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
        return fail("first duplicate RAM submit did not restore from RAM");
    }
    if (ram_second.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam) {
        return fail("second duplicate RAM submit claimed the same pinned entry");
    }
    return 0;
}

int exercise_oversize_drop(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamDropBytes));
    if (const int rc = verify_ram_tier(engine, kRamDropBytes); rc != 0) { return rc; }
    const auto keep                          = tokens_a();
    const ninfer::GenerationResult first     = engine.generate(engine.prepare_tokens(keep), greedy(4, false));
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(2, false));
    if (engine.runtime_stats().kv_ram_drops == 0) {
        return fail("1 MiB RAM budget did not drop the captured bundle");
    }
    if (engine.runtime_stats().kv_ram_evictions != 0) {
        return fail("1 MiB oversize drop counted as FIFO eviction");
    }
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const ninfer::GenerationResult miss =
        engine.generate(engine.prepare_tokens(history), greedy(2, true));
    if (miss.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        miss.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("dropped RAM entry was reused");
    }
    return 0;
}

int exercise_mtp(const char* artifact) {
    ninfer::Engine engine(mtp_options(artifact));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    if (const int rc = capture_then_hit(engine, tokens_a(), tokens_c(), "MTP site 2"); rc != 0) {
        return rc;
    }
    const auto keep = tokens_d();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(8, false));
    if (first.generated_token_ids.size() != 8) {
        return fail("MTP suffix source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_e()), greedy(4, false));
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const std::vector<ninfer::TokenId> continued = concat(history, {198, 198, 198, 198});
    const auto restores_before                   = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(continued), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        hit.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        hit.reused_prompt_tokens != history.size()) {
        std::cerr << "MTP suffix RAM restore failed: source="
                  << static_cast<int>(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("MTP suffix RAM restore did not increment kv_ram_restores");
    }
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(continued), greedy(4, false));
    if (hit.generated_token_ids != baseline.generated_token_ids) {
        return fail("MTP suffix RAM reuse changed greedy output");
    }
    return 0;
}

int exercise_checkpoint_dirty_lane(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto input = [&]() {
        ninfer::PromptInput prompt;
        prompt.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        prompt.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return prompt;
    };
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input()), greedy(4, false));
    if (first.generated_token_ids.size() != 4) {
        return fail("dirty-lane checkpoint source did not complete");
    }
    const ninfer::GenerationResult occupant =
        engine.generate(engine.prepare_tokens(tokens_c()), greedy(4, false));
    if (occupant.generated_token_ids.size() != 4) {
        return fail("dirty-lane checkpoint occupant did not complete");
    }
    ninfer::PromptInput replay = input();
    replay.options.preserve_thinking = false;
    const auto captures_before       = engine.runtime_stats().kv_ram_captures;
    const auto restores_before       = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare(std::move(replay)), greedy(4, true));
    if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        (hit.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint &&
         hit.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnCheckpoint) ||
        hit.reused_prompt_tokens == 0) {
        std::cerr << "dirty-lane checkpoint restore failed: source="
                  << static_cast<int>(hit.prefix_reuse_source)
                  << " path=" << static_cast<int>(hit.prefix_reuse_path)
                  << " reused=" << hit.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1 ||
        engine.runtime_stats().kv_ram_captures != captures_before + 1) {
        return fail("dirty-lane checkpoint restore did not capture the occupant and restore");
    }
    return 0;
}

int exercise_spill_drop(const char* artifact) {
    constexpr std::size_t kOneEntryBytes = 220ULL * 1024ULL * 1024ULL;
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kOneEntryBytes));
    if (const int rc = verify_ram_tier(engine, kOneEntryBytes); rc != 0) { return rc; }
    const auto keep_a = tokens_a();
    const auto keep_b = tokens_c();
    const ninfer::GenerationResult first_a =
        engine.generate(engine.prepare_tokens(keep_a), greedy(4, false));
    const ninfer::GenerationResult first_b =
        engine.generate(engine.prepare_tokens(keep_b), greedy(4, false));
    if (first_a.generated_token_ids.size() != 4 || first_b.generated_token_ids.size() != 4) {
        return fail("spill-drop sources did not complete");
    }
    const std::vector<ninfer::TokenId> history_a = resume_prefix(keep_a, first_a.generated_token_ids);
    const ninfer::MemorySummary memory            = engine.memory_summary();
    if (memory.kv_ram_used_bytes == 0 || memory.kv_ram_used_bytes * 2 <= kOneEntryBytes) {
        std::cerr << "spill-drop budget still fits two entries: used="
                  << memory.kv_ram_used_bytes << " cap=" << kOneEntryBytes << '\n';
        return 1;
    }
    const auto drops_before                      = engine.runtime_stats().kv_ram_drops;
    const ninfer::GenerationResult ram_a =
        engine.generate(engine.prepare_tokens(history_a), greedy(2, true));
    if (const int rc = expect_ram_hit(ram_a, static_cast<std::uint32_t>(history_a.size()),
                                      "spill-drop restore");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_drops <= drops_before) {
        return fail("dirty-lane spill that did not fit still stored the occupant");
    }
    const std::vector<ninfer::TokenId> history_b = resume_prefix(keep_b, first_b.generated_token_ids);
    const ninfer::GenerationResult miss_b =
        engine.generate(engine.prepare_tokens(history_b), greedy(2, true));
    if (miss_b.prefix_reuse_source != ninfer::PrefixReuseSource::None ||
        miss_b.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "dropped occupant was reused: source="
                  << static_cast<int>(miss_b.prefix_reuse_source)
                  << " path=" << static_cast<int>(miss_b.prefix_reuse_path) << '\n';
        return 1;
    }
    return 0;
}

int exercise_teardown_after_restore(const char* artifact) {
    ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const auto keep = tokens_a();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(keep), greedy(4, false));
    if (first.generated_token_ids.size() != 4) {
        return fail("teardown source did not complete");
    }
    (void)engine.generate(engine.prepare_tokens(tokens_c()), greedy(2, false));
    const std::vector<ninfer::TokenId> history = resume_prefix(keep, first.generated_token_ids);
    const ninfer::GenerationResult hit =
        engine.generate(engine.prepare_tokens(history), greedy(2, true));
    if (const int rc = expect_ram_hit(hit, static_cast<std::uint32_t>(history.size()), "teardown");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_shared_pool_c3(const char* artifact) {
    ninfer::Engine engine(pooled_c3_options(artifact));
    if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
    const ninfer::MemorySummary memory = engine.memory_summary();
    if (memory.max_context != 512 || memory.kv_capacity != 640 ||
        memory.kv_capacity_page_groups != 10) {
        std::cerr << "C=3 pool is max_context=" << memory.max_context
                  << " kv_capacity=" << memory.kv_capacity
                  << " pages=" << memory.kv_capacity_page_groups << '\n';
        return 1;
    }

    const auto small_a = pad_tokens(tokens_a(), 128);
    const auto small_b = pad_tokens(tokens_c(), 128);
    const auto small_c = pad_tokens(tokens_e(), 128);
    const auto tiny_d  = pad_tokens(tokens_d(), 64);
    const auto extra_f = pad_tokens({700, 701, 702, 703, 704, 705}, 200);

    ninfer::GenerationHandle wave1_a = engine.submit(engine.prepare_tokens(small_a), greedy(8, false));
    ninfer::GenerationHandle wave1_b = engine.submit(engine.prepare_tokens(small_b), greedy(8, false));
    ninfer::GenerationHandle wave1_c = engine.submit(engine.prepare_tokens(small_c), greedy(8, false));
    std::uint32_t max_prefilling     = 0;
    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) { return stats.running_requests == 3; },
            "C=3 wave 1 three-in-flight")) {
        return 1;
    }
    const ninfer::GenerationResult first_a = wave1_a.wait();
    const ninfer::GenerationResult first_b = wave1_b.wait();
    const ninfer::GenerationResult first_c = wave1_c.wait();
    if (first_a.generated_token_ids.size() != 8 || first_b.generated_token_ids.size() != 8 ||
        first_c.generated_token_ids.size() != 8) {
        return fail("C=3 wave 1 did not generate eight tokens on each chat");
    }
    if (first_a.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        first_b.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        first_c.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        first_a.reused_prompt_tokens != 0 || first_b.reused_prompt_tokens != 0 ||
        first_c.reused_prompt_tokens != 0) {
        return fail("C=3 wave 1 reused a prefix");
    }
    if (engine.runtime_stats().kv_ram_captures != 0 || engine.runtime_stats().kv_ram_restores != 0) {
        return fail("C=3 wave 1 captured or restored while three 3-page chats still fit");
    }
    if (max_prefilling > 1) { return fail("C=3 wave 1 ran more than one prefill owner"); }

    const std::vector<ninfer::TokenId> history_a = resume_prefix(small_a, first_a.generated_token_ids);
    const std::vector<ninfer::TokenId> history_b = resume_prefix(small_b, first_b.generated_token_ids);
    const std::vector<ninfer::TokenId> history_c = resume_prefix(small_c, first_c.generated_token_ids);
    const auto continued_a                       = pad_tokens(history_a, 200);
    const auto continued_b                       = pad_tokens(history_b, 200);
    const auto continued_c                       = pad_tokens(history_c, 200);
    if (history_a.size() >= continued_a.size() || history_b.size() >= continued_b.size() ||
        history_c.size() >= continued_c.size()) {
        return fail("C=3 continuation fixtures are not suffix prompts");
    }

    const auto restores_after_wave1 = engine.runtime_stats().kv_ram_restores;
    const auto captures_after_wave1 = engine.runtime_stats().kv_ram_captures;

    ninfer::GenerationHandle large_a =
        engine.submit(engine.prepare_tokens(continued_a), greedy(48, true));
    ninfer::GenerationHandle large_b =
        engine.submit(engine.prepare_tokens(continued_b), greedy(48, true));
    ninfer::GenerationHandle large_c =
        engine.submit(engine.prepare_tokens(continued_c), greedy(48, true));
    ninfer::GenerationHandle backfill_d =
        engine.submit(engine.prepare_tokens(tiny_d), greedy(2, false));
    ninfer::GenerationHandle blocked_f =
        engine.submit(engine.prepare_tokens(extra_f), greedy(8, false));

    if (!wait_scheduler(
            engine, &max_prefilling,
            [](const ninfer::RuntimeStats& stats) {
                return stats.running_requests >= 3 && stats.waiting_requests >= 2;
            },
            "C=3 two large plus fits-now backfill")) {
        return 1;
    }
    const ninfer::RuntimeStats occupancy = engine.runtime_stats();
    if (occupancy.kv_ram_captures <= captures_after_wave1) {
        return fail("C=3 second large admission did not capture the third retained chat");
    }
    if (occupancy.kv_ram_restores != restores_after_wave1) {
        return fail("C=3 protected large head restored before the fitting backfill finished");
    }

    const ninfer::GenerationResult tiny   = backfill_d.wait();
    const ninfer::RuntimeStats after_tiny = engine.runtime_stats();
    if (tiny.generated_token_ids.size() != 2 ||
        tiny.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("C=3 tiny backfill did not FullReset two tokens");
    }
    if (after_tiny.kv_ram_captures <= captures_after_wave1) {
        return fail("C=3 second large admission did not capture the third retained chat");
    }
    if (after_tiny.kv_ram_restores != restores_after_wave1) {
        return fail("C=3 protected large head restored before the fitting backfill finished");
    }
    if (after_tiny.waiting_requests == 0) {
        return fail("C=3 tiny backfill drained the protected head");
    }
    if (max_prefilling > 1) { return fail("C=3 wave 2 ran more than one prefill owner"); }

    const ninfer::GenerationResult restored = large_c.wait();
    if (const int rc = expect_suffix_hit(restored, ninfer::PrefixReuseSource::HostRam,
                                         static_cast<std::uint32_t>(history_c.size()),
                                         static_cast<std::uint32_t>(continued_c.size()),
                                         "C=3 RAM suffix");
        rc != 0) {
        return rc;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_after_wave1 + 1) {
        std::cerr << "C=3 RAM suffix restores=" << engine.runtime_stats().kv_ram_restores
                  << ", expected " << restores_after_wave1 + 1 << '\n';
        return 1;
    }

    const ninfer::GenerationResult vram_a = large_a.wait();
    const ninfer::GenerationResult vram_b = large_b.wait();
    const ninfer::GenerationResult late_f = blocked_f.wait();
    if (const int rc = expect_suffix_hit(vram_a, ninfer::PrefixReuseSource::VramResident,
                                         static_cast<std::uint32_t>(history_a.size()),
                                         static_cast<std::uint32_t>(continued_a.size()),
                                         "C=3 VRAM suffix A");
        rc != 0) {
        return rc;
    }
    if (const int rc = expect_suffix_hit(vram_b, ninfer::PrefixReuseSource::VramResident,
                                         static_cast<std::uint32_t>(history_b.size()),
                                         static_cast<std::uint32_t>(continued_b.size()),
                                         "C=3 VRAM suffix B");
        rc != 0) {
        return rc;
    }
    if (late_f.generated_token_ids.size() != 8 ||
        late_f.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        return fail("C=3 blocked 4-page request did not FullReset after the protected head");
    }
    if (restored.timings.first_token_seconds >= late_f.timings.first_token_seconds) {
        std::cerr << "C=3 blocked request first_token=" << late_f.timings.first_token_seconds
                  << " preceded protected head first_token="
                  << restored.timings.first_token_seconds << '\n';
        return 1;
    }
    if (tiny.timings.first_token_seconds >= restored.timings.first_token_seconds) {
        std::cerr << "C=3 tiny backfill first_token=" << tiny.timings.first_token_seconds
                  << " did not precede protected head first_token="
                  << restored.timings.first_token_seconds << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(continued_c), greedy(48, false));
    if (restored.generated_token_ids != baseline.generated_token_ids) {
        return fail("C=3 RAM suffix reuse changed greedy output");
    }

    const std::vector<ninfer::TokenId> exact_c =
        resume_prefix(continued_c, restored.generated_token_ids);
    const ninfer::GenerationResult exact =
        engine.generate(engine.prepare_tokens(exact_c), greedy(4, true));
    if (const int rc =
            expect_exact_hit(exact, static_cast<std::uint32_t>(exact_c.size()), "C=3 exact resume");
        rc != 0) {
        return rc;
    }
    return 0;
}

int exercise_artifact(const char* artifact) {
    {
        ninfer::Engine engine(ordinary_options(artifact, 1, 4096, kRamHitBytes, 128));
        if (const int rc = verify_ram_tier(engine, kRamHitBytes); rc != 0) { return rc; }
        std::cerr << "ram_real: site 2 / negative / checkpoint / cancel\n";
        if (const int rc = exercise_site2(engine); rc != 0) { return rc; }
        if (const int rc = exercise_negative(engine); rc != 0) { return rc; }
        if (const int rc = exercise_checkpoint(engine); rc != 0) { return rc; }
        if (const int rc = exercise_partial_prefill(engine); rc != 0) { return rc; }
    }
    std::cerr << "ram_real: exclusive FIFO occupancy\n";
    if (const int rc = exercise_exclusive_occupancy(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: site 1 concurrency=2\n";
    if (const int rc = exercise_site1(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: shared pool C=3\n";
    if (const int rc = exercise_shared_pool_c3(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: VRAM wins equal reuse\n";
    if (const int rc = exercise_vram_wins(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: longer RAM beats shorter VRAM\n";
    if (const int rc = exercise_longer_ram_beats_vram(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: site 3 victim capture\n";
    if (const int rc = exercise_site3_victim(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: suffix prefill after RAM restore\n";
    if (const int rc = exercise_suffix_prefill(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: RAM disabled\n";
    if (const int rc = exercise_ram_disabled(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: queued matcher\n";
    if (const int rc = exercise_queued_ram_hit(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: C=2 sequential empty lane keeps first chat in VRAM\n";
    if (const int rc = exercise_c2_keeps_first_vram(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: C=2 continue A refreshes recency so C covers B\n";
    if (const int rc = exercise_c2_continue_refreshes_recency(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: C=2 LRU covers oldest dirty lane\n";
    if (const int rc = exercise_c2_lru_covers_oldest(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: C=2 RAM restore covers LRU dirty lane\n";
    if (const int rc = exercise_c2_ram_covers_lru_dirty(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: C=3 sequential empty lane keeps first two chats in VRAM\n";
    if (const int rc = exercise_c3_keeps_empty_lane(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: site 3 / pass 1 / overlapping submit\n";
    if (const int rc = exercise_site3_and_pass1(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: duplicate RAM submit\n";
    if (const int rc = exercise_duplicate_ram_submit(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: oversize drop\n";
    if (const int rc = exercise_oversize_drop(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: dirty-lane checkpoint\n";
    if (const int rc = exercise_checkpoint_dirty_lane(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: spill drop of dirty-lane occupant\n";
    if (const int rc = exercise_spill_drop(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: teardown after restore\n";
    if (const int rc = exercise_teardown_after_restore(artifact); rc != 0) { return rc; }
    std::cerr << "ram_real: MTP\n";
    if (const int rc = exercise_mtp(artifact); rc != 0) { return rc; }
    return 0;
}

} // namespace

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if ((groupwise == nullptr || *groupwise == '\0') && (nvfp4 == nullptr || *nvfp4 == '\0')) {
        std::cout << "skip: neither NINFER_QWEN3_6_27B_WEIGHTS nor "
                     "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS is set\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (const int result = exercise_artifact(groupwise); result != 0) { return result; }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (const int result = exercise_artifact(nvfp4); result != 0) { return result; }
    }
    std::cout << "ok\n";
    return 0;
}
