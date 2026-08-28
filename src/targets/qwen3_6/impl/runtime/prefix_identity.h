#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct PrefixHash128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    [[nodiscard]] friend bool operator==(const PrefixHash128& a, const PrefixHash128& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    [[nodiscard]] friend bool operator!=(const PrefixHash128& a, const PrefixHash128& b) noexcept {
        return !(a == b);
    }
};

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;

    [[nodiscard]] std::span<const std::uint8_t> token_types() const noexcept {
        return token_types_;
    }
    [[nodiscard]] std::span<const std::int32_t> positions(std::size_t axis) const {
        return positions_.at(axis);
    }
    [[nodiscard]] std::span<const VisionItem> vision_items() const noexcept { return vision_items_; }

    [[nodiscard]] std::size_t packed_bytes() const;
    void pack(void* dst) const;
    void unpack(const void* src, std::size_t bytes);

    void test_tamper_content_digest(std::size_t item, std::uint8_t byte);

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

[[nodiscard]] std::vector<PrefixHash128> prefix_hash_chain(const PreparedPromptData& prompt);

[[nodiscard]] PrefixHash128 prefix_hash_at(std::span<const TokenId> tokens,
                                           const ResidentPrefixIdentity& identity,
                                           std::size_t count);

// D17: RAM DFlash checkpoint reuse is gated on a captured backend image, not live sequence.kv.
[[nodiscard]] constexpr bool dflash_rewrite_checkpoint_ready(
    bool backend_image_present, std::uint32_t dflash_context_frontier,
    std::uint32_t reuse_base) noexcept {
    return backend_image_present && dflash_context_frontier >= reuse_base;
}

} // namespace ninfer::targets::qwen3_6::detail
