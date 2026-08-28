#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct KvRamSnapshot {
    std::size_t capacity_bytes  = 0;
    // Sum of indexed Record::bytes, including a claimed-but-not-consumed pin.
    // Retired copy blocks still occupying the pin are excluded until reap.
    std::size_t used_bytes      = 0;
    std::size_t entry_count     = 0;
    std::uint64_t captures      = 0;
    std::uint64_t restores      = 0;
    std::uint64_t evictions     = 0;
    std::uint64_t drops         = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
