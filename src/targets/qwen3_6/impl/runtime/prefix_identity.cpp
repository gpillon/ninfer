#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

constexpr std::uint64_t kFnvPrime64  = 1099511628211ULL;
constexpr std::uint64_t kFnvOffset64 = 14695981039346656037ULL;
constexpr std::uint64_t kChain1Xor   = 0xA5A5A5A5A5A5A5A5ULL;

void mix_u8(PrefixHash128& hash, std::uint8_t byte) {
    hash.lo = (hash.lo ^ static_cast<std::uint64_t>(byte)) * kFnvPrime64;
    hash.hi = (hash.hi ^ static_cast<std::uint64_t>(byte)) * kFnvPrime64;
}

void mix_bytes(PrefixHash128& hash, const void* data, std::size_t bytes) {
    const auto* raw = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { mix_u8(hash, raw[i]); }
}

void mix_le32(PrefixHash128& hash, std::uint32_t value) {
    mix_u8(hash, static_cast<std::uint8_t>(value));
    mix_u8(hash, static_cast<std::uint8_t>(value >> 8));
    mix_u8(hash, static_cast<std::uint8_t>(value >> 16));
    mix_u8(hash, static_cast<std::uint8_t>(value >> 24));
}

void mix_le64(PrefixHash128& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        mix_u8(hash, static_cast<std::uint8_t>(value >> shift));
    }
}

void mix_i32(PrefixHash128& hash, std::int32_t value) {
    mix_le32(hash, static_cast<std::uint32_t>(value));
}

void mix_f64(PrefixHash128& hash, double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    mix_le64(hash, bits);
}

void mix_token(PrefixHash128& hash, TokenId token, std::uint8_t type, std::int32_t p0,
               std::int32_t p1, std::int32_t p2) {
    mix_i32(hash, token);
    mix_u8(hash, type);
    mix_i32(hash, p0);
    mix_i32(hash, p1);
    mix_i32(hash, p2);
}

void mix_vision_item(PrefixHash128& hash, const VisionItem& item) {
    mix_u8(hash, static_cast<std::uint8_t>(item.modality));
    mix_i32(hash, item.grid.temporal);
    mix_i32(hash, item.grid.height);
    mix_i32(hash, item.grid.width);
    mix_le64(hash, static_cast<std::uint64_t>(item.patch_begin));
    mix_le64(hash, static_cast<std::uint64_t>(item.patch_count));
    mix_bytes(hash, item.content_digest.data(), item.content_digest.size());
    mix_le32(hash, static_cast<std::uint32_t>(item.timestamps.size()));
    for (double timestamp : item.timestamps) { mix_f64(hash, timestamp); }
    mix_le32(hash, static_cast<std::uint32_t>(item.token_spans.size()));
    for (const TokenSpan& span : item.token_spans) {
        mix_le64(hash, static_cast<std::uint64_t>(span.begin));
        mix_le64(hash, static_cast<std::uint64_t>(span.count));
    }
}

std::size_t item_end(const VisionItem& item) {
    if (item.token_spans.empty()) { return 0; }
    const TokenSpan& last = item.token_spans.back();
    return last.begin + last.count;
}

PrefixHash128 initial_hash() {
    return PrefixHash128{kFnvOffset64, kFnvOffset64 ^ kChain1Xor};
}

struct Writer {
    std::uint8_t* p   = nullptr;
    std::uint8_t* end = nullptr;

    void u8(std::uint8_t value) {
        if (p >= end) { throw std::logic_error("prefix identity pack overflow"); }
        *p++ = value;
    }
    void u32(std::uint32_t value) {
        u8(static_cast<std::uint8_t>(value));
        u8(static_cast<std::uint8_t>(value >> 8));
        u8(static_cast<std::uint8_t>(value >> 16));
        u8(static_cast<std::uint8_t>(value >> 24));
    }
    void u64(std::uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) {
            u8(static_cast<std::uint8_t>(value >> shift));
        }
    }
    void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }
    void f64(double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void bytes(const void* data, std::size_t n) {
        const auto* raw = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { u8(raw[i]); }
    }
};

struct Reader {
    const std::uint8_t* p   = nullptr;
    const std::uint8_t* end = nullptr;

    [[nodiscard]] std::uint8_t u8() {
        if (p >= end) { throw std::logic_error("prefix identity unpack overflow"); }
        return *p++;
    }
    [[nodiscard]] std::uint32_t u32() {
        const std::uint32_t a = u8();
        const std::uint32_t b = u8();
        const std::uint32_t c = u8();
        const std::uint32_t d = u8();
        return a | (b << 8) | (c << 16) | (d << 24);
    }
    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>(u8()) << shift;
        }
        return value;
    }
    [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    [[nodiscard]] double f64() {
        const std::uint64_t bits = u64();
        double value             = 0;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    void bytes(void* data, std::size_t n) {
        auto* raw = static_cast<std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { raw[i] = u8(); }
    }
};

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

} // namespace

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    return true;
}

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      resident_tokens.begin()) &&
           resident_identity.matches(prompt, count);
}

std::size_t ResidentPrefixIdentity::packed_bytes() const {
    std::size_t bytes = 8; // token_count, vision_item_count
    bytes += token_types_.size();
    bytes += 3 * token_types_.size() * sizeof(std::int32_t);
    for (const VisionItem& item : vision_items_) {
        bytes += 1 + 12 + 16 + 32 + 4 + item.timestamps.size() * 8 + 4 +
                 item.token_spans.size() * 16;
    }
    return bytes;
}

void ResidentPrefixIdentity::pack(void* dst) const {
    if (dst == nullptr && packed_bytes() != 0) {
        throw std::invalid_argument("prefix identity pack destination is null");
    }
    auto* raw = static_cast<std::uint8_t*>(dst);
    Writer w{raw, raw + packed_bytes()};
    w.u32(static_cast<std::uint32_t>(token_types_.size()));
    w.u32(static_cast<std::uint32_t>(vision_items_.size()));
    w.bytes(token_types_.data(), token_types_.size());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::int32_t position : positions_[axis]) { w.i32(position); }
    }
    for (const VisionItem& item : vision_items_) {
        w.u8(static_cast<std::uint8_t>(item.modality));
        w.i32(item.grid.temporal);
        w.i32(item.grid.height);
        w.i32(item.grid.width);
        w.u64(static_cast<std::uint64_t>(item.patch_begin));
        w.u64(static_cast<std::uint64_t>(item.patch_count));
        w.bytes(item.content_digest.data(), item.content_digest.size());
        w.u32(static_cast<std::uint32_t>(item.timestamps.size()));
        for (double timestamp : item.timestamps) { w.f64(timestamp); }
        w.u32(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            w.u64(static_cast<std::uint64_t>(span.begin));
            w.u64(static_cast<std::uint64_t>(span.count));
        }
    }
    if (w.p != w.end) { throw std::logic_error("prefix identity pack size mismatch"); }
}

void ResidentPrefixIdentity::unpack(const void* src, std::size_t bytes) {
    if (src == nullptr && bytes != 0) {
        throw std::invalid_argument("prefix identity unpack source is null");
    }
    const auto* raw = static_cast<const std::uint8_t*>(src);
    Reader r{raw, raw + bytes};
    const std::uint32_t token_count = r.u32();
    const std::uint32_t item_count  = r.u32();
    token_types_.resize(token_count);
    r.bytes(token_types_.data(), token_types_.size());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        positions_[axis].resize(token_count);
        for (std::uint32_t i = 0; i < token_count; ++i) { positions_[axis][i] = r.i32(); }
    }
    vision_items_.resize(item_count);
    for (VisionItem& item : vision_items_) {
        item.modality      = static_cast<PromptModality>(r.u8());
        item.grid.temporal = r.i32();
        item.grid.height   = r.i32();
        item.grid.width    = r.i32();
        item.patch_begin   = static_cast<std::size_t>(r.u64());
        item.patch_count   = static_cast<std::size_t>(r.u64());
        r.bytes(item.content_digest.data(), item.content_digest.size());
        const std::uint32_t timestamp_count = r.u32();
        item.timestamps.resize(timestamp_count);
        for (double& timestamp : item.timestamps) { timestamp = r.f64(); }
        const std::uint32_t span_count = r.u32();
        item.token_spans.resize(span_count);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(r.u64());
            span.count = static_cast<std::size_t>(r.u64());
        }
    }
    if (r.p != r.end) { throw std::logic_error("prefix identity unpack trailing bytes"); }
}

void ResidentPrefixIdentity::test_tamper_content_digest(std::size_t item, std::uint8_t byte) {
    if (item >= vision_items_.size()) {
        throw std::out_of_range("prefix identity vision item is out of range");
    }
    vision_items_[item].content_digest[0] = byte;
}

std::vector<PrefixHash128> prefix_hash_chain(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    std::vector<PrefixHash128> chain(tokens + 1);
    chain[0]                 = initial_hash();
    std::size_t item_cursor  = 0;
    for (std::size_t k = 1; k <= tokens; ++k) {
        PrefixHash128 hash = chain[k - 1];
        const std::size_t i = k - 1;
        mix_token(hash, prompt.token_ids[i], prompt.token_types[i], prompt.positions[i],
                  prompt.positions[tokens + i], prompt.positions[2 * tokens + i]);
        while (item_cursor < prompt.vision_items.size()) {
            const VisionItem& item = prompt.vision_items[item_cursor];
            if (item.token_spans.empty()) {
                ++item_cursor;
                continue;
            }
            const std::size_t end = item_end(item);
            if (end != k) { break; }
            mix_vision_item(hash, item);
            ++item_cursor;
        }
        chain[k] = hash;
    }
    return chain;
}

PrefixHash128 prefix_hash_at(std::span<const TokenId> tokens, const ResidentPrefixIdentity& identity,
                             std::size_t count) {
    if (count > tokens.size() || count > identity.size()) {
        throw std::out_of_range("prefix hash count exceeds resident identity");
    }
    PrefixHash128 hash      = initial_hash();
    std::size_t item_cursor = 0;
    const auto items        = identity.vision_items();
    for (std::size_t k = 1; k <= count; ++k) {
        const std::size_t i = k - 1;
        mix_token(hash, tokens[i], identity.token_types()[i], identity.positions(0)[i],
                  identity.positions(1)[i], identity.positions(2)[i]);
        while (item_cursor < items.size()) {
            const VisionItem& item = items[item_cursor];
            if (item.token_spans.empty()) {
                ++item_cursor;
                continue;
            }
            const std::size_t end = item_end(item);
            if (end != k) { break; }
            mix_vision_item(hash, item);
            ++item_cursor;
        }
    }
    return hash;
}

} // namespace ninfer::targets::qwen3_6::detail
