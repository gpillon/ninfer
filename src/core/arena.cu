#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", cudaFree(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", cudaFreeHost(ptr));
        ptr = nullptr;
    }
}

} // namespace

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = cudaMemset(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err = cudaMemcpy(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ != nullptr && saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }

    base_ = ptr;
    cap_  = capacity_bytes;
    off_  = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) { free_device(base_); }
    base_ = other.base_;
    cap_  = other.cap_;
    off_  = other.off_;
    peak_ = other.peak_;
    owns_ = other.owns_;

    other.base_ = nullptr;
    other.cap_  = 0;
    other.off_  = 0;
    other.peak_ = 0;
    other.owns_ = true;
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept { off_ = 0; }

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMallocHost(&ptr, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMallocHost failed", err));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

HostPinnedArena::HostPinnedArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("HostPinnedArena capacity must be nonzero");
    }
    void* ptr             = nullptr;
    const cudaError_t err = cudaHostAlloc(&ptr, capacity_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaHostAlloc failed", err));
    }
    base_ = ptr;
    cap_  = capacity_bytes;
    free_.push_back(FreeSpan{0, capacity_bytes});
}

HostPinnedArena::~HostPinnedArena() { free_pinned(base_); }

HostPinnedArena::HostPinnedArena(HostPinnedArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), used_(other.used_), free_(std::move(other.free_)),
      live_(std::move(other.live_)) {
    other.base_ = nullptr;
    other.cap_  = 0;
    other.used_ = 0;
}

HostPinnedArena& HostPinnedArena::operator=(HostPinnedArena&& other) noexcept {
    if (this == &other) { return *this; }
    free_pinned(base_);
    base_       = other.base_;
    cap_        = other.cap_;
    used_       = other.used_;
    free_       = std::move(other.free_);
    live_       = std::move(other.live_);
    other.base_ = nullptr;
    other.cap_  = 0;
    other.used_ = 0;
    return *this;
}

void* HostPinnedArena::try_alloc(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("HostPinnedArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }
    if (bytes > cap_) { return nullptr; }

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base_);
    for (std::size_t index = 0; index < free_.size(); ++index) {
        const FreeSpan span              = free_[index];
        const std::uintptr_t span_addr   = checked_add_uintptr(base_addr, span.offset);
        const std::uintptr_t aligned_addr = align_up_addr(span_addr, align);
        const std::uintptr_t lead_addr    = aligned_addr - span_addr;
        if (lead_addr > std::numeric_limits<std::size_t>::max()) { continue; }
        const auto lead = static_cast<std::size_t>(lead_addr);
        if (lead > span.size || bytes > span.size - lead) { continue; }

        const std::size_t alloc_offset = span.offset + lead;
        const std::size_t tail         = span.size - lead - bytes;
        free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index));
        if (tail != 0) { insert_free(alloc_offset + bytes, tail); }
        if (lead != 0) { insert_free(span.offset, lead); }

        void* ptr         = static_cast<unsigned char*>(base_) + alloc_offset;
        live_[ptr]        = LiveBlock{alloc_offset, bytes};
        used_            += bytes;
        return ptr;
    }
    return nullptr;
}

bool HostPinnedArena::can_alloc(std::size_t bytes, std::size_t align) const {
    if (base_ == nullptr) { throw std::runtime_error("HostPinnedArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }
    if (bytes > cap_) { return false; }

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base_);
    for (const FreeSpan& span : free_) {
        const std::uintptr_t span_addr    = checked_add_uintptr(base_addr, span.offset);
        const std::uintptr_t aligned_addr = align_up_addr(span_addr, align);
        const std::uintptr_t lead_addr    = aligned_addr - span_addr;
        if (lead_addr > std::numeric_limits<std::size_t>::max()) { continue; }
        const auto lead = static_cast<std::size_t>(lead_addr);
        if (lead <= span.size && bytes <= span.size - lead) { return true; }
    }
    return false;
}

void HostPinnedArena::free(void* block) {
    if (block == nullptr) { throw std::invalid_argument("HostPinnedArena free of null"); }
    const auto it = live_.find(block);
    if (it == live_.end()) { throw std::logic_error("HostPinnedArena free of unknown block"); }
    const LiveBlock live = it->second;
    live_.erase(it);
    used_ -= live.size;
    insert_free(live.offset, live.size);
}

void* HostPinnedArena::base() const noexcept { return base_; }

std::size_t HostPinnedArena::capacity() const noexcept { return cap_; }

std::size_t HostPinnedArena::used() const noexcept { return used_; }

void HostPinnedArena::insert_free(std::size_t offset, std::size_t size) {
    if (size == 0) { return; }
    auto it = free_.begin();
    while (it != free_.end() && it->offset < offset) { ++it; }
    if (it != free_.begin()) {
        auto prev = it - 1;
        if (prev->offset + prev->size == offset) {
            prev->size += size;
            if (it != free_.end() && prev->offset + prev->size == it->offset) {
                prev->size += it->size;
                free_.erase(it);
            }
            return;
        }
    }
    if (it != free_.end() && offset + size == it->offset) {
        it->offset = offset;
        it->size  += size;
        return;
    }
    free_.insert(it, FreeSpan{offset, size});
}

} // namespace ninfer
