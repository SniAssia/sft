// timer.hpp
// Lightweight timing accumulators for the benchmark. C++ side measures batch
// FORMATION cost (scheduler + collation) and the STALL the consumer sees when a
// prefetched pool isn't ready. Python measures scoring/training. Together they
// give formation vs training vs total with overlap efficiency.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>

namespace uds {

using Clock = std::chrono::steady_clock;

inline double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// Nanosecond accumulator + count, thread-safe.
struct TimeStat {
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> count{0};

    void add(double seconds) {
        total_ns.fetch_add(static_cast<uint64_t>(seconds * 1e9), std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }
    double total_s() const { return total_ns.load() / 1e9; }
    uint64_t n() const { return count.load(); }
    double mean_ms() const {
        uint64_t c = count.load();
        return c ? (total_ns.load() / 1e6) / static_cast<double>(c) : 0.0;
    }
};

// RAII scoped timer that adds elapsed to a TimeStat on destruction.
class ScopedTimer {
public:
    explicit ScopedTimer(TimeStat& stat) : stat_(stat), t0_(Clock::now()) {}
    ~ScopedTimer() {
        double dt = std::chrono::duration<double>(Clock::now() - t0_).count();
        stat_.add(dt);
    }
private:
    TimeStat& stat_;
    Clock::time_point t0_;
};

} // namespace uds
