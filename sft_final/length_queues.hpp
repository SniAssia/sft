// Phase 7 — four persistent, thread-safe length-aware queues.
// O(1) insertion by precomputed band; no sorting, no tokenization here.
#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <cassert>

#include "sample.hpp"
#include "shard_format.hpp"

namespace uds {

class SampleQueue {
public:
    void push(Sample&& s) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push_back(std::move(s));
        }
        cv_.notify_one();
    }

    // Non-blocking.
    std::optional<Sample> try_pop() {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return std::nullopt;
        Sample s = std::move(q_.front());
        q_.pop_front();
        return s;
    }

    // Blocking with timeout; returns nullopt on timeout or when closed+empty.
    std::optional<Sample> pop_wait(int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                     [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        Sample s = std::move(q_.front());
        q_.pop_front();
        return s;
    }

    size_t size() const { std::lock_guard<std::mutex> lk(m_); return q_.size(); }
    void close() { { std::lock_guard<std::mutex> lk(m_); closed_ = true; } cv_.notify_all(); }
    bool closed() const { std::lock_guard<std::mutex> lk(m_); return closed_; }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Sample> q_;
    bool closed_ = false;
};

// Bundles the bands. Routing is a pure array index — the "O(1) insertion".
// Arrays are sized MAX_BANDS (compile time); only num_bands_ of them are used.
class LengthAwareQueues {
public:
    explicit LengthAwareQueues(uint32_t num_bands = 4) : num_bands_(num_bands) {
        assert(num_bands >= 1 && num_bands <= static_cast<uint32_t>(MAX_BANDS));
    }
    uint32_t num_bands() const { return num_bands_; }

    void push(Sample&& s) {
        uint32_t b = s.band;
        queues_[b].push(std::move(s));
    }
    SampleQueue& band(uint32_t b) { return queues_[b]; }
    const SampleQueue& band(uint32_t b) const { return queues_[b]; }

    size_t size(uint32_t b) const { return queues_[b].size(); }
    size_t total() const {
        size_t t = 0;
        for (uint32_t b = 0; b < num_bands_; ++b) t += queues_[b].size();
        return t;
    }
    void close_all() { for (uint32_t b = 0; b < num_bands_; ++b) queues_[b].close(); }

    // --- Option-A demand signaling (scheduler -> streamer) ---
    // Scheduler sets a band "hungry" when it finds that queue empty; the streamer
    // reads these flags and preferentially pushes samples of hungry bands. Cleared
    // by the streamer once it has fed that band.
    void set_hungry(uint32_t b) { hungry_[b].store(true, std::memory_order_relaxed); }
    void clear_hungry(uint32_t b) { hungry_[b].store(false, std::memory_order_relaxed); }
    bool is_hungry(uint32_t b) const { return hungry_[b].load(std::memory_order_relaxed); }

    // --- Resident-window exhaustion (streamer -> scheduler) ---
    // exhausted[b] == true means the shards currently resident in memory hold NO
    // more band-b samples to push. Combined with an empty queue, the scheduler
    // treats band b as truly dry (fill 100% from the other band, no refill signal).
    // Cleared when the streamer loads a new window that contains band-b samples.
    void set_exhausted(uint32_t b) { exhausted_[b].store(true, std::memory_order_relaxed); }
    void clear_exhausted(uint32_t b) { exhausted_[b].store(false, std::memory_order_relaxed); }
    bool is_exhausted(uint32_t b) const { return exhausted_[b].load(std::memory_order_relaxed); }

private:
    uint32_t num_bands_;
    std::array<SampleQueue, MAX_BANDS> queues_;
    std::array<std::atomic<bool>, MAX_BANDS> hungry_{};     // all false initially
    std::array<std::atomic<bool>, MAX_BANDS> exhausted_{};  // resident-window dry
};

} // namespace uds