// length_queues.hpp
// Phase 7 — four persistent, thread-safe length-aware queues.
// O(1) insertion by precomputed band; no sorting, no tokenization here.
#pragma once
#include <array>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

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

// Bundles the 4 bands. Routing is a pure array index — the "O(1) insertion".
class LengthAwareQueues {
public:
    void push(Sample&& s) {
        uint32_t b = s.band;
        queues_[b].push(std::move(s));
    }
    SampleQueue& band(uint32_t b) { return queues_[b]; }
    const SampleQueue& band(uint32_t b) const { return queues_[b]; }

    size_t size(uint32_t b) const { return queues_[b].size(); }
    size_t total() const {
        size_t t = 0; for (auto& q : queues_) t += q.size(); return t;
    }
    void close_all() { for (auto& q : queues_) q.close(); }

private:
    std::array<SampleQueue, NUM_BANDS> queues_;
};

} // namespace uds
