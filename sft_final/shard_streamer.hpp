// shard_streamer.hpp
// Phases 5/6/10 — background workers that: shuffle shard order (per epoch),
// load shards, shuffle samples inside each shard, and push into the length
// queues. Bounded occupancy provides backpressure so RAM stays flat.
#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "distributed.hpp"
#include "length_queues.hpp"
#include "shard_reader.hpp"

namespace uds {

struct StreamerConfig {
    std::vector<std::string> shards;   // ALL shard paths (rank subset chosen inside)
    DistConfig dist;
    size_t max_queue_occupancy = 50000; // per-band soft cap (backpressure)
    int num_epochs = -1;                // -1 => stream forever until stop()
    uint64_t shuffle_seed = 2024;
};

class ShardStreamer {
public:
    explicit ShardStreamer(StreamerConfig cfg, LengthAwareQueues& queues)
        : cfg_(std::move(cfg)), queues_(queues) {}

    ~ShardStreamer() { stop(); }

    void start() {
        running_ = true;
        worker_ = std::thread([this] { loop_(); });
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    bool done() const { return done_.load(); }
    uint64_t samples_streamed() const { return streamed_.load(); }

private:
    void loop_() {
        std::mt19937_64 rng(cfg_.shuffle_seed ^ (0x100000001b3ULL * (cfg_.dist.rank + 1)));
        int epoch = 0;
        while (running_ && (cfg_.num_epochs < 0 || epoch < cfg_.num_epochs)) {
            auto my_shards = assign_shards(cfg_.shards, cfg_.dist, epoch);
            for (const auto& path : my_shards) {
                if (!running_) break;
                stream_one_shard_(path, rng);
            }
            ++epoch;
        }
        done_ = true;
    }

    void stream_one_shard_(const std::string& path, std::mt19937_64& rng) {
        ShardReader reader(path);
        const uint32_t n = reader.num_samples();
        std::vector<uint32_t> order(n);
        for (uint32_t i = 0; i < n; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);   // intra-shard shuffle

        for (uint32_t idx : order) {
            if (!running_) return;
            // backpressure: block while the target band is saturated
            uint32_t band = reader.band_of(idx);
            while (running_ && queues_.size(band) >= cfg_.max_queue_occupancy)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!running_) return;
            queues_.push(reader.get(idx));
            streamed_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    StreamerConfig cfg_;
    LengthAwareQueues& queues_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> done_{false};
    std::atomic<uint64_t> streamed_{0};
};

} // namespace uds
