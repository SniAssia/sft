// shard_streamer.hpp
// Phases 5/6/10 — background streaming with a RESIDENT WINDOW.
//
// The streamer keeps a window of W shards "resident". For that window it knows,
// per band, how many samples remain to be pushed (pending[b]). When pending[b]
// hits 0 it marks band b EXHAUSTED, so the scheduler can tell "temporarily empty,
// more coming" from "dry for the loaded shards".
//
// LOAD-TIME BANDS:
//   The band is NOT read from the shard index. It is computed here from the
//   record's total_len + the cutoffs in StreamerConfig, so changing cutoffs
//   never requires rebuilding shards. Band count is runtime (band_cutoffs.size()
//   + 2); arrays are sized MAX_BANDS, loops bound on queues_.num_bands().
//
// TERMINATION FIX:
//   Once loop_() exits nothing more will ever be pushed, so every band is
//   exhausted by definition. We set that explicitly before publishing done_.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "distributed.hpp"
#include "length_queues.hpp"
#include "shard_reader.hpp"
#include "shard_format.hpp"

namespace uds {

struct StreamerConfig {
    std::vector<std::string> shards;    // ALL shard paths (rank subset chosen inside)
    DistConfig dist;
    size_t max_queue_occupancy = 50000; // per-band soft cap (backpressure)
    int num_epochs = -1;                // -1 => stream forever until stop()
    uint64_t shuffle_seed = 2024;
    int resident_window = 4;            // W: shards held resident together

    // --- load-time band config ---
    std::vector<uint32_t> band_cutoffs{512, 1536};  // k-1 internal cutoffs, ascending
    uint32_t max_seq_len = 2048;                    // > this => CHUNKED (last band)
};

class ShardStreamer {
public:
    explicit ShardStreamer(StreamerConfig cfg, LengthAwareQueues& queues)
        : cfg_(std::move(cfg)), queues_(queues) {
        if (cfg_.resident_window < 1) cfg_.resident_window = 1;
    }

    ~ShardStreamer() { stop(); }

    void start() { running_ = true; worker_ = std::thread([this] { loop_(); }); }
    void stop() { running_ = false; if (worker_.joinable()) worker_.join(); }

    bool done() const { return done_.load(); }
    uint64_t samples_streamed() const { return streamed_.load(); }

private:
    struct WinItem { uint32_t shard; uint32_t idx; uint32_t band; };

    void loop_() {
        std::mt19937_64 rng(cfg_.shuffle_seed ^ (0x100000001b3ULL * (cfg_.dist.rank + 1)));
        int epoch = 0;
        while (running_ && (cfg_.num_epochs < 0 || epoch < cfg_.num_epochs)) {
            auto my_shards = assign_shards(cfg_.shards, cfg_.dist, epoch);
            const int W = cfg_.resident_window;
            for (size_t start = 0; start < my_shards.size() && running_; start += W) {
                size_t end = std::min(start + static_cast<size_t>(W), my_shards.size());
                std::vector<std::string> window(my_shards.begin() + start,
                                                my_shards.begin() + end);
                stream_window_(window, rng);
            }
            ++epoch;
        }
        // Nothing more will ever be pushed => every band is exhausted.
        // Published BEFORE done_ so any consumer seeing done_ also sees exhausted.
        for (uint32_t b = 0; b < queues_.num_bands(); ++b) queues_.set_exhausted(b);
        done_.store(true, std::memory_order_release);
    }

    void stream_window_(const std::vector<std::string>& window, std::mt19937_64& rng) {
        std::vector<std::unique_ptr<ShardReader>> readers;
        readers.reserve(window.size());
        for (const auto& p : window) readers.push_back(std::make_unique<ShardReader>(p));

        std::vector<WinItem> items;
        std::array<uint64_t, MAX_BANDS> pending{};   // sized by ceiling, used to num_bands()
        for (uint32_t s = 0; s < readers.size(); ++s) {
            const uint32_t n = readers[s]->num_samples();
            for (uint32_t i = 0; i < n; ++i) {
                // band computed at LOAD time from length + config cutoffs
                uint32_t tl  = readers[s]->total_len_of(i);
                bool     isk = readers[s]->is_chunked_of(i) != 0;
                uint32_t b   = band_from_len(tl, isk, cfg_.band_cutoffs, cfg_.max_seq_len);
                items.push_back({s, i, b});
                pending[b]++;
            }
        }
        std::shuffle(items.begin(), items.end(), rng);   // intra-window shuffle

        // A band present in this window is NOT exhausted; a band absent IS.
        for (uint32_t b = 0; b < queues_.num_bands(); ++b) {
            if (pending[b] > 0) queues_.clear_exhausted(b);
            else queues_.set_exhausted(b);
        }

        for (const auto& it : items) {
            if (!running_) return;
            while (running_ && queues_.size(it.band) >= cfg_.max_queue_occupancy
                   && !any_hungry_())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!running_) return;
            Sample smp = readers[it.shard]->get(it.idx);
            smp.band = it.band;                  // load-time band wins over stored one
            queues_.push(std::move(smp));
            queues_.clear_hungry(it.band);
            if (--pending[it.band] == 0) queues_.set_exhausted(it.band);
            streamed_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool any_hungry_() const {
        for (uint32_t b = 0; b < queues_.num_bands(); ++b)
            if (queues_.is_hungry(b)) return true;
        return false;
    }

    StreamerConfig cfg_;
    LengthAwareQueues& queues_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> done_{false};
    std::atomic<uint64_t> streamed_{0};
};

} // namespace uds