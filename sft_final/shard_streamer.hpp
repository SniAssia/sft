// shard_streamer.hpp
// Phases 5/6/10 — background streaming with a RESIDENT WINDOW.
//
// Instead of one shard at a time, the streamer keeps a window of W shards
// "resident". For that window it knows, per band, how many samples remain to be
// pushed (pending[b]). When pending[b] hits 0 it marks band b EXHAUSTED on the
// queues, so the scheduler can tell "temporarily empty, more coming" (exhausted
// == false -> signal a refill) from "dry for the loaded shards" (exhausted ==
// true -> fall back 100% to the other band). When the window is fully streamed,
// the next W shards load and exhaustion flags reset for bands they contain.
//
// TERMINATION FIX:
//   The per-window exhaustion bookkeeping is a *refill* hint, not a reliable
//   end-of-stream signal — a band can end up empty-but-not-exhausted at true
//   end of stream (e.g. a band cleared at the start of the final window whose
//   last set_exhausted() races with consumption, or a rank whose shard set is
//   empty so stream_window_ never runs). That single stale flag keeps
//   all_categories_dry() false forever and the epoch never ends.
//   Once loop_() exits, nothing more will EVER be pushed for any band, so by
//   definition every band is exhausted. We set that explicitly before done_.
#pragma once
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

namespace uds {

struct StreamerConfig {
    std::vector<std::string> shards;    // ALL shard paths (rank subset chosen inside)
    DistConfig dist;
    size_t max_queue_occupancy = 50000; // per-band soft cap (backpressure)
    int num_epochs = -1;                // -1 => stream forever until stop()
    uint64_t shuffle_seed = 2024;
    int resident_window = 4;            // W: shards held resident together
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
    // one (shard, local index, band) entry in the resident window
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

        // End of stream (all epochs done, or stopped). Nothing more will ever be
        // pushed, so every band is exhausted by definition. Mark them all so the
        // scheduler's "empty && exhausted" dryness test can reach a terminal
        // state even if a per-window flag was missed or a band never appeared.
        // done_ is published AFTER the flags so any consumer that observes
        // done_ == true also observes exhausted == true.
        for (uint32_t b = 0; b < NUM_BANDS; ++b) queues_.set_exhausted(b);
        done_.store(true, std::memory_order_release);
    }

    void stream_window_(const std::vector<std::string>& window, std::mt19937_64& rng) {
        // Open readers, build the combined resident item list + per-band pending.
        std::vector<std::unique_ptr<ShardReader>> readers;
        readers.reserve(window.size());
        for (const auto& p : window) readers.push_back(std::make_unique<ShardReader>(p));

        std::vector<WinItem> items;
        std::array<uint64_t, NUM_BANDS> pending{};
        for (uint32_t s = 0; s < readers.size(); ++s) {
            const uint32_t n = readers[s]->num_samples();
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t b = readers[s]->band_of(i);
                items.push_back({s, i, b});
                pending[b]++;
            }
        }
        std::shuffle(items.begin(), items.end(), rng);   // intra-window shuffle

        // A band present in this window is NOT exhausted; a band absent IS.
        for (uint32_t b = 0; b < NUM_BANDS; ++b) {
            if (pending[b] > 0) queues_.clear_exhausted(b);
            else queues_.set_exhausted(b);
        }

        // Stream items; decrement pending; mark exhausted when a band hits 0.
        for (const auto& it : items) {
            if (!running_) return;
            while (running_ && queues_.size(it.band) >= cfg_.max_queue_occupancy
                   && !any_hungry_())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!running_) return;
            queues_.push(readers[it.shard]->get(it.idx));
            queues_.clear_hungry(it.band);
            if (--pending[it.band] == 0) queues_.set_exhausted(it.band);
            streamed_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool any_hungry_() const {
        for (uint32_t b = 0; b < NUM_BANDS; ++b)
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