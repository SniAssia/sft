// batch_scheduler.hpp
// Phase 8 — round-robin category scheduler with exhaustion-aware fallback.
//
// One category per batch (never mixed across categories); categories rotate
// round-robin across successive batches. Within a category the two bands are
// filled at the configured split (e.g. CAT1 = 40% Short / 60% Medium).
//
// Empty-band rule (no waiting):
//   * band queue empty & NOT exhausted -> fill the rest from the other band now,
//     AND raise hungry[b] so the streamer refills it in the background (next time
//     this category comes round it uses the configured split again).
//   * band queue empty & exhausted     -> fill the rest from the other band, no
//     signal (nothing more is coming from the resident shards).
// Category cascade:
//   * both bands of the current category dry (empty & exhausted) -> skip to the
//     next category; if all categories dry -> return an empty pool (the caller
//     waits for the next resident window, or ends the epoch if the streamer is
//     done).
//   The empty-band rule above deliberately treats "empty but NOT exhausted" as
//   "more is coming" so it can signal a refill. That is correct mid-stream but
//   fatal at end of stream: if the streamer has finished but a band's exhausted
//   flag is stale, that band reads as non-dry forever, all_categories_dry()
//   never becomes true, and the consumer spins (skipped_ climbs, no pool ever
//   emitted). Once the streamer signals it is DONE, there is no "more coming":
//   an empty band is dry, full stop. notify_stream_done() flips the scheduler
//   into that terminal interpretation so the tail drains 100% from whatever is
//   left and the epoch can actually end.
#pragma once
#include <array>
#include <atomic>
#include <random>
#include <vector>

#include "length_queues.hpp"
#include "sample.hpp"

namespace uds {

// A category: two bands filled at `mix`. Single-band (e.g. Chunked) sets both
struct BatchProfile {
    std::array<int, 2>   bands = {BAND_SHORT, BAND_MEDIUM};
    std::array<float, 2> mix   = {0.5f, 0.5f};
    bool is_chunked = false;
};

struct SchedulerConfig {
    size_t B = 64;
    std::vector<BatchProfile> profiles;   // categories in round-robin order
    bool baseline = false;                // true => random length-agnostic batching
    uint64_t seed = 777;
};

class BatchScheduler {
public:
    BatchScheduler(SchedulerConfig cfg, LengthAwareQueues& queues)
        : cfg_(std::move(cfg)), queues_(queues), rng_(cfg_.seed) {
        if (cfg_.profiles.empty()) install_default_profiles_();
    }

    CandidatePool next_pool() {
        return cfg_.baseline ? baseline_pool_() : round_robin_pool_();
    }

    uint64_t empty_alerts(uint32_t band) const {
        return empty_alerts_[band].load(std::memory_order_relaxed);
    }
    uint64_t fallback_pools() const { return fallback_pools_.load(std::memory_order_relaxed); }
    uint64_t skipped_categories() const { return skipped_.load(std::memory_order_relaxed); }

    // --- termination support ---------------------------------------------------
    // Called by the prefetcher once the streamer reports done(). After this the
    // scheduler treats any empty band as dry regardless of its exhausted flag, so
    // the residual tail drains 100% from whatever remains and all_categories_dry()
    // can finally become true. Idempotent.
    void notify_stream_done() { stream_done_.store(true, std::memory_order_relaxed); }
    bool stream_done() const { return stream_done_.load(std::memory_order_relaxed); }

    // Total samples still sitting across all band queues. The prefetcher uses
    // this as the authoritative "is there anything left to form" signal — it is
    // immune to any single stale exhausted flag.
    size_t queues_total() const { return queues_.total(); }

    // a category is dry when every band it uses is empty (and, mid-stream, also
    // exhausted). Public so the prefetcher can gate epoch-end on it.
    bool all_categories_dry() const {
        for (const auto& cat : cfg_.profiles)
            if (!category_dry_(cat)) return false;
        return true;
    }

private:
    // ---------------- round-robin category path ----------------
    CandidatePool round_robin_pool_() {
        const int nc = static_cast<int>(cfg_.profiles.size());
        for (int tries = 0; tries < nc; ++tries) {
            int ci = rr_index_;
            rr_index_ = (rr_index_ + 1) % nc;           // advance for next call
            const BatchProfile& cat = cfg_.profiles[ci];

            if (category_dry_(cat)) { skipped_.fetch_add(1, std::memory_order_relaxed); continue; }

            CandidatePool pool = fill_category_(cat, ci);
            if (!pool.samples.empty()) return pool;
            // category looked non-dry but produced nothing (transient) -> try next
        }
        return CandidatePool{};   // all categories dry/empty this instant
    }

    // A band is "dry" when its queue is empty AND either it is exhausted (no more
    // from the resident shards) OR the streamer is done (no more, ever). The
    // stream_done_ clause is what lets the drain tail terminate.
    bool band_dry_(int b) const {
        const uint32_t ub = static_cast<uint32_t>(b);
        return queues_.size(ub) == 0 &&
               (queues_.is_exhausted(ub) || stream_done_.load(std::memory_order_relaxed));
    }

    bool category_dry_(const BatchProfile& cat) const {
        if (cat.is_chunked || cat.bands[0] == cat.bands[1]) return band_dry_(cat.bands[0]);
        return band_dry_(cat.bands[0]) && band_dry_(cat.bands[1]);
    }

    CandidatePool fill_category_(const BatchProfile& cat, int ci) {
        CandidatePool pool;
        pool.profile_index = ci;
        pool.profile_bands = cat.bands;
        pool.is_chunked = cat.is_chunked;
        pool.mixed = !cat.is_chunked && cat.bands[0] != cat.bands[1];
        pool.band = static_cast<uint32_t>(cat.bands[0]);

        // single-band (chunked) category
        if (cat.is_chunked || cat.bands[0] == cat.bands[1]) {
            drain_(static_cast<uint32_t>(cat.bands[0]), cfg_.B, pool);
            return pool;
        }

        size_t t0 = static_cast<size_t>(cat.mix[0] * cfg_.B + 0.5f);
        if (t0 > cfg_.B) t0 = cfg_.B;
        size_t t1 = cfg_.B - t0;

        uint32_t b0 = static_cast<uint32_t>(cat.bands[0]);
        uint32_t b1 = static_cast<uint32_t>(cat.bands[1]);

        size_t got0 = drain_(b0, t0, pool);
        size_t got1 = drain_(b1, t1, pool);

        // fallback: fill the remainder from the other band (no waiting)
        size_t remaining = cfg_.B - pool.samples.size();
        if (remaining > 0) {
            if (got0 < t0 && queues_.size(b1) > 0) {          // band0 short -> use band1
                if (!queues_.is_exhausted(b0)) queues_.set_hungry(b0);  // signal refill
                empty_alerts_[b0].fetch_add(1, std::memory_order_relaxed);
                if (drain_(b1, remaining, pool) > 0) pool.fell_back = true;
            } else if (got1 < t1 && queues_.size(b0) > 0) {   // band1 short -> use band0
                if (!queues_.is_exhausted(b1)) queues_.set_hungry(b1);
                empty_alerts_[b1].fetch_add(1, std::memory_order_relaxed);
                if (drain_(b0, remaining, pool) > 0) pool.fell_back = true;
            }
        }
        if (pool.fell_back) fallback_pools_.fetch_add(1, std::memory_order_relaxed);
        return pool;
    }

    // non-blocking drain of up to `count` from a band
    size_t drain_(uint32_t band, size_t count, CandidatePool& pool) {
        size_t taken = 0;
        while (taken < count) {
            auto s = queues_.band(band).try_pop();
            if (!s) break;
            pool.samples.push_back(std::move(*s));
            ++taken;
        }
        return taken;
    }

    // ---------------- baseline (length-agnostic) path ----------------
    CandidatePool baseline_pool_() {
        CandidatePool pool;
        pool.baseline = true;
        pool.mixed = true;
        // round-robin across all bands, taking one at a time until B or all empty
        size_t safety = 0;
        while (pool.samples.size() < cfg_.B && safety < cfg_.B * queues_.num_bands()) {
            bool any = false;
            for (uint32_t b = 0; b < queues_.num_bands() && pool.samples.size() < cfg_.B; ++b) {
                auto s = queues_.band(b).try_pop();
                if (s) { pool.samples.push_back(std::move(*s)); any = true; }
            }
            if (!any) break;
            ++safety;
        }
        return pool;
    }

    void install_default_profiles_() {
        // CAT1 Short40/Medium60, CAT2 Medium45/Long55, CAT3 Chunked
        cfg_.profiles = {
            BatchProfile{ {BAND_SHORT,  BAND_MEDIUM}, {0.40f, 0.60f}, false },
            BatchProfile{ {BAND_MEDIUM, BAND_LONG},   {0.45f, 0.55f}, false },
            BatchProfile{ {BAND_CHUNKED,BAND_CHUNKED},{1.00f, 0.00f}, true  },
        };
    }

    SchedulerConfig cfg_;
    LengthAwareQueues& queues_;
    std::mt19937_64 rng_;
    int rr_index_ = 0;

    std::atomic<bool> stream_done_{false};

    std::array<std::atomic<uint64_t>, MAX_BANDS> empty_alerts_{};
    std::atomic<uint64_t> fallback_pools_{0};
    std::atomic<uint64_t> skipped_{0};
};

} // namespace uds