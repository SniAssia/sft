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
//
// A BASELINE mode (baseline=true) ignores categories: it draws B random samples
// across all fit bands and lets the collator pad to the batch max — the "previous
// method" used for the padding / proxy-time comparison.
#pragma once
#include <array>
#include <atomic>
#include <random>
#include <vector>

#include "length_queues.hpp"
#include "sample.hpp"

namespace uds {

// A category: two bands filled at `mix`. Single-band (e.g. Chunked) sets both
// bands equal and mix = {1,0}.
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

    // a category is dry when every band's queue is empty AND exhausted
    bool category_dry_(const BatchProfile& cat) const {
        auto band_dry = [&](int b) {
            return queues_.size(static_cast<uint32_t>(b)) == 0 &&
                   queues_.is_exhausted(static_cast<uint32_t>(b));
        };
        if (cat.is_chunked || cat.bands[0] == cat.bands[1]) return band_dry(cat.bands[0]);
        return band_dry(cat.bands[0]) && band_dry(cat.bands[1]);
    }
    bool all_categories_dry() const {
        for (const auto& cat : cfg_.profiles)
            if (!category_dry_(cat)) return false;
        return true;
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
        while (pool.samples.size() < cfg_.B && safety < cfg_.B * NUM_BANDS) {
            bool any = false;
            for (uint32_t b = 0; b < NUM_BANDS && pool.samples.size() < cfg_.B; ++b) {
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

    std::array<std::atomic<uint64_t>, NUM_BANDS> empty_alerts_{};
    std::atomic<uint64_t> fallback_pools_{0};
    std::atomic<uint64_t> skipped_{0};
};

} // namespace uds
