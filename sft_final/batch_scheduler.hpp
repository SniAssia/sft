// batch_scheduler.hpp
// Phase 8 — the configurable batch scheduler. Emits UDS candidate pools of
// size B. Two modes:
//   HOMOGENEOUS (default): each pool is drawn from ONE fit band (chosen by
//       weighted random over non-empty bands) so downstream SVD/collation sees
//       minimal padding, while content stays diverse.
//   MIXED: pool composed across fit bands per configured ratios (the 30/70 knob) —
//       more padding, full control of composition.
// Chunked samples are ALWAYS pooled separately (Option-B path), never padded
// into a fit pool. Chunked pools are emitted at a configured rate.
#pragma once
#include <random>
#include <vector>

#include "length_queues.hpp"
#include "sample.hpp"

namespace uds {

enum class PoolMode { Homogeneous, Mixed };

struct SchedulerConfig {
    size_t B = 64;                          // candidate pool (oversampling) size, B > K
    PoolMode mode = PoolMode::Homogeneous;
    // weights over fit bands [SHORT, MEDIUM, LONG]; chunked handled separately
    std::array<float, 3> fit_band_weights = {1.f, 1.f, 1.f};
    float chunked_rate = 0.1f;              // fraction of pools drawn from CHUNKED
    int pop_timeout_ms = 50;
    uint64_t seed = 777;
};

class BatchScheduler {
public:
    BatchScheduler(SchedulerConfig cfg, LengthAwareQueues& queues)
        : cfg_(cfg), queues_(queues), rng_(cfg.seed) {}

    // Blocks (bounded) until a full pool of size B is assembled, or returns a
    // short/empty pool if the queues are draining and cannot fill one.
    CandidatePool next_pool() {
        if (want_chunked_()) {
            CandidatePool p = drain_band_(BAND_CHUNKED, cfg_.B);
            if (!p.samples.empty()) { p.is_chunked = true; p.band = BAND_CHUNKED; return p; }
            // fall through to fit pool if no chunked samples available
        }
        return cfg_.mode == PoolMode::Homogeneous ? homogeneous_pool_() : mixed_pool_();
    }

private:
    bool want_chunked_() {
        if (cfg_.chunked_rate <= 0.f) return false;
        std::uniform_real_distribution<float> u(0.f, 1.f);
        return u(rng_) < cfg_.chunked_rate && queues_.size(BAND_CHUNKED) > 0;
    }

    // Pick one non-empty fit band by weight; fill the whole pool from it.
    CandidatePool homogeneous_pool_() {
        int band = pick_fit_band_();
        CandidatePool p;
        if (band < 0) return p;               // all empty
        p = drain_band_(static_cast<uint32_t>(band), cfg_.B);
        p.band = static_cast<uint32_t>(band);
        p.is_chunked = false;
        return p;
    }

    // Compose across fit bands per weights (normalized to counts summing to B).
    CandidatePool mixed_pool_() {
        CandidatePool p;
        float wsum = cfg_.fit_band_weights[0] + cfg_.fit_band_weights[1] + cfg_.fit_band_weights[2];
        if (wsum <= 0.f) return p;
        for (int b = 0; b < 3; ++b) {
            size_t take = static_cast<size_t>((cfg_.fit_band_weights[b] / wsum) * cfg_.B);
            CandidatePool part = drain_band_(static_cast<uint32_t>(b), take);
            for (auto& s : part.samples) p.samples.push_back(std::move(s));
        }
        p.band = BAND_MEDIUM;   // heterogeneous; label as MEDIUM for bookkeeping
        p.is_chunked = false;
        return p;
    }

    int pick_fit_band_() {
        float w[3];
        float wsum = 0.f;
        for (int b = 0; b < 3; ++b) {
            bool nonempty = queues_.size(static_cast<uint32_t>(b)) > 0;
            w[b] = nonempty ? cfg_.fit_band_weights[b] : 0.f;
            wsum += w[b];
        }
        if (wsum <= 0.f) return -1;
        std::uniform_real_distribution<float> u(0.f, wsum);
        float r = u(rng_), acc = 0.f;
        for (int b = 0; b < 3; ++b) { acc += w[b]; if (r <= acc) return b; }
        return 2;
    }

    // Pop up to `count` samples from one band. Waits briefly for the streamer.
    CandidatePool drain_band_(uint32_t band, size_t count) {
        CandidatePool p;
        p.samples.reserve(count);
        while (p.samples.size() < count) {
            auto s = queues_.band(band).pop_wait(cfg_.pop_timeout_ms);
            if (!s) break;   // timeout: queue drained for now
            p.samples.push_back(std::move(*s));
        }
        return p;
    }

    SchedulerConfig cfg_;
    LengthAwareQueues& queues_;
    std::mt19937_64 rng_;
};

} // namespace uds
